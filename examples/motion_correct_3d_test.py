"""
3D motion correction test script with synthetic displacement.

This script:
1. Loads a pre-aligned HDF5 video using pyflowreg readers
2. Processes it (resize 50%, crop 25px boundaries, normalize)
3. Creates a second frame with synthetic 3D motion displacements
4. Performs 3D motion correction using compute_flow
5. Visualizes original, displaced, and corrected volumes in napari
"""

import time
from pathlib import Path

import napari
import numpy as np
from pyflowreg.util.io.factory import get_video_file_reader
from scipy.ndimage import zoom, gaussian_filter

from flowreg3d.core.optical_flow_3d import get_displacement, imregister_wrapper
from flowreg3d.motion_generation.motion_generators import (get_default_3d_generator, get_low_disp_3d_generator,
                                                           get_test_3d_generator, get_high_disp_3d_generator)
from flowreg3d.util.random import fix_seed


def process_3d_stack(video_data):
    """
    Process 3D stack: resize 50%, crop 25px boundaries, normalize.

    Args:
        video_data: Input video array (T, Y, X) or (T, Y, X, C)

    Returns:
        Processed video array
    """
    print(f"\nProcessing 3D stack...")
    print(f"  Input shape: {video_data.shape}")

    # Convert to float32 for processing
    video = video_data.astype(np.float32)

    # Step 1: Resize to 50% (0.5x)
    resize_factor = 1
    print(f"  Resizing to {100*resize_factor}%...")
    if video.ndim == 4:  # Has channels
        zoom_factors = (1.0, resize_factor, resize_factor, 1.0)  # Don't resize time or channels
    else:
        zoom_factors = (1.0, resize_factor, resize_factor)  # Don't resize time

    video_resized = zoom(video, zoom_factors, order=1)  # Linear interpolation
    print(f"  After resize: {video_resized.shape}")

    # Step 2: Crop 25 pixels from all XY boundaries
    print("  Cropping 25px from boundaries...")
    if video_resized.shape[1] > 50 and video_resized.shape[2] > 50:
        if video_resized.ndim == 4:
            video_cropped = video_resized[:, 25:-25, 25:-25, :]
        else:
            video_cropped = video_resized[:, 25:-25, 25:-25]
    else:
        print("  Warning: Video too small to crop, skipping crop step")
        video_cropped = video_resized
    print(f"  After crop: {video_cropped.shape}")

    # Step 3: Normalize to [0, 1]
    print("  Normalizing...")
    vmin = video_cropped.min()
    vmax = video_cropped.max()

    if vmax > vmin:
        video_normalized = (video_cropped - vmin) / (vmax - vmin)
    else:
        video_normalized = video_cropped

    print(f"  Final shape: {video_normalized.shape}")
    print(f"  Value range: [{video_normalized.min():.3f}, {video_normalized.max():.3f}]")

    return video_normalized


def warp_volume_bw3d_torch(volume, flow):
    """
    Backward warp a 3D volume using torch grid_sample (fast GPU/CPU operation).

    Args:
        volume: Input volume (Z, Y, X) or (Z, Y, X, C)
        flow: Flow field (Z, Y, X, 3) where last dim is (dx, dy, dz)

    Returns:
        Warped volume with same shape as input
    """
    import torch

    flow = -flow
    v = torch.from_numpy(volume).float()
    f = torch.from_numpy(flow).float()
    if v.ndim == 3:
        v = v[..., None]
    Z, H, W, C = v.shape

    # Create normalized grid from -1 to 1
    zz = torch.linspace(-1, 1, Z)
    yy = torch.linspace(-1, 1, H)
    xx = torch.linspace(-1, 1, W)
    Zg, Yg, Xg = torch.meshgrid(zz, yy, xx, indexing='ij')

    # Add normalized flow to grid
    nz = Zg + 2 * f[..., 0] / (Z - 1)
    ny = Yg + 2 * f[..., 1] / (H - 1)
    nx = Xg + 2 * f[..., 2] / (W - 1)

    # Stack to create sampling grid
    grid = torch.stack([nx, ny, nz], dim=-1)[None]

    # Reshape volume for grid_sample
    v = v.permute(3, 0, 1, 2)[None]

    # Apply backward warping
    out = torch.nn.functional.grid_sample(v, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Reshape output
    out = out[0].permute(1, 2, 3, 0).numpy()

    return out[..., 0] if out.shape[-1] == 1 else out


def warp_volume_splat3d(volume, flow):
    """
    Forward warp a 3D volume using splatting (fast scatter operation).

    Args:
        volume: Input volume (Z, Y, X) or (Z, Y, X, C)
        flow: Flow field (Z, Y, X, 3) where last dim is (dx, dy, dz)

    Returns:
        Warped volume with same shape as input
    """
    Z, H, W = volume.shape[:3]
    z, y, x = np.meshgrid(np.arange(Z), np.arange(H), np.arange(W), indexing='ij')

    # Target coordinates (flow is [dx, dy, dz])
    tx = (x + flow[..., 0]).ravel()  # dx
    ty = (y + flow[..., 1]).ravel()  # dy
    tz = (z + flow[..., 2]).ravel()  # dz

    # Bilinear interpolation weights
    iz = np.floor(tz).astype(np.int64)
    fz = tz - iz
    iy = np.floor(ty).astype(np.int64)
    fy = ty - iy
    ix = np.floor(tx).astype(np.int64)
    fx = tx - ix

    # Clamp indices to bounds
    iz0 = np.clip(iz, 0, Z - 1)
    iz1 = np.clip(iz + 1, 0, Z - 1)
    iy0 = np.clip(iy, 0, H - 1)
    iy1 = np.clip(iy + 1, 0, H - 1)
    ix0 = np.clip(ix, 0, W - 1)
    ix1 = np.clip(ix + 1, 0, W - 1)

    # Trilinear interpolation weights
    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w100 = fx * (1 - fy) * (1 - fz)
    w010 = (1 - fx) * fy * (1 - fz)
    w110 = fx * fy * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * fz
    w101 = fx * (1 - fy) * fz
    w011 = (1 - fx) * fy * fz
    w111 = fx * fy * fz

    def accum(values):
        V = values.ravel()
        idx = lambda zz, yy, xx: (zz * H + yy) * W + xx
        N = Z * H * W
        out = np.zeros(N, dtype=np.float64)
        den = np.zeros(N, dtype=np.float64)

        # Splat to 8 neighboring voxels
        for w, zz, yy, xx in [(w000, iz0, iy0, ix0), (w100, iz0, iy0, ix1), (w010, iz0, iy1, ix0),
                              (w110, iz0, iy1, ix1), (w001, iz1, iy0, ix0), (w101, iz1, iy0, ix1),
                              (w011, iz1, iy1, ix0), (w111, iz1, iy1, ix1)]:
            idv = idx(zz, yy, xx)
            np.add.at(out, idv, V * w)
            np.add.at(den, idv, w)

        # Normalize by weights
        den[den == 0] = 1.0
        return (out / den).reshape(Z, H, W).astype(values.dtype)

    if volume.ndim == 4:
        C = volume.shape[3]
        return np.stack([accum(volume[..., c]) for c in range(C)], axis=-1)
    return accum(volume)


def warp_volume_pc3d(volume, flow):
    """
    Forward warp a 3D volume using griddata (scatter operation).

    Args:
        volume: Input volume (Z, Y, X) or (Z, Y, X, C)
        flow: Flow field (Z, Y, X, 3) where last dim is (dx, dy, dz)

    Returns:
        Warped volume with same shape as input
    """
    from scipy.interpolate import griddata

    Z, H, W = volume.shape[:3]

    # Create original grid coordinates
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(Z, dtype=np.float32), np.arange(H, dtype=np.float32),
                                         np.arange(W, dtype=np.float32), indexing='ij')

    # Compute target coordinates (where each pixel moves TO)
    target_x = grid_x + flow[:, :, :, 0]  # dx component
    target_y = grid_y + flow[:, :, :, 1]  # dy component
    target_z = grid_z + flow[:, :, :, 2]  # dz component

    # Forward warp: scatter source pixels to target locations
    if volume.ndim == 4:  # Has channels
        warped = np.zeros_like(volume)
        for c in range(volume.shape[3]):
            # Flatten arrays for griddata
            source_points = np.column_stack([target_x.flatten(), target_y.flatten(), target_z.flatten()])
            source_values = volume[:, :, :, c].flatten()
            target_points = np.column_stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])

            # Use griddata for forward warping (scatter operation)
            warped[:, :, :, c] = griddata(source_points, source_values, target_points, method='linear',
                                          fill_value=0).reshape(Z, H, W)
    else:
        # Flatten arrays for griddata
        source_points = np.column_stack([target_x.flatten(), target_y.flatten(), target_z.flatten()])
        source_values = volume.flatten()
        target_points = np.column_stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])

        # Use griddata for forward warping (scatter operation)
        warped = griddata(source_points, source_values, target_points, method='linear', fill_value=0).reshape(Z, H, W)

    return warped


def compute_3d_optical_flow(frame1, frame2, flow_params):
    """
    Compute 3D optical flow between two frames.
    Handles preprocessing internally.

    Args:
        frame1: Reference frame (Z, Y, X) or (Z, Y, X, C)
        frame2: Target frame (Z, Y, X) or (Z, Y, X, C)
        flow_params: Dictionary of flow parameters

    Returns:
        Flow field with shape (Z, Y, X, 3) where last dimension contains [dx, dy, dz]
    """
    print("\nComputing 3D optical flow...")
    print(f"  Input shapes: {frame1.shape}, {frame2.shape}")

    # Ensure float32
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)

    if f1.ndim == 3:  # No channels, add channel dimension
        f1 = f1[..., np.newaxis]
        f2 = f2[..., np.newaxis]

    # Apply Gaussian filtering with sigma=2 in each spatial dimension
    print("  Applying Gaussian filter (sigma=2)...")
    for c in range(f1.shape[-1]):
        f1[..., c] = gaussian_filter(f1[..., c], sigma=0.5)
        f2[..., c] = gaussian_filter(f2[..., c], sigma=0.5)

    # Normalize per channel based on frame1 statistics
    print("  Normalizing frames...")
    mins = f1.min(axis=(0, 1, 2), keepdims=True)
    maxs = f1.max(axis=(0, 1, 2), keepdims=True)
    ranges = maxs - mins

    # Avoid division by zero
    ranges = np.where(ranges > 0, ranges, 1.0)

    f1_norm = (f1 - mins) / ranges
    f2_norm = (f2 - mins) / ranges

    t0 = time.perf_counter()

    # Call get_displacement which returns (Z, Y, X, 3) with (dx, dy, dz)
    flow = get_displacement(f1_norm, f2_norm, **flow_params)

    t_elapsed = time.perf_counter() - t0
    print(f"  Flow computation time: {t_elapsed:.2f} seconds")

    # Print flow statistics
    print(f"  Flow field shape: {flow.shape}")
    print(f"  Flow magnitude stats:")
    print(
        f"    dx: min={flow[:, :, :, 0].min():.2f}, max={flow[:, :, :, 0].max():.2f}, mean={flow[:, :, :, 0].mean():.2f}")
    print(
        f"    dy: min={flow[:, :, :, 1].min():.2f}, max={flow[:, :, :, 1].max():.2f}, mean={flow[:, :, :, 1].mean():.2f}")
    print(
        f"    dz: min={flow[:, :, :, 2].min():.2f}, max={flow[:, :, :, 2].max():.2f}, mean={flow[:, :, :, 2].mean():.2f}")

    total_magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
    print(
        f"  Total magnitude: min={total_magnitude.min():.2f}, max={total_magnitude.max():.2f}, mean={total_magnitude.mean():.2f}")

    return flow


def create_displaced_frame_with_generator(video, generator_type='high_disp'):
    """
    Create a second 3D frame with synthetic motion displacements.

    Args:
        video: Input video array (Z, Y, X) or (Z, Y, X, C)
        generator_type: Type of motion generator ('default', 'low_disp', 'test', 'high_disp')

    Returns:
        Tuple of (displaced_video, ground_truth_flow)
    """
    print(f"\nCreating displaced frame with synthetic motion...")
    print(f"  Generator type: {generator_type}")

    depth, height, width = video.shape[:3]

    # Select generator
    if generator_type == 'default':
        generator = get_default_3d_generator()
    elif generator_type == 'low_disp':
        generator = get_low_disp_3d_generator()
    elif generator_type == 'test':
        generator = get_test_3d_generator()
    elif generator_type == 'high_disp':
        generator = get_high_disp_3d_generator()
    else:
        print(f"  Unknown generator type, using high_disp")
        generator = get_high_disp_3d_generator()

    # Generate flow field
    flow_gt, invalid_mask = generator(depth=depth, height=height, width=width)

    # Print flow statistics
    print(f"  Ground truth flow shape: {flow_gt.shape}")
    print(f"  Flow magnitude stats:")
    print(
        f"    dx: min={flow_gt[:, :, :, 0].min():.2f}, max={flow_gt[:, :, :, 0].max():.2f}, mean={flow_gt[:, :, :, 0].mean():.2f}")
    print(
        f"    dy: min={flow_gt[:, :, :, 1].min():.2f}, max={flow_gt[:, :, :, 1].max():.2f}, mean={flow_gt[:, :, :, 1].mean():.2f}")
    print(
        f"    dz: min={flow_gt[:, :, :, 2].min():.2f}, max={flow_gt[:, :, :, 2].max():.2f}, mean={flow_gt[:, :, :, 2].mean():.2f}")

    # Create displaced frame using backward warping with negated flow
    # This simulates forward motion: backward_warp(video, -flow) ≈ forward_warp(video, flow)
    # Using torch for speed
    displaced = warp_volume_splat3d(video, flow_gt)

    # Crop boundaries to remove invalid regions (10 pixels from each edge)
    boundary = 10
    if displaced.ndim == 4:  # Has channels
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
        flow_gt = flow_gt[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        flow_gt = flow_gt[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]

    print(f"  Warping complete, cropped {boundary}px boundaries")
    print(f"  Final displaced shape: {displaced.shape}")

    return displaced, flow_gt


def evaluate_flow_accuracy(flow_est, flow_gt, boundary=25):
    """
    Evaluate flow estimation accuracy using End-Point Error (EPE).

    Args:
        flow_est: Estimated flow field (Z, Y, X, 3)
        flow_gt: Ground truth flow field (Z, Y, X, 3)
        boundary: Pixels to exclude from boundaries

    Returns:
        EPE value
    """
    # Crop boundaries
    if boundary > 0:
        flow_est_cropped = flow_est[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
        flow_gt_cropped = flow_gt[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        flow_est_cropped = flow_est
        flow_gt_cropped = flow_gt

    # Compute End-Point Error
    epe = np.mean(np.linalg.norm(flow_est_cropped - flow_gt_cropped, axis=-1))

    return epe


def visualize_in_napari(original, displaced, corrected, flow_est=None, flow_gt=None):
    """
    Visualize original, displaced, and corrected volumes in napari.

    Args:
        original: Original processed video
        displaced: Displaced video
        corrected: Motion-corrected video
        flow_est: Estimated flow field (Z, Y, X, 3)
        flow_gt: Ground truth flow field (Z, Y, X, 3)
    """
    print("\nLaunching napari viewer...")

    viewer = napari.Viewer(title="3D Motion Correction Test")

    # Add original volume
    viewer.add_image(original, name="Original", colormap='green', blending='additive', contrast_limits=[0, 1],
        visible=True, opacity=0.7)

    # Add displaced volume
    viewer.add_image(displaced, name="Displaced (Synthetic Motion)", colormap='magenta', blending='additive',
        contrast_limits=[0, 1], visible=True, opacity=0.7)

    # Add corrected volume
    viewer.add_image(corrected, name="Corrected (Motion Compensated)", colormap='cyan', blending='additive',
        contrast_limits=[0, 1], visible=True, opacity=1.0)

    # Add flow magnitude for estimated flow - add channel dimension for napari
    if flow_est is not None:
        flow_est_magnitude = np.sqrt(flow_est[:, :, :, 0] ** 2 + flow_est[:, :, :, 1] ** 2 + flow_est[:, :, :, 2] ** 2)
        # Add empty channel dimension to match image dimensions
        flow_est_magnitude = flow_est_magnitude[..., np.newaxis]
        viewer.add_image(flow_est_magnitude, name="Estimated Flow Magnitude", colormap='viridis', visible=False,
            contrast_limits=[0, flow_est_magnitude.max()])

    # Add flow magnitude for ground truth - use same colormap as estimated
    if flow_gt is not None:
        flow_gt_magnitude = np.sqrt(flow_gt[:, :, :, 0] ** 2 + flow_gt[:, :, :, 1] ** 2 + flow_gt[:, :, :, 2] ** 2)
        # Add empty channel dimension to match image dimensions
        flow_gt_magnitude = flow_gt_magnitude[..., np.newaxis]
        viewer.add_image(flow_gt_magnitude, name="Ground Truth Flow Magnitude", colormap='viridis', visible=False,
            contrast_limits=[0, flow_gt_magnitude.max()])

        # Add flow error magnitude
        flow_error = flow_est - flow_gt if flow_est is not None else None
        if flow_error is not None:
            flow_error_magnitude = np.sqrt(
                flow_error[:, :, :, 0] ** 2 + flow_error[:, :, :, 1] ** 2 + flow_error[:, :, :, 2] ** 2)
            # Add channel dimension to prevent transposition issue in napari
            flow_error_magnitude = flow_error_magnitude[..., np.newaxis]
            viewer.add_image(flow_error_magnitude, name="Flow Error Magnitude", colormap='hot', visible=False,
                contrast_limits=[0, flow_error_magnitude.max()])

    print("\nViewer controls:")
    print("  - Use slider at bottom to navigate through Z slices")
    print("  - Toggle layers on/off with eye icons")
    print("  - Adjust opacity with sliders")
    print("  - Press '3' for 3D view")
    print("\nColor coding:")
    print("  - Green: Original")
    print("  - Magenta: Displaced (with synthetic motion)")
    print("  - Cyan: Corrected (motion compensated)")
    print("  - Overlap regions show color mixing")
    print("\nAdditional layers:")
    print("  - Flow magnitudes and error available as optional layers")

    napari.run()


def main():
    """
    Main workflow for 3D motion correction testing.
    """
    print("=" * 60)
    print("3D Motion Correction Test with Synthetic Displacement")
    print("=" * 60)

    # Fix random seed for reproducibility
    fix_seed(seed=1, deterministic=True, verbose=True)
    print()

    # Get the script's directory and navigate to data folder
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent  # Go up from examples/ to repo root
    aligned_file = repo_root / "data" / "aligned_sequence" / "compensated.HDF5"

    if not aligned_file.exists():
        print(f"\n✗ Error: Aligned file not found: {aligned_file}")
        print("  Please run the motion compensation script first.")
        return None

    # Load the pre-aligned video using pyflowreg reader
    print(f"\nLoading pre-aligned video: {aligned_file}")

    # Use binning=9 to get a proper 3D stack as mentioned in the instructions
    reader = get_video_file_reader(str(aligned_file), buffer_size=100, bin_size=9)

    print(f"  Reader shape: {reader.shape}")
    print(f"  Frames: {reader.frame_count}")
    print(f"  Channels: {reader.n_channels}")

    # Read all frames to create 3D stack
    video_3d = []
    while reader.has_batch():
        batch = reader.read_batch()
        video_3d.append(batch)

    if video_3d:
        video_3d = np.concatenate(video_3d, axis=0)
    else:
        # Fallback if batch reading doesn't work
        video_3d = reader[:]

    reader.close()

    print(f"\nLoaded 3D stack shape: {video_3d.shape}")

    # Process the 3D stack
    processed = process_3d_stack(video_3d)

    # Create displaced version with synthetic 3D motion
    displaced, flow_gt = create_displaced_frame_with_generator(processed, generator_type='high_disp'
        # Use high displacement with enhanced expansion
    )

    # Crop original to match displaced size after boundary removal
    boundary = 10
    if processed.ndim == 4:  # Has channels
        original_cropped = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        original_cropped = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary]

    print("\nPreparing frames for motion correction...")

    # Set up flow parameters for 3D optical flow
    # get_displacement expects: alpha=(2,2,2), update_lag=10, iterations=20, min_level=0,
    # levels=50, eta=0.8, a_smooth=0.5, a_data=0.45, const_assumption='gc', uvw=None, weight=None
    flow_params = {'alpha': (0.25, 0.25, 0.25),  # 3D alpha values for x, y, z axes
        'iterations': 100, 'a_data': 0.45, 'a_smooth': 1.0, 'weight': np.array([0.5, 0.5], dtype=np.float64),
        'levels': 50, 'eta': 0.8, 'update_lag': 5, 'min_level': 5,
        'const_assumption': 'gc',  # gradient constancy
        'uvw': None  # Initial flow field
    }

    # Compute 3D optical flow (preprocessing is done internally)
    flow_est = compute_3d_optical_flow(original_cropped, displaced, flow_params)

    # Apply motion correction using imregister_wrapper (backwards warping)
    print("\nApplying motion correction...")
    # imregister_wrapper warps displaced to align with original_cropped
    # flow_est contains (dx, dy, dz) components that map from displaced to original
    corrected = imregister_wrapper(displaced,  # frame to warp
        flow_est[:, :, :, 0],  # u (dx displacement)
        flow_est[:, :, :, 1],  # v (dy displacement)
        flow_est[:, :, :, 2],  # w (dz displacement)
        original_cropped,  # reference for boundary conditions
        interpolation_method='cubic')

    # Evaluate accuracy if we have ground truth
    epe = evaluate_flow_accuracy(flow_est, flow_gt, boundary=25)
    print(f"\nEnd-Point Error (EPE): {epe:.2f} pixels")

    # Print correction quality metrics
    print("\nCorrection quality metrics:")

    # Compute difference between original and corrected
    diff_original_corrected = np.mean(np.abs(original_cropped - corrected))
    diff_original_displaced = np.mean(np.abs(original_cropped - displaced))

    print(f"  Mean absolute difference (original vs displaced): {diff_original_displaced:.4f}")
    print(f"  Mean absolute difference (original vs corrected): {diff_original_corrected:.4f}")
    print(f"  Improvement ratio: {diff_original_displaced / diff_original_corrected:.2f}x")

    # Visualize in napari
    visualize_in_napari(original_cropped, displaced, corrected, flow_est, flow_gt)

    print("\n" + "=" * 60)
    print("Motion correction test complete!")
    print("=" * 60)

    return original_cropped, displaced, corrected, flow_est, flow_gt


if __name__ == "__main__":
    results = main()
    if results is not None:
        original, displaced, corrected, flow_est, flow_gt = results
