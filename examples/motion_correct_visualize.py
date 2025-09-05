"""
3D motion correction visualization with sparse vector field arrows.

This script:
1. Loads a pre-aligned HDF5 video at full resolution (no downsampling)
2. Creates synthetic 3D motion displacements
3. Performs 3D motion correction at full resolution
4. Visualizes with sparse white arrow vectors showing displacement field
"""

import time
from pathlib import Path

import napari
import numpy as np
from pyflowreg.util.io.factory import get_video_file_reader
from scipy.ndimage import gaussian_filter

from flowreg3d.core.optical_flow_3d import get_displacement, imregister_wrapper
from flowreg3d.motion_generation.motion_generators import get_high_disp_3d_generator
from flowreg3d.util.random import fix_seed


def normalize_volume(volume):
    """
    Normalize volume to [0, 1] range.
    
    Args:
        volume: Input volume array
    
    Returns:
        Normalized volume
    """
    vmin = volume.min()
    vmax = volume.max()
    
    if vmax > vmin:
        return (volume - vmin) / (vmax - vmin)
    return volume


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


def compute_3d_optical_flow(frame1, frame2, flow_params):
    """
    Compute 3D optical flow between two frames at full resolution.
    
    Args:
        frame1: Reference frame (Z, Y, X) or (Z, Y, X, C)
        frame2: Target frame (Z, Y, X) or (Z, Y, X, C)
        flow_params: Dictionary of flow parameters
    
    Returns:
        Flow field (Z, Y, X, 3) where last dim is (dx, dy, dz)
    """
    print("\nComputing 3D optical flow at full resolution...")
    print(f"  Input shapes: {frame1.shape}, {frame2.shape}")
    
    # Ensure float32
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    if f1.ndim == 3:  # No channels, add channel dimension
        f1 = f1[..., np.newaxis]
        f2 = f2[..., np.newaxis]
    
    # Light Gaussian filtering for denoising
    print("  Applying light Gaussian filter (sigma=0.5)...")
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
    magnitude = np.sqrt(np.sum(flow ** 2, axis=-1))
    print(f"  Flow statistics:")
    print(f"    Max displacement: {magnitude.max():.2f} voxels")
    print(f"    Mean displacement: {magnitude.mean():.2f} voxels")
    print(f"    Std displacement: {magnitude.std():.2f} voxels")
    
    return flow


def create_sparse_vector_field(flow, spacing=50):
    """
    Create sparse vector field for visualization.
    
    Args:
        flow: Flow field (Z, Y, X, 3) where last dim is (dx, dy, dz)
        spacing: Distance between sampled vectors
    
    Returns:
        Tuple of (start_points, end_points, magnitudes)
    """
    Z, Y, X = flow.shape[:3]
    
    # Create sampling grid with specified spacing
    z_sample = np.arange(spacing//2, Z, spacing)
    y_sample = np.arange(spacing//2, Y, spacing)
    x_sample = np.arange(spacing//2, X, spacing)
    
    # Meshgrid for sample points
    zz, yy, xx = np.meshgrid(z_sample, y_sample, x_sample, indexing='ij')
    
    # Sample flow at grid points
    flow_z = flow[zz, yy, xx, 0]
    flow_y = flow[zz, yy, xx, 1]
    flow_x = flow[zz, yy, xx, 2]
    
    # Create start points (current positions)
    start_points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    
    # Create end points (displaced positions)
    # No scaling - show actual displacement
    scale = 1.0  # Show true displacement magnitude
    end_z = zz.ravel() + scale * flow_z.ravel()
    end_y = yy.ravel() + scale * flow_y.ravel()
    end_x = xx.ravel() + scale * flow_x.ravel()
    end_points = np.stack([end_z, end_y, end_x], axis=1)
    
    # Compute magnitudes for potential color coding
    magnitudes = np.sqrt(flow_z.ravel()**2 + flow_y.ravel()**2 + flow_x.ravel()**2)
    
    # Filter out very small vectors for cleaner visualization
    min_magnitude = 0.1
    mask = magnitudes > min_magnitude
    start_points = start_points[mask]
    end_points = end_points[mask]
    magnitudes = magnitudes[mask]
    
    print(f"\n  Created {len(start_points)} vector arrows (spacing={spacing}, scale={scale})")
    print(f"  Vector magnitude range: [{magnitudes.min():.2f}, {magnitudes.max():.2f}]")
    
    return start_points, end_points, magnitudes


def visualize_with_vectors(original, displaced, corrected, flow_est):
    """
    Visualize volumes with sparse vector field arrows.
    
    Args:
        original: Original volume
        displaced: Displaced volume
        corrected: Motion-corrected volume
        flow_est: Estimated flow field (Z, Y, X, 3)
    """
    print("\nLaunching napari viewer with vector field visualization...")
    
    viewer = napari.Viewer(title="3D Motion Correction with Vector Field")
    
    # Remove channel dimension from volumes if they have it for consistent display
    if original.ndim == 4 and original.shape[-1] == 1:
        original = original[..., 0]
    if displaced.ndim == 4 and displaced.shape[-1] == 1:
        displaced = displaced[..., 0]
    if corrected.ndim == 4 and corrected.shape[-1] == 1:
        corrected = corrected[..., 0]
    
    # Add original volume
    viewer.add_image(
        original, 
        name="Original", 
        colormap='green', 
        blending='additive', 
        contrast_limits=[0, 1],
        visible=True, 
        opacity=0.7
    )
    
    # Add displaced volume
    viewer.add_image(
        displaced, 
        name="Displaced (Synthetic Motion)", 
        colormap='magenta', 
        blending='additive',
        contrast_limits=[0, 1], 
        visible=True, 
        opacity=0.7
    )
    
    # Add corrected volume
    viewer.add_image(
        corrected, 
        name="Corrected (Motion Compensated)", 
        colormap='cyan', 
        blending='additive',
        contrast_limits=[0, 1], 
        visible=True, 
        opacity=1.0
    )
    
    # Add flow magnitude layer first
    flow_magnitude = np.sqrt(np.sum(flow_est**2, axis=-1))
    # Add channel dimension to prevent transposition issue in napari
    flow_magnitude = flow_magnitude[..., np.newaxis]
    viewer.add_image(
        flow_magnitude, 
        name="Flow Magnitude", 
        colormap='viridis',
        visible=True,
        contrast_limits=[0, flow_magnitude.max()]
    )
    
    # Create and add sparse vector field with real 3D arrows (add last so they render on top)
    start_points, end_points, magnitudes = create_sparse_vector_field(flow_est, spacing=50)
    
    # Create vectors for napari's vectors layer (real arrows)
    # Napari transpose swaps last two dims: ZYX -> ZXY -> YXZ
    vectors = []
    for i in range(len(start_points)):
        # After two transposes: [Z,Y,X] -> [Z,X,Y] -> [Y,X,Z]
        transformed_start = np.array([start_points[i][1], start_points[i][2], start_points[i][0]])  # Y, X, Z
        direction = end_points[i] - start_points[i]
        transformed_direction = np.array([direction[1], direction[2], direction[0]])  # dy, dx, dz
        vectors.append([transformed_start, transformed_direction])
    vectors = np.array(vectors)
    
    # Add vectors layer with real 3D arrows in white (last, so they appear on top)
    viewer.add_vectors(
        vectors,
        edge_color='white',
        edge_width=1.5,
        length=1.0,
        name='Displacement Vectors (3D Arrows)',
        opacity=0.9,
        visible=True
    )
    
    print("\nViewer controls:")
    print("  - Use slider at bottom to navigate through Z slices")
    print("  - Toggle layers on/off with eye icons")
    print("  - Adjust opacity with sliders")
    print("  - Press '3' for 3D view to see vectors in 3D space")
    print("\nVisualization layers:")
    print("  - Green: Original volume")
    print("  - Magenta: Displaced volume (with synthetic motion)")
    print("  - Cyan: Corrected volume (motion compensated)")
    print("  - White arrows: Sparse displacement vector field (sampled every 25 voxels)")
    print("  - Flow Magnitude: Optional heatmap of displacement magnitude")
    
    napari.run()


def main():
    """
    Main workflow for full-resolution 3D motion correction visualization.
    """
    print("=" * 60)
    print("3D Motion Correction Visualization (Full Resolution)")
    print("=" * 60)
    
    # Fix random seed for reproducibility
    fix_seed(seed=1, deterministic=True, verbose=True)
    print()
    
    # Get the script's directory and navigate to data folder
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    aligned_file = repo_root / "data" / "aligned_sequence" / "compensated.HDF5"
    
    if not aligned_file.exists():
        print(f"\n✗ Error: Aligned file not found: {aligned_file}")
        print("  Please run the motion compensation script first.")
        return None
    
    # Load the pre-aligned video at full resolution
    print(f"\nLoading pre-aligned video at full resolution: {aligned_file}")

    reader = get_video_file_reader(
        str(aligned_file),
        buffer_size=100,
        bin_size=9
    )
    
    print(f"  Reader shape: {reader.shape}")
    print(f"  Frames: {reader.frame_count}")
    print(f"  Channels: {reader.n_channels}")
    
    # Read all frames to create 3D stack
    print("  Loading frames...")
    video_3d = []
    while reader.has_batch():
        batch = reader.read_batch()
        video_3d.append(batch)
    
    if video_3d:
        video_3d = np.concatenate(video_3d, axis=0)
    else:
        video_3d = reader[:]
    
    reader.close()
    
    print(f"\nLoaded 3D stack shape: {video_3d.shape}")
    
    # Normalize the volume
    video_normalized = normalize_volume(video_3d)
    
    # Create displaced version with synthetic 3D motion
    print("\nCreating displaced frame with synthetic motion...")
    
    depth, height, width = video_normalized.shape[:3]
    
    # Use high displacement generator
    generator = get_high_disp_3d_generator()
    flow_gt, invalid_mask = generator(depth=depth, height=height, width=width)
    
    # Apply displacement
    displaced = warp_volume_splat3d(video_normalized, flow_gt)
    
    # Crop boundaries to remove edge artifacts
    boundary = 10
    if video_normalized.ndim == 4:
        original_cropped = video_normalized[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
        flow_gt = flow_gt[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        original_cropped = video_normalized[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        flow_gt = flow_gt[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    
    print(f"  Cropped to remove {boundary}px boundaries")
    print(f"  Working shape: {original_cropped.shape}")
    
    # Set up flow parameters for full resolution
    flow_params = {
        'alpha': (0.25, 0.25, 0.25),  # 3D alpha values
        'iterations': 100,
        'a_data': 0.45,
        'a_smooth': 1.0,
        'weight': np.array([0.5, 0.5], dtype=np.float64),
        'levels': 50,
        'eta': 0.8,
        'update_lag': 5,
        'min_level': 5,
        'const_assumption': 'gc',
        'uvw': None
    }
    
    # Compute 3D optical flow at full resolution
    flow_est = compute_3d_optical_flow(original_cropped, displaced, flow_params)
    
    # Apply motion correction
    print("\nApplying motion correction...")
    corrected = imregister_wrapper(
        displaced,
        flow_est[:, :, :, 0],  # u (dx displacement)
        flow_est[:, :, :, 1],  # v (dy displacement)
        flow_est[:, :, :, 2],  # w (dz displacement)
        original_cropped,
        interpolation_method='cubic'
    )
    
    # Compute error metrics
    print("\nComputing error metrics...")
    epe = np.mean(np.linalg.norm(flow_est - flow_gt, axis=-1))
    print(f"  End-Point Error (EPE): {epe:.2f} pixels")
    
    diff_original_displaced = np.mean(np.abs(original_cropped - displaced))
    diff_original_corrected = np.mean(np.abs(original_cropped - corrected))
    
    print(f"  Mean absolute difference (original vs displaced): {diff_original_displaced:.4f}")
    print(f"  Mean absolute difference (original vs corrected): {diff_original_corrected:.4f}")
    print(f"  Improvement ratio: {diff_original_displaced / diff_original_corrected:.2f}x")
    
    # Visualize with vector field
    visualize_with_vectors(original_cropped, displaced, corrected, flow_est)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    return original_cropped, displaced, corrected, flow_est


if __name__ == "__main__":
    results = main()