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
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, map_coordinates
import napari

from pyflowreg.util.io.factory import get_video_file_reader
from flowreg3d.motion_generation.motion_generators import (
    get_default_3d_generator,
    get_low_disp_3d_generator,
    get_test_3d_generator,
    get_high_disp_3d_generator
)
from flowreg3d.core.optical_flow_3d import get_displacement


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
    print("  Resizing to 50%...")
    if video.ndim == 4:  # Has channels
        zoom_factors = (1.0, 0.5, 0.5, 1.0)  # Don't resize time or channels
    else:
        zoom_factors = (1.0, 0.5, 0.5)  # Don't resize time
    
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


def preprocess_for_flow(frame1, frame2):
    """
    Preprocess frames for optical flow computation.
    Normalizes each channel independently based on frame1 statistics.
    
    Args:
        frame1: Reference frame (Z, Y, X) or (Z, Y, X, C)
        frame2: Target frame (Z, Y, X) or (Z, Y, X, C)
    
    Returns:
        Preprocessed frame1, frame2
    """
    # Ensure float32
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    if f1.ndim == 3:  # No channels, add channel dimension
        f1 = f1[..., np.newaxis]
        f2 = f2[..., np.newaxis]
    
    # Normalize per channel based on frame1 statistics
    mins = f1.min(axis=(0, 1, 2), keepdims=True)
    maxs = f1.max(axis=(0, 1, 2), keepdims=True)
    ranges = maxs - mins
    
    # Avoid division by zero
    ranges = np.where(ranges > 0, ranges, 1.0)
    
    f1_norm = (f1 - mins) / ranges
    f2_norm = (f2 - mins) / ranges
    
    return f1_norm, f2_norm


def warp_volume_with_flow(volume, flow):
    """
    Warp a 3D volume using a displacement field.
    
    Args:
        volume: Input volume (Z, Y, X) or (Z, Y, X, C)
        flow: Displacement field (Z, Y, X, 3) where last dim is (dz, dy, dx)
    
    Returns:
        Warped volume with same shape as input
    """
    depth, height, width = volume.shape[:3]
    has_channels = volume.ndim == 4
    
    # Create coordinate grids
    zi, yi, xi = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    
    # Apply displacement (backward mapping for interpolation)
    coords_z = zi - flow[:, :, :, 0]
    coords_y = yi - flow[:, :, :, 1]  
    coords_x = xi - flow[:, :, :, 2]
    
    # Stack coordinates for map_coordinates
    coords = np.array([coords_z, coords_y, coords_x])
    
    if has_channels:
        warped = np.zeros_like(volume)
        for c in range(volume.shape[3]):
            warped[:, :, :, c] = map_coordinates(
                volume[:, :, :, c],
                coords,
                order=1,  # Linear interpolation
                mode='constant',
                cval=0
            )
    else:
        warped = map_coordinates(
            volume,
            coords,
            order=1,  # Linear interpolation
            mode='constant',
            cval=0
        )
    
    return warped


def compute_3d_optical_flow(frame1, frame2, flow_params):
    """
    Compute 3D optical flow between two frames.
    
    Args:
        frame1: Reference frame (Z, Y, X, C)
        frame2: Target frame (Z, Y, X, C)
        flow_params: Dictionary of flow parameters
    
    Returns:
        Flow field (Z, Y, X, 3) where last dim is (dz, dy, dx)
    """
    print("\nComputing 3D optical flow...")
    print(f"  Input shapes: {frame1.shape}, {frame2.shape}")
    
    t0 = time.perf_counter()
    
    # Call get_displacement which returns (Z, Y, X, 3) with (dz, dy, dx)
    flow = get_displacement(frame1, frame2, **flow_params)
    
    t_elapsed = time.perf_counter() - t0
    print(f"  Flow computation time: {t_elapsed:.2f} seconds")
    
    # Print flow statistics
    print(f"  Flow field shape: {flow.shape}")
    print(f"  Flow magnitude stats:")
    print(f"    Z: min={flow[:,:,:,0].min():.2f}, max={flow[:,:,:,0].max():.2f}, mean={flow[:,:,:,0].mean():.2f}")
    print(f"    Y: min={flow[:,:,:,1].min():.2f}, max={flow[:,:,:,1].max():.2f}, mean={flow[:,:,:,1].mean():.2f}")
    print(f"    X: min={flow[:,:,:,2].min():.2f}, max={flow[:,:,:,2].max():.2f}, mean={flow[:,:,:,2].mean():.2f}")
    
    total_magnitude = np.sqrt(np.sum(flow**2, axis=-1))
    print(f"  Total magnitude: min={total_magnitude.min():.2f}, max={total_magnitude.max():.2f}, mean={total_magnitude.mean():.2f}")
    
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
    print(f"    Z: min={flow_gt[:,:,:,0].min():.2f}, max={flow_gt[:,:,:,0].max():.2f}, mean={flow_gt[:,:,:,0].mean():.2f}")
    print(f"    Y: min={flow_gt[:,:,:,1].min():.2f}, max={flow_gt[:,:,:,1].max():.2f}, mean={flow_gt[:,:,:,1].mean():.2f}")
    print(f"    X: min={flow_gt[:,:,:,2].min():.2f}, max={flow_gt[:,:,:,2].max():.2f}, mean={flow_gt[:,:,:,2].mean():.2f}")
    
    # Warp the volume
    displaced = warp_volume_with_flow(video, flow_gt)
    
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
    
    # Add flow magnitude for estimated flow
    if flow_est is not None:
        flow_est_magnitude = np.sqrt(
            flow_est[:, :, :, 0]**2 + 
            flow_est[:, :, :, 1]**2 + 
            flow_est[:, :, :, 2]**2
        )
        viewer.add_image(
            flow_est_magnitude,
            name="Estimated Flow Magnitude",
            colormap='viridis',
            visible=False,
            contrast_limits=[0, flow_est_magnitude.max()]
        )
    
    # Add flow magnitude for ground truth
    if flow_gt is not None:
        flow_gt_magnitude = np.sqrt(
            flow_gt[:, :, :, 0]**2 + 
            flow_gt[:, :, :, 1]**2 + 
            flow_gt[:, :, :, 2]**2
        )
        viewer.add_image(
            flow_gt_magnitude,
            name="Ground Truth Flow Magnitude",
            colormap='plasma',
            visible=False,
            contrast_limits=[0, flow_gt_magnitude.max()]
        )
        
        # Add flow error magnitude
        flow_error = flow_est - flow_gt if flow_est is not None else None
        if flow_error is not None:
            flow_error_magnitude = np.sqrt(
                flow_error[:, :, :, 0]**2 + 
                flow_error[:, :, :, 1]**2 + 
                flow_error[:, :, :, 2]**2
            )
            viewer.add_image(
                flow_error_magnitude,
                name="Flow Error Magnitude",
                colormap='hot',
                visible=False,
                contrast_limits=[0, flow_error_magnitude.max()]
            )
    
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
    reader = get_video_file_reader(
        str(aligned_file),
        buffer_size=100,
        bin_size=9  # Important: binning=9 returns perfect 3D stack
    )
    
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
    displaced, flow_gt = create_displaced_frame_with_generator(
        processed, 
        generator_type='high_disp'  # Use high displacement with enhanced expansion
    )
    
    # Crop original to match displaced size after boundary removal
    boundary = 10
    if processed.ndim == 4:  # Has channels
        original_cropped = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        original_cropped = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    
    # Preprocess frames for optical flow
    print("\nPreparing frames for motion correction...")
    frame1_norm, frame2_norm = preprocess_for_flow(original_cropped, displaced)
    
    # Set up flow parameters for 3D optical flow
    # get_displacement expects: alpha=(2,2,2), update_lag=10, iterations=20, min_level=0, 
    # levels=50, eta=0.8, a_smooth=0.5, a_data=0.45, const_assumption='gc', uvw=None, weight=None
    flow_params = {
        'alpha': (8, 8, 8),  # 3D alpha values for z, y, x
        'iterations': 100,
        'a_data': 0.45,
        'a_smooth': 1.0,
        'weight': np.array([0.5, 0.5], dtype=np.float64) if frame1_norm.shape[-1] == 2 else None,
        'levels': 50,
        'eta': 0.8,
        'update_lag': 5,
        'min_level': 5,  # Use fast mode for initial testing
        'const_assumption': 'gc',  # gradient constancy
        'uvw': None  # Initial flow field
    }
    
    # Compute 3D optical flow
    flow_est = compute_3d_optical_flow(frame1_norm, frame2_norm, flow_params)
    
    # Apply motion correction (warp displaced back to original)
    print("\nApplying motion correction...")
    corrected = warp_volume_with_flow(displaced, -flow_est)  # Negative flow for inverse warp
    
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
    print(f"  Improvement ratio: {diff_original_displaced/diff_original_corrected:.2f}x")
    
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