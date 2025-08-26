"""
Simplified 3D stack processing and displacement visualization script.

This script:
1. Loads a pre-aligned HDF5 video using pyflowreg readers
2. Processes it (resize 50%, crop 25px boundaries, normalize)
3. Creates a second frame with synthetic 3D motion displacements
4. Visualizes both in napari for comparison
"""

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


def create_displaced_frame_with_generator(video, generator_type='high_disp'):
    """
    Create a second 3D frame with synthetic motion displacements.
    
    Args:
        video: Input video array (Z, Y, X) or (Z, Y, X, C)
        generator_type: Type of motion generator ('default', 'low_disp', 'test', 'high_disp')
    
    Returns:
        Tuple of (displaced_video, flow_field)
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
    flow, invalid_mask = generator(depth=depth, height=height, width=width)
    
    # Print flow statistics
    print(f"  Flow field shape: {flow.shape}")
    print(f"  Flow magnitude stats:")
    print(f"    Z: min={flow[:,:,:,0].min():.2f}, max={flow[:,:,:,0].max():.2f}, mean={flow[:,:,:,0].mean():.2f}")
    print(f"    Y: min={flow[:,:,:,1].min():.2f}, max={flow[:,:,:,1].max():.2f}, mean={flow[:,:,:,1].mean():.2f}")
    print(f"    X: min={flow[:,:,:,2].min():.2f}, max={flow[:,:,:,2].max():.2f}, mean={flow[:,:,:,2].mean():.2f}")
    
    # Warp the volume
    displaced = warp_volume_with_flow(video, flow)
    
    # Crop boundaries to remove invalid regions (10 pixels from each edge)
    boundary = 10
    if displaced.ndim == 4:  # Has channels
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
        flow = flow[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        displaced = displaced[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        flow = flow[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    
    print(f"  Warping complete, cropped {boundary}px boundaries")
    print(f"  Final displaced shape: {displaced.shape}")
    
    return displaced, flow


def visualize_in_napari(original, displaced, flow=None):
    """
    Visualize original and displaced volumes in napari.
    
    Args:
        original: Original processed video
        displaced: Displaced video
        flow: Optional flow field (Z, Y, X, 3)
        invalid_mask: Optional invalid pixel mask
    """
    print("\nLaunching napari viewer...")
    
    viewer = napari.Viewer(title="3D Stack with Synthetic Motion")
    
    # Add original volume
    viewer.add_image(
        original,
        name="Original",
        colormap='green',
        blending='additive',
        contrast_limits=[0, 1],
        visible=True
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
    
    # Add flow magnitude if provided
    if flow is not None:
        flow_magnitude = np.sqrt(
            flow[:, :, :, 0]**2 + 
            flow[:, :, :, 1]**2 + 
            flow[:, :, :, 2]**2
        )
        viewer.add_image(
            flow_magnitude,
            name="Flow Magnitude",
            colormap='viridis',
            visible=False,
            contrast_limits=[0, flow_magnitude.max()]
        )
    
    
    print("\nViewer controls:")
    print("  - Use slider at bottom to navigate through Z slices")
    print("  - Toggle layers on/off with eye icons")
    print("  - Adjust opacity with sliders")
    print("  - Press '3' for 3D view")
    print("  - Green: Original | Magenta: Displaced")
    print("  - Overlap regions appear white/yellow")
    print("  - Flow magnitude available as additional layer")
    
    napari.run()


def main():
    """
    Main workflow for 3D stack processing and displacement visualization.
    """
    print("=" * 60)
    print("3D Stack Processing and Displacement Visualization")
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
    # Options: 'default', 'low_disp', 'test', 'high_disp'
    displaced, flow = create_displaced_frame_with_generator(
        processed, 
        generator_type='high_disp'  # Use high displacement with enhanced expansion
    )
    
    # Crop original to match displaced size after boundary removal
    boundary = 10
    if processed.ndim == 4:  # Has channels
        processed = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary, :]
    else:
        processed = processed[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    
    # Visualize in napari
    visualize_in_napari(processed, displaced, flow)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return processed, displaced, flow


if __name__ == "__main__":
    original, displaced, flow = main()