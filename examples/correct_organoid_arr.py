"""
3D motion correction for organoid timelapse data.

This script:
1. Loads an organoid timelapse TIFF file
2. Creates a reference volume from early timepoints
3. Performs 3D motion correction using flowreg3d
4. Saves the corrected data
5. Visualizes original and corrected in napari
"""

import time
from pathlib import Path
import numpy as np
import napari
from scipy.ndimage import gaussian_filter

from flowreg3d.util.io.tiff_3d import TIFFFileReader3D, TIFFFileWriter3D
from flowreg3d.core.optical_flow_3d import get_displacement, imregister_wrapper
from flowreg3d.motion_correction.compensate_arr_3D import compensate_arr_3D
from flowreg3d.motion_correction.OF_options_3D import OFOptions


def main():
    """
    Main workflow for organoid motion correction.
    """
    print("=" * 60)
    print("3D Organoid Motion Correction")
    print("=" * 60)
    
    # Setup paths
    input_file = Path("../data/xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei-1.tif")
    output_dir = Path("../data/corrected")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "organoid_corrected.tif"
    
    if not input_file.exists():
        print(f"\n✗ Error: Input file not found: {input_file}")
        print("  Please ensure the organoid data is in the data/ directory.")
        return None
    
    # Load the organoid data using 3D TIFF reader
    print(f"\nLoading organoid data: {input_file}")
    
    # Try common dimension orderings for microscopy data
    # This file has TZYX format (no explicit channel dimension)
    reader = TIFFFileReader3D(str(input_file), buffer_size=5, dim_order='TZYX')
    
    print(f"  Data shape: {reader.shape}")
    print(f"  Timepoints: {reader.frame_count}")
    print(f"  Z slices: {reader.depth}")
    print(f"  Image size: {reader.height} x {reader.width}")
    print(f"  Channels: {reader.n_channels}")
    print(f"  Data type: {reader.dtype}")
    
    # Read all data into memory (for small datasets)
    # For large datasets, process in batches
    print("\nReading all timepoints...")
    video_data = reader[:]  # Shape: (T, Z, Y, X, C)
    reader.close()
    
    T, Z, Y, X, C = video_data.shape
    print(f"Loaded data shape: {video_data.shape}")
    
    # Create reference volume from first few timepoints
    reference = video_data[10]
    
    # Setup OF options for 3D motion correction
    options = OFOptions(
        alpha=(0.001, 0.001, 0.001),  # Low regularization for 3D
        quality_setting="quality",  # Balance speed and quality
        sigma=[[0.000001, 0.01, 0.01, 0.01]],
        levels=50,
        iterations=100,
        eta=0.8,
        min_level=0,  # Don't go too coarse
        update_lag=5,
        a_smooth=1.0,
        a_data=0.45,
        save_w=True,  # Save displacement fields
        output_typename="float32",  # Save memory
        verbose=True,
        buffer_size=21,
        cc_initialization=True,
        reference_frames=[10]
    )
    
    # Option 1: Use compensate_arr_3D for all-at-once processing
    print("\nPerforming 3D motion correction...")

    def progress_callback(current, total):
        print(f"  Processing volume {current}/{total} ({100*current/total:.1f}%)")

    # Run motion correction
    t_start = time.perf_counter()
    registered, flow = compensate_arr_3D(
        video_data,
        reference,
        options,
        progress_callback=progress_callback
    )
    t_elapsed = time.perf_counter() - t_start

    print(f"\nMotion correction complete in {t_elapsed:.1f} seconds!")
    print(f"Registered shape: {registered.shape}")
    print(f"Flow fields shape: {flow.shape}")

    # Compute statistics
    flow_magnitude = np.sqrt(np.sum(flow**2, axis=-1))
    print(f"\nMotion statistics:")
    print(f"  Max displacement: {flow_magnitude.max():.2f} voxels")
    print(f"  Mean displacement: {flow_magnitude.mean():.2f} voxels")
    print(f"  Std displacement: {flow_magnitude.std():.2f} voxels")

    # Save corrected data
    print(f"\nSaving corrected data to: {output_file}")

    
    # Visualize in napari
    print("\nLaunching napari viewer...")
    viewer = napari.Viewer(title="Organoid Motion Correction")
    
    # Add original data (first channel for visualization)
    if C > 1:
        viewer.add_image(
            video_data[..., 0],
            name="Original - Channel 0 (Cytosol)",
            colormap='green',
            blending='additive',
            contrast_limits=[video_data[..., 0].min(), video_data[..., 0].max()],
            visible=True
        )
        
        viewer.add_image(
            video_data[..., 1], 
            name="Original - Channel 1 (Nuclei)",
            colormap='magenta',
            blending='additive',
            contrast_limits=[video_data[..., 1].min(), video_data[..., 1].max()],
            visible=False
        )
        
        # Add corrected data
        viewer.add_image(
            registered[..., 0],
            name="Corrected - Channel 0 (Cytosol)",
            colormap='cyan',
            blending='additive',
            contrast_limits=[registered[..., 0].min(), registered[..., 0].max()],
            visible=True
        )
        
        viewer.add_image(
            registered[..., 1],
            name="Corrected - Channel 1 (Nuclei)",
            colormap='yellow',
            blending='additive',
            contrast_limits=[registered[..., 1].min(), registered[..., 1].max()],
            visible=False
        )
    else:
        # Single channel
        viewer.add_image(
            video_data,
            name="Original",
            colormap='green',
            blending='additive',
            visible=True
        )
        
        viewer.add_image(
            registered,
            name="Corrected",
            colormap='cyan',
            blending='additive',
            visible=True
        )
    
    # Add flow magnitude visualization
    flow_magnitude = np.sqrt(np.sum(flow**2, axis=-1))
    viewer.add_image(
        flow_magnitude,
        name="Motion Magnitude",
        colormap='hot',
        visible=False,
        contrast_limits=[0, flow_magnitude.max()]
    )
    
    print("\nViewer controls:")
    print("  - Use slider at bottom to navigate through time")
    print("  - Use slider on right to navigate through Z slices")
    print("  - Toggle layers on/off with eye icons")
    print("  - Press '3' for 3D view")
    print("\nColor coding:")
    print("  - Green/Magenta: Original channels")
    print("  - Cyan/Yellow: Corrected channels")
    print("  - Hot: Motion magnitude")
    
    napari.run()
    
    print("\n" + "=" * 60)
    print("Organoid correction complete!")
    print("=" * 60)
    
    return video_data, registered, flow


if __name__ == "__main__":
    results = main()