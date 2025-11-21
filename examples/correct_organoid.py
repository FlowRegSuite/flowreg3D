"""
3D motion correction for organoid timelapse using compensate_recording_3D.

This script demonstrates the file-based workflow for 3D motion correction,
analogous to the 2D jupiter_demo.py that uses compensate_recording.
"""

import os
from pathlib import Path
import numpy as np
import napari

from flowreg3d.motion_correction.OF_options_3D import OFOptions
from flowreg3d.motion_correction.compensate_recording_3D import compensate_recording
from flowreg3d.util.io.factory import get_video_file_reader


def main():
    """
    Main workflow for organoid motion correction using file-based processing.
    """
    print("=" * 60)
    print("3D Organoid Motion Correction (File-based)")
    print("=" * 60)
    
    # Setup paths
    input_file = Path("../data/xtitched_organoid_timelapse_ch0_cytosol_ch1_nuclei-1.tif")
    output_folder = Path("../data/organoid_corrected_2")
    output_folder.mkdir(exist_ok=True, parents=True)
    
    if not input_file.exists():
        print(f"\n✗ Error: Input file not found: {input_file}")
        print("  Please ensure the organoid data is in the data/ directory.")
        return None
    
    # Create OF_options matching the 2D jupiter demo pattern
    options = OFOptions(
        input_file=str(input_file),
        input_dim_order="TZYX",  # Specify dimension order for 4D TIFF
        output_path=str(output_folder / "ref_1"),
        output_format="TIFF",
        alpha=(0.02, 0.02, 0.02),
        quality_setting="quality",
        reference_frames=[0, 1, 2],
        sigma=[[0.000001, 0.4, 0.4, 0.4]],
        iterations=100,
        eta=0.8,
        min_level=0,
        update_lag=5,
        a_smooth=1.0,
        a_data=0.45,
        verbose=True,
        buffer_size=5,
    )
    
    # Run motion compensation
    print("\nRunning 3D motion compensation...")
    compensate_recording(options)
    print("Motion compensation complete!")
    
    # Read the original video for comparison
    print(f"\nReading original video from {input_file}")
    orig_reader = get_video_file_reader(str(input_file), dim_order="TZYX")
    original_frames = orig_reader[:]  # Get all frames
    orig_reader.close()
    
    # Read the compensated video
    compensated_file = output_folder / "hdf5_comp_3d" / "compensated.HDF5"
    print(f"Reading compensated video from {compensated_file}")
    
    vid_reader = get_video_file_reader(str(compensated_file))
    compensated_frames = vid_reader[:]  # Get all frames
    vid_reader.close()
    
    # Read displacement fields if saved
    displacement_file = output_folder / "hdf5_comp_3d" / "displacements.HDF5"
    flow = None
    if displacement_file.exists():
        print(f"Reading displacement fields from {displacement_file}")
        flow_reader = get_video_file_reader(str(displacement_file))
        flow = flow_reader[:]  # Get all displacement fields
        flow_reader.close()
        print(f"Flow fields shape: {flow.shape}")
    
    total_frames = len(compensated_frames)
    print(f"\nLoaded {total_frames} timepoints")
    print(f"Original shape: {original_frames.shape}")
    print(f"Compensated shape: {compensated_frames.shape}")
    
    # Compute motion statistics if flow is available
    if flow is not None:
        flow_magnitude = np.sqrt(np.sum(flow**2, axis=-1))
        print(f"\nMotion statistics:")
        print(f"  Max displacement: {flow_magnitude.max():.2f} voxels")
        print(f"  Mean displacement: {flow_magnitude.mean():.2f} voxels")
        print(f"  Std displacement: {flow_magnitude.std():.2f} voxels")
    
    # Visualize in napari
    print("\nLaunching napari viewer...")
    viewer = napari.Viewer(title="Organoid Motion Correction (File-based)")
    
    # Determine number of channels
    T, Z, Y, X, C = compensated_frames.shape
    
    # Add original data
    if C > 1:
        viewer.add_image(
            original_frames[..., 0],
            name="Original - Channel 0 (Cytosol)",
            colormap='green',
            blending='additive',
            contrast_limits=[original_frames[..., 0].min(), original_frames[..., 0].max()],
            visible=True
        )
        
        viewer.add_image(
            original_frames[..., 1], 
            name="Original - Channel 1 (Nuclei)",
            colormap='magenta',
            blending='additive',
            contrast_limits=[original_frames[..., 1].min(), original_frames[..., 1].max()],
            visible=False
        )
        
        # Add corrected data
        viewer.add_image(
            compensated_frames[..., 0],
            name="Corrected - Channel 0 (Cytosol)",
            colormap='cyan',
            blending='additive',
            contrast_limits=[compensated_frames[..., 0].min(), compensated_frames[..., 0].max()],
            visible=True
        )
        
        viewer.add_image(
            compensated_frames[..., 1],
            name="Corrected - Channel 1 (Nuclei)",
            colormap='yellow',
            blending='additive',
            contrast_limits=[compensated_frames[..., 1].min(), compensated_frames[..., 1].max()],
            visible=False
        )
    else:
        # Single channel
        viewer.add_image(
            original_frames,
            name="Original",
            colormap='green',
            blending='additive',
            visible=True
        )
        
        viewer.add_image(
            compensated_frames,
            name="Corrected",
            colormap='cyan',
            blending='additive',
            visible=True
        )
    
    # Add flow magnitude visualization if available
    if flow is not None:
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
    if flow is not None:
        print("  - Hot: Motion magnitude")
    
    napari.run()
    
    print("\n" + "=" * 60)
    print("Organoid correction complete!")
    print("=" * 60)
    
    return original_frames, compensated_frames, flow


if __name__ == "__main__":
    results = main()