"""
Tests for the TIFF reshape CLI command.

Tests conversion of flat TIFF files to proper 3D volumetric stacks.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import argparse

from flowreg3d.util.io.tiff_3d import TIFFFileWriter3D, TIFFFileReader3D
from flowreg3d.cli.tiff_reshape import reshape_tiff, add_tiff_reshape_parser


@pytest.fixture
def flat_tiff_file():
    """Create a flat TIFF file simulating a 3D stack stored as 2D slices."""
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Create test data: 3 volumes, 10 slices per volume, 64x64 pixels, 2 channels
        n_volumes = 3
        slices_per_volume = 10
        height, width = 64, 64
        n_channels = 2

        # Total frames = volumes * slices_per_volume
        total_frames = n_volumes * slices_per_volume

        # Create flat data (as would be stored in a flat TIFF)
        flat_data = np.random.randint(0, 255,
                                     (total_frames, height, width, n_channels),
                                     dtype=np.uint8)

        # Write as flat TIFF using 3D writer but with T dimension only
        writer = TIFFFileWriter3D(tmp.name, dim_order='TYXC')

        # Write each frame individually to simulate flat structure
        for frame in flat_data:
            # Add fake Z dimension of 1
            frame_3d = frame[np.newaxis, :, :, :]  # (1, H, W, C)
            writer.write_frames(frame_3d)

        writer.close()

        yield tmp.name, n_volumes, slices_per_volume, flat_data

        # Cleanup - with retry for Windows file locks
        import time
        for attempt in range(3):
            try:
                Path(tmp.name).unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)  # Give Windows time to release the file
                else:
                    pass  # Ignore on final attempt


def test_reshape_basic(flat_tiff_file):
    """Test basic reshaping of flat TIFF to 3D stack."""
    input_path, n_volumes, slices_per_volume, original_data = flat_tiff_file

    with tempfile.NamedTemporaryFile(suffix='_3d.tif', delete=False) as output_tmp:
        output_path = output_tmp.name

    try:
        # Create args object simulating CLI arguments
        args = argparse.Namespace(
            input_file=input_path,
            output_file=output_path,
            slices_per_volume=slices_per_volume,
            frames_per_slice=1,
            start_volume=None,
            end_volume=None,
            volume_stride=1,
            channels=None,
            dim_order=None,
            compression='none',
            output_dim_order='TZYXC',
            imagej=False,
            dry_run=False,
            verbose=False,
            overwrite=True
        )

        # Run reshape
        result = reshape_tiff(args)
        assert result == 0, "Reshape should succeed"

        # Read back the reshaped file
        # Don't specify dim_order - let it auto-detect from ImageJ metadata
        reader = TIFFFileReader3D(output_path)
        reshaped_data = reader[:]  # Read all data
        reader.close()

        # Check dimensions
        assert reshaped_data.shape[0] == n_volumes, "Should have correct number of volumes"
        assert reshaped_data.shape[1] == slices_per_volume, "Should have correct slices per volume"
        assert reshaped_data.shape[2:] == original_data.shape[1:], "Should preserve H, W, C dimensions"

        # Check data integrity - reshaping should preserve the data
        flat_reshaped = reshaped_data.reshape(-1, *reshaped_data.shape[2:])
        np.testing.assert_array_equal(flat_reshaped, original_data,
                                     "Data should be preserved after reshaping")

    finally:
        Path(output_path).unlink(missing_ok=True)


def test_reshape_with_volume_selection(flat_tiff_file):
    """Test reshaping with volume range selection."""
    input_path, n_volumes, slices_per_volume, original_data = flat_tiff_file

    with tempfile.NamedTemporaryFile(suffix='_3d.tif', delete=False) as output_tmp:
        output_path = output_tmp.name

    try:
        # Select volumes 1-2 (0-indexed, so volumes at index 1 and 2)
        start_vol = 1
        end_vol = 3

        args = argparse.Namespace(
            input_file=input_path,
            output_file=output_path,
            slices_per_volume=slices_per_volume,
            frames_per_slice=1,
            start_volume=start_vol,
            end_volume=end_vol,
            volume_stride=1,
            channels=None,
            dim_order=None,
            compression='none',
            output_dim_order='TZYXC',
            imagej=False,
            dry_run=False,
            verbose=False,
            overwrite=True
        )

        # Run reshape
        result = reshape_tiff(args)
        assert result == 0, "Reshape should succeed"

        # Read back
        # Don't specify dim_order - let it auto-detect from ImageJ metadata
        reader = TIFFFileReader3D(output_path)
        reshaped_data = reader[:]
        reader.close()

        # Check we got the right number of volumes
        expected_volumes = end_vol - start_vol
        assert reshaped_data.shape[0] == expected_volumes, f"Should have {expected_volumes} volumes"

        # Check we got the right data
        expected_start_frame = start_vol * slices_per_volume
        expected_end_frame = end_vol * slices_per_volume
        expected_data = original_data[expected_start_frame:expected_end_frame]

        flat_reshaped = reshaped_data.reshape(-1, *reshaped_data.shape[2:])
        np.testing.assert_array_equal(flat_reshaped, expected_data,
                                     "Selected volumes should match original data")

    finally:
        Path(output_path).unlink(missing_ok=True)


def test_reshape_with_stride(flat_tiff_file):
    """Test reshaping with volume stride."""
    input_path, n_volumes, slices_per_volume, original_data = flat_tiff_file

    with tempfile.NamedTemporaryFile(suffix='_3d.tif', delete=False) as output_tmp:
        output_path = output_tmp.name

    try:
        # Take every 2nd volume
        stride = 2

        args = argparse.Namespace(
            input_file=input_path,
            output_file=output_path,
            slices_per_volume=slices_per_volume,
            frames_per_slice=1,
            start_volume=None,
            end_volume=None,
            volume_stride=stride,
            channels=None,
            dim_order=None,
            compression='none',
            output_dim_order='TZYXC',
            imagej=False,
            dry_run=False,
            verbose=False,
            overwrite=True
        )

        # Run reshape
        result = reshape_tiff(args)
        assert result == 0, "Reshape should succeed"

        # Read back
        # Don't specify dim_order - let it auto-detect from ImageJ metadata
        reader = TIFFFileReader3D(output_path)
        reshaped_data = reader[:]
        reader.close()

        # Check we got the right number of volumes
        expected_volumes = (n_volumes + stride - 1) // stride  # Ceiling division
        assert reshaped_data.shape[0] == expected_volumes, f"Should have {expected_volumes} volumes with stride {stride}"

        # Check we got volumes 0 and 2 (skipping volume 1)
        for vol_idx in range(expected_volumes):
            original_vol_idx = vol_idx * stride
            start_frame = original_vol_idx * slices_per_volume
            end_frame = start_frame + slices_per_volume

            expected_volume = original_data[start_frame:end_frame]
            actual_volume = reshaped_data[vol_idx]

            np.testing.assert_array_equal(actual_volume, expected_volume,
                                         f"Volume {vol_idx} should match original volume {original_vol_idx}")

    finally:
        Path(output_path).unlink(missing_ok=True)


def test_reshape_dry_run(flat_tiff_file):
    """Test dry run mode doesn't write output."""
    input_path, n_volumes, slices_per_volume, _ = flat_tiff_file

    with tempfile.NamedTemporaryFile(suffix='_3d.tif', delete=False) as output_tmp:
        output_path = output_tmp.name

    # Delete the temp file so we can check it doesn't get created
    Path(output_path).unlink()

    try:
        args = argparse.Namespace(
            input_file=input_path,
            output_file=output_path,
            slices_per_volume=slices_per_volume,
            frames_per_slice=1,
            start_volume=None,
            end_volume=None,
            volume_stride=1,
            channels=None,
            dim_order=None,
            compression='none',
            output_dim_order='TZYXC',
            imagej=False,
            dry_run=True,  # Dry run mode
            verbose=False,
            overwrite=True
        )

        # Run reshape in dry run mode
        result = reshape_tiff(args)
        assert result == 0, "Dry run should succeed"

        # Check output file was NOT created
        assert not Path(output_path).exists(), "Dry run should not create output file"

    finally:
        Path(output_path).unlink(missing_ok=True)


def test_cli_parser():
    """Test the CLI argument parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Add tiff-reshape subcommand
    add_tiff_reshape_parser(subparsers)

    # Test parsing basic command
    args = parser.parse_args(['tiff-reshape', 'input.tif', 'output.tif'])
    assert args.input_file == 'input.tif'
    assert args.output_file == 'output.tif'
    assert args.slices_per_volume is None  # Should auto-detect

    # Test with explicit slices
    args = parser.parse_args(['tiff-reshape', 'input.tif', 'output.tif', '-z', '30'])
    assert args.slices_per_volume == 30

    # Test volume selection
    args = parser.parse_args(['tiff-reshape', 'input.tif', 'output.tif',
                              '--start-volume', '5', '--end-volume', '10'])
    assert args.start_volume == 5
    assert args.end_volume == 10

    # Test stride
    args = parser.parse_args(['tiff-reshape', 'input.tif', 'output.tif', '--stride', '2'])
    assert args.volume_stride == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])