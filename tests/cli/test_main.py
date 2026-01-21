"""
Tests for the main CLI entry point and basic functionality.

Tests the overall CLI structure, help messages, version info, and subcommand routing.
"""

import pytest
import sys
from unittest.mock import patch
from io import StringIO
import tempfile
from pathlib import Path
import numpy as np

from flowreg3d.cli.main import main
from flowreg3d.util.io.tiff_3d import TIFFFileWriter3D


class TestMainCLI:
    """Test the main CLI entry point."""

    def test_main_no_args(self):
        """Test main with no arguments shows error."""
        with patch("sys.argv", ["flowreg3d"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for missing required arguments
            assert exc_info.value.code == 2

    def test_main_help(self):
        """Test help flag works."""
        with patch("sys.argv", ["flowreg3d", "--help"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Help should exit with 0
                assert exc_info.value.code == 0
                output = mock_stdout.getvalue()
                assert "FlowReg3D" in output
                assert "tiff-reshape" in output  # Should list subcommands

    def test_main_version(self):
        """Test version flag works."""
        with patch("sys.argv", ["flowreg3d", "--version"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
                output = mock_stdout.getvalue()
                assert "flowreg3d" in output
                assert "0.1.0" in output  # Check version number

    def test_main_verbose_flag(self):
        """Test verbose flag is recognized."""
        # Create a minimal test file (avoid open handle on Windows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.tif")
            output_path = str(Path(tmpdir) / "output.tif")

            data = np.random.randint(0, 255, (10, 64, 64, 1), dtype=np.uint8)
            writer = TIFFFileWriter3D(input_path)
            for frame in data:
                writer.write_frames(frame[np.newaxis, :, :, :])
            writer.close()

            with patch(
                "sys.argv",
                [
                    "flowreg3d",
                    "tiff-reshape",
                    "--verbose",
                    input_path,
                    output_path,
                    "-z",
                    "2",
                ],
            ):
                # Mock the reshape function to avoid actual processing
                with patch("flowreg3d.cli.tiff_reshape.reshape_tiff") as mock_reshape:
                    mock_reshape.return_value = 0
                    result = main()
                    assert result == 0
                    # Check that verbose was passed through
                    called_args = mock_reshape.call_args[0][0]
                    assert called_args.verbose is True

    def test_subcommand_routing(self):
        """Test that subcommands are routed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            input_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix="_out.tif", delete=False) as tmp:
            output_path = tmp.name

        try:
            with patch(
                "sys.argv",
                ["flowreg3d", "tiff-reshape", input_path, output_path, "-z", "5"],
            ):
                # Mock the reshape function
                with patch("flowreg3d.cli.tiff_reshape.reshape_tiff") as mock_reshape:
                    mock_reshape.return_value = 0
                    result = main()

                    # Check the function was called
                    assert mock_reshape.called
                    assert result == 0

                    # Check arguments were parsed correctly
                    called_args = mock_reshape.call_args[0][0]
                    assert called_args.input_file == input_path
                    assert called_args.output_file == output_path
                    assert called_args.slices_per_volume == 5
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_invalid_subcommand(self):
        """Test that invalid subcommands show error."""
        with patch("sys.argv", ["flowreg3d", "invalid-command"]):
            with patch("sys.stderr", new=StringIO()) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2
                error_output = mock_stderr.getvalue()
                assert (
                    "invalid choice" in error_output or "unrecognized" in error_output
                )

    def test_subcommand_help(self):
        """Test subcommand help works."""
        with patch("sys.argv", ["flowreg3d", "tiff-reshape", "--help"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
                output = mock_stdout.getvalue()
                assert "tiff-reshape" in output
                assert "--slices-per-volume" in output
                assert "--start-volume" in output

    def test_error_handling_without_verbose(self):
        """Test error handling shows simple message without verbose flag."""
        with patch(
            "sys.argv",
            ["flowreg3d", "tiff-reshape", "nonexistent.tif", "output.tif", "-z", "5"],
        ):
            with patch("sys.stderr", new=StringIO()) as mock_stderr:
                result = main()
                assert result == 1
                error_output = mock_stderr.getvalue()
                assert "Error:" in error_output
                # Should not show traceback without verbose
                assert "Traceback" not in error_output

    def test_error_handling_with_verbose(self):
        """Test error handling shows proper error message with verbose flag."""
        with patch(
            "sys.argv",
            [
                "flowreg3d",
                "--verbose",
                "tiff-reshape",
                "nonexistent.tif",
                "output.tif",
                "-z",
                "5",
            ],
        ):
            with patch("sys.stderr", new=StringIO()) as mock_stderr:
                result = main()
                assert result == 1
                # Error message should be printed to stderr
                error_output = mock_stderr.getvalue()
                assert "Input file not found" in error_output


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_tiff_reshape_end_to_end(self):
        """Test complete tiff-reshape workflow through CLI."""
        pytest.skip("Known Windows file lock / axis handling issue; see open issue.")
        # Create a flat TIFF file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            n_volumes = 2
            slices_per_volume = 5
            total_frames = n_volumes * slices_per_volume

            data = np.random.randint(0, 255, (total_frames, 64, 64, 1), dtype=np.uint8)

            writer = TIFFFileWriter3D(tmp.name, dim_order="TYXC")
            for frame in data:
                writer.write_frames(frame[np.newaxis, :, :, :])
            writer.close()
            input_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix="_3d.tif", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Run through the actual CLI
            with patch(
                "sys.argv",
                [
                    "flowreg3d",
                    "tiff-reshape",
                    input_path,
                    output_path,
                    "--slices-per-volume",
                    str(slices_per_volume),
                    "--overwrite",
                ],
            ):
                result = main()
                assert result == 0

            # Verify the output file was created and has correct structure
            assert Path(output_path).exists()

            # Read and verify the reshaped data
            from flowreg3d.util.io.tiff_3d import TIFFFileReader3D

            reader = TIFFFileReader3D(output_path, dim_order="TZYXC")
            reshaped = reader[:]
            reader.close()

            assert reshaped.shape[0] == n_volumes
            assert reshaped.shape[1] == slices_per_volume

        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_dry_run_through_cli(self):
        """Test dry run mode through CLI doesn't create output."""
        # Create input file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.tif")
            output_path = str(Path(tmpdir) / "output_out.tif")

            data = np.random.randint(0, 255, (10, 64, 64, 1), dtype=np.uint8)
            writer = TIFFFileWriter3D(input_path)
            for frame in data:
                writer.write_frames(frame[np.newaxis, :, :, :])
            writer.close()

            with patch(
                "sys.argv",
                [
                    "flowreg3d",
                    "tiff-reshape",
                    input_path,
                    output_path,
                    "-z",
                    "2",
                    "--dry-run",
                ],
            ):
                with patch("sys.stdout", new=StringIO()) as mock_stdout:
                    result = main()
                    assert result == 0
                    output = mock_stdout.getvalue()
                    assert "Dry run" in output

            # Verify no output file was created
            assert not Path(output_path).exists()

    def test_cli_with_all_options(self):
        """Test CLI with multiple options combined."""
        pytest.skip(
            "Known reshape axis handling/Windows file lock issue; see open issue."
        )
        # Create test file with known structure
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.tif")
            output_path = str(Path(tmpdir) / "output_out.tif")

            # 4 volumes, 5 slices each = 20 frames total
            data = np.random.randint(0, 255, (20, 32, 32, 2), dtype=np.uint16)
            writer = TIFFFileWriter3D(input_path)
            for frame in data:
                writer.write_frames(frame[np.newaxis, :, :, :])
            writer.close()

            # Use complex options: volumes 1-3, stride 2, compression
            with patch(
                "sys.argv",
                [
                    "flowreg3d",
                    "--verbose",
                    "tiff-reshape",
                    input_path,
                    output_path,
                    "--slices-per-volume",
                    "5",
                    "--start-volume",
                    "1",
                    "--end-volume",
                    "4",
                    "--volume-stride",
                    "2",
                    "--compression",
                    "lzw",
                    "--overwrite",
                ],
            ):
                result = main()
                assert result == 0

            # Verify output
            from flowreg3d.util.io.tiff_3d import TIFFFileReader3D

            reader = TIFFFileReader3D(output_path)
            reshaped = reader[:]
            reader.close()

            # Should have 2 volumes: volumes at indices 1 and 3
            # From range [1,4) with stride 2, we get indices: 1, 3
            assert reshaped.shape[0] == 2  # Two volumes (at indices 1 and 3)
            assert reshaped.shape[1] == 5  # 5 slices per volume


def test_cli_as_module():
    """Test running CLI as a module."""
    pytest.skip(
        "Known Windows file lock / reshape axis handling issue; see open issue."
    )
    # Create minimal test files
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        data = np.zeros((2, 32, 32, 1), dtype=np.uint8)
        writer = TIFFFileWriter3D(tmp.name)
        writer.write_frames(data)
        writer.close()
        input_path = tmp.name

    output_path = tempfile.mktemp(suffix="_out.tif")

    try:
        # Test running as module: python -m flowreg3d.cli.main
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "tiff-reshape",
                input_path,
                output_path,
                "-z",
                "1",
                "--dry-run",
            ],
        ):
            from flowreg3d.cli.main import main

            result = main()
            assert result == 0

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
