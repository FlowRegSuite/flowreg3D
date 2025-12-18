"""
Tests for concatenating per-volume 3D files into a single TIFF movie.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from flowreg3d.cli.concat_tiffs import add_concat_tiffs_parser, concat_tiffs
from flowreg3d.util.io.tiff_3d import TIFFFileReader3D, TIFFFileWriter3D


def _write_volume(path: Path, volume: np.ndarray):
    writer = TIFFFileWriter3D(str(path))
    writer.write_frames(volume)
    writer.close()


def test_concat_tiffs_basic():
    """Concatenate three per-volume files into one movie."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "frames"
        input_dir.mkdir()
        output_path = Path(tmpdir) / "movie.tif"

        n_volumes = 3
        volume_shape = (4, 8, 8, 1)  # Z, Y, X, C
        volumes = []

        for idx in range(n_volumes):
            vol = np.full(volume_shape, idx, dtype=np.uint16)
            volumes.append(vol)
            _write_volume(input_dir / f"frame_{idx:03d}.tif", vol)

        args = argparse.Namespace(
            input_folder=str(input_dir),
            output_file=str(output_path),
            pattern="*.tif",
            dim_order=None,
            scale=None,
            dry_run=False,
            verbose=False,
            overwrite=True,
            output_dim_order="TZYXC",
        )

        result = concat_tiffs(args)
        assert result == 0
        assert output_path.exists()

        reader = TIFFFileReader3D(str(output_path))
        stacked = reader[:]
        reader.close()

        expected = np.stack(volumes, axis=0)
        np.testing.assert_array_equal(stacked, expected)


def test_concat_tiffs_dry_run():
    """Dry run should not write output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "frames"
        input_dir.mkdir()
        output_path = Path(tmpdir) / "movie.tif"

        vol = np.zeros((2, 4, 4, 1), dtype=np.uint8)
        _write_volume(input_dir / "frame_000.tif", vol)

        args = argparse.Namespace(
            input_folder=str(input_dir),
            output_file=str(output_path),
            pattern="*.tif",
            dim_order=None,
            scale=None,
            dry_run=True,
            verbose=False,
            overwrite=False,
            output_dim_order="TZYXC",
        )

        result = concat_tiffs(args)
        assert result == 0
        assert not output_path.exists()


def test_concat_tiffs_parser():
    """Parser should configure the concat-tiffs command."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    add_concat_tiffs_parser(subparsers)

    args = parser.parse_args(["concat-tiffs", "/input", "/output.tif"])
    assert args.input_folder == "/input"
    assert args.output_file == "/output.tif"
    assert args.pattern == "*.tif*"
    assert args.scale is None

    args = parser.parse_args(
        [
            "concat-tiffs",
            "/input",
            "/output.tif",
            "--channel-suffixes",
            "_ch1.tif",
            "_ch2.tif",
            "--scale",
            "0.5",
            "0.5",
            "1",
        ]
    )
    assert args.channel_suffixes == ["_ch1.tif", "_ch2.tif"]
    assert args.scale == [0.5, 0.5, 1.0]


def test_concat_tiffs_multichannel_suffixes():
    """Concatenate paired channel files using suffix matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "frames"
        input_dir.mkdir()
        output_path = Path(tmpdir) / "movie.tif"

        n_volumes = 2
        volume_shape = (2, 6, 6, 1)
        channel_suffixes = ["_ch1.tif", "_ch2.tif"]

        for idx in range(n_volumes):
            vol_a = np.full(volume_shape, idx, dtype=np.uint8)
            vol_b = np.full(volume_shape, idx + 10, dtype=np.uint8)

            _write_volume(input_dir / f"frame_{idx:03d}{channel_suffixes[0]}", vol_a)
            _write_volume(input_dir / f"frame_{idx:03d}{channel_suffixes[1]}", vol_b)

        args = argparse.Namespace(
            input_folder=str(input_dir),
            output_file=str(output_path),
            pattern="*.tif*",
            dim_order=None,
            channel_suffixes=channel_suffixes,
            scale=None,
            dry_run=False,
            verbose=False,
            overwrite=True,
            output_dim_order="TZYXC",
        )

        result = concat_tiffs(args)
        assert result == 0

        reader = TIFFFileReader3D(str(output_path))
        stacked = reader[:]
        reader.close()

        assert stacked.shape == (n_volumes, *volume_shape[:-1], 2)
        np.testing.assert_array_equal(
            stacked[0, ..., 0], np.full(volume_shape[:-1], 0, dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            stacked[0, ..., 1], np.full(volume_shape[:-1], 10, dtype=np.uint8)
        )


def test_concat_tiffs_with_scale():
    """Apply scaling to volumes before concatenation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "frames"
        input_dir.mkdir()
        output_path = Path(tmpdir) / "movie_scaled.tif"

        n_volumes = 2
        volume_shape = (4, 8, 8, 1)  # Z, Y, X, C
        scale = [0.5, 0.5, 1.0]  # X, Y, Z order
        expected_shape = (
            n_volumes,
            int(round(volume_shape[0] * scale[2])),
            int(round(volume_shape[1] * scale[1])),
            int(round(volume_shape[2] * scale[0])),
            1,
        )

        for idx in range(n_volumes):
            vol = np.full(volume_shape, idx + 1, dtype=np.uint8)
            _write_volume(input_dir / f"frame_{idx:03d}.tif", vol)

        args = argparse.Namespace(
            input_folder=str(input_dir),
            output_file=str(output_path),
            pattern="*.tif",
            dim_order=None,
            scale=scale,
            dry_run=False,
            verbose=False,
            overwrite=True,
            output_dim_order="TZYXC",
        )

        result = concat_tiffs(args)
        assert result == 0

        reader = TIFFFileReader3D(str(output_path))
        stacked = reader[:]
        reader.close()

        assert stacked.shape == expected_shape
        # Constant volumes should remain constant after scaling
        assert stacked.min() == stacked.max()
        assert stacked[0].min() == 1
