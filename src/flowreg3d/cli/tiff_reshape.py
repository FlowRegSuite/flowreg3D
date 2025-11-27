"""
TIFF reshape command for converting flat TIFFs to proper 3D volumetric stacks.

This tool handles conversion of flat TIFF files (where Z slices are stored as
sequential frames) into properly formatted 3D TIFF stacks that can be read by
napari and the 3D motion correction pipeline.

Uses the 2D TIFF reader with ScanImage support for auto-detection of structure,
then writes proper 3D formatted output.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from flowreg3d.util.resize_util_3D import imresize_fused_gauss_cubic3D

# Import the 2D reader with ScanImage support for reading flat files
from flowreg3d.util.io.tiff import TIFFFileReader
from flowreg3d.util.io._scanimage import (
    parse_scanimage_metadata,
    format_scanimage_metadata_report,
)

# Import 3D writer for properly formatted output
from flowreg3d.util.io.tiff_3d import TIFFFileWriter3D


class ReshapeTIFFReader(TIFFFileReader):
    """Specialized reader that preserves multi-channel ImageJ hyperstacks.

    ScanImage recordings store channels as interleaved pages, which the base
    ``TIFFFileReader`` already handles via ``deinterleave``. ImageJ hyperstacks
    (and other axis-aware TIFFs) instead encode channels along an explicit axis
    (e.g., ``TZCYX``). The base reader falls back to page-wise reading in that
    case, returning only the first channel. For the reshape CLI we prefer
    tifffile's series-aware path whenever channels live on their own axis while
    keeping the ScanImage behavior untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_reader_used = False

    def _should_use_series_reader(self) -> bool:
        """Return True when we should rely on tifffile's multi-axis series."""

        # Preserve ScanImage semantics entirely – it expects page-wise access
        # with optional deinterleaving.
        if getattr(self, "_is_scanimage", False):
            return False

        if getattr(self, "deinterleave", 1) > 1:
            return False

        series = getattr(self, "_tiff_series", None)
        if series is None:
            return False

        axes = getattr(series, "axes", "") or ""
        if not axes:
            return False

        # Only switch readers if a dedicated channel/sample axis exists.
        has_channel_axis = "C" in axes or ("S" in axes and self.n_channels > 1)
        if not has_channel_axis:
            return False

        return True

    def _read_raw_frames(self, frame_indices):  # type: ignore[override]
        """Use axis-aware reading when it preserves multi-channel data."""

        original_indices = frame_indices

        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            frame_indices = list(range(start, stop, step))
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = frame_indices.tolist()
        elif not isinstance(frame_indices, list):
            frame_indices = list(frame_indices)

        n_frames = len(frame_indices)
        if n_frames == 0:
            return np.empty(
                (0, self.height, self.width, self.n_channels), dtype=self.dtype
            )

        if self._should_use_series_reader():
            self._series_reader_used = True
            output = np.zeros(
                (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
            )
            self._read_series_mode(frame_indices, output)
            return output

        return super()._read_raw_frames(original_indices)

    @property
    def series_reader_used(self) -> bool:
        return self._series_reader_used

    @property
    def series_axes(self) -> str:
        series = getattr(self, "_tiff_series", None)
        return getattr(series, "axes", "") or ""

    def prefers_series_reader(self) -> bool:
        return self._should_use_series_reader()


def _parse_scale(scale_values):
    """Validate and normalize per-axis scale factors (X, Y, Z order)."""
    if scale_values is None:
        return None
    if len(scale_values) != 3:
        raise ValueError("Scale must be three floats: sx sy sz (X, Y, Z order)")
    sx, sy, sz = map(float, scale_values)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("Scale values must be positive")
    return sx, sy, sz


def _target_shape(zyx_shape, scale):
    """Return (Z, Y, X, C) after scaling."""
    if scale is None:
        return zyx_shape
    z, y, x, c = zyx_shape
    sx, sy, sz = scale
    return (
        max(1, int(round(z * sz))),
        max(1, int(round(y * sy))),
        max(1, int(round(x * sx))),
        c,
    )


def _resize_volume(volume, scale, target_zyx):
    """Resize a volume shaped (Z, Y, X, C) using fused pyramid resize."""
    if scale is None:
        return volume
    target_z, target_y, target_x = target_zyx[:3]
    return imresize_fused_gauss_cubic3D(volume, (target_z, target_y, target_x))


def add_tiff_reshape_parser(subparsers):
    """Add the tiff-reshape subcommand to the CLI parser."""
    parser = subparsers.add_parser(
        "tiff-reshape",
        help="Convert flat TIFF files to proper 3D volumetric stacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Convert flat TIFF files to proper 3D volumetric stacks.

This tool reshapes TIFF files where 3D volumes are stored as sequential 2D slices
into properly formatted 3D stacks with explicit volume structure.

The tool can:
- Auto-detect slices per volume from ScanImage metadata
- Manually specify slices per volume
- Extract specific volume ranges
- Apply temporal stride/sampling
- Handle multi-channel data

Output format will be TZYXC (Time, Z, Y, X, Channels) which is compatible with:
- napari 3D visualization
- FlowReg3D motion correction
- Standard 3D analysis tools
        """,
        epilog="""
Examples:
  # Auto-detect structure from ScanImage metadata
  %(prog)s input.tif output.tif

  # Manually specify 30 slices per volume
  %(prog)s input.tif output.tif --slices-per-volume 30

  # Extract volumes 10-50
  %(prog)s input.tif output.tif --start-volume 10 --end-volume 50

  # Sample every 2nd volume
  %(prog)s input.tif output.tif --volume-stride 2

  # Combine: volumes 5-20, every 2nd, with 25 slices each
  %(prog)s input.tif output.tif -z 25 --start-volume 5 --end-volume 20 --volume-stride 2

  # Dry run to check detected parameters
  %(prog)s input.tif output.tif --dry-run
        """,
    )

    # Positional arguments
    parser.add_argument(
        "input_file", type=str, help="Input TIFF file (flat or improperly formatted)"
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output TIFF file (will be properly formatted as TZYXC)",
    )

    # Structure specification
    structure_group = parser.add_argument_group("Structure specification")
    structure_group.add_argument(
        "--slices-per-volume",
        "-z",
        type=int,
        default=None,
        help="Number of Z slices per volume (auto-detect if not specified)",
    )

    structure_group.add_argument(
        "--frames-per-slice",
        "-f",
        type=int,
        default=1,
        help="Number of frames per Z slice for averaging (default: 1)",
    )

    # Volume selection
    selection_group = parser.add_argument_group("Volume selection")
    selection_group.add_argument(
        "--start-volume",
        "-s",
        type=int,
        default=None,
        help="First volume to extract (0-based index)",
    )

    selection_group.add_argument(
        "--end-volume",
        "-e",
        type=int,
        default=None,
        help="Last volume to extract (exclusive, like Python slicing)",
    )

    selection_group.add_argument(
        "--volume-stride",
        "--stride",
        type=int,
        default=1,
        help="Extract every Nth volume (default: 1 = all volumes)",
    )

    # Processing options
    processing_group = parser.add_argument_group("Processing options")
    processing_group.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Number of channels (auto-detect if not specified)",
    )

    processing_group.add_argument(
        "--dim-order",
        type=str,
        default=None,
        help="Dimension order of input file if known (e.g., TZYX, ZTYX)",
    )

    processing_group.add_argument(
        "--scale",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        default=None,
        help="Scale factors for X, Y, Z axes (e.g., 0.25 0.25 1.0) before writing",
    )

    processing_group.add_argument(
        "--compression",
        type=str,
        choices=["none", "lzw", "zlib", "jpeg"],
        default="lzw",
        help="Compression for output file (default: lzw)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output-dim-order",
        type=str,
        default="TZYXC",
        help="Dimension order for output file (default: TZYXC)",
    )

    output_group.add_argument(
        "--imagej", action="store_true", help="Write ImageJ-compatible metadata"
    )

    output_group.add_argument(
        "--split-channels",
        action="store_true",
        help="Write one output file per channel (appends _ch{index} before the extension)",
    )

    # Utility options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected parameters without writing output",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    # Set the function to call
    parser.set_defaults(func=reshape_tiff)

    return parser


def reshape_tiff(args):
    """Execute the TIFF reshape operation."""
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Validate output file
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        print(f"Error: Output file exists: {output_path}", file=sys.stderr)
        print("Use --overwrite to replace it", file=sys.stderr)
        return 1

    try:
        scale = _parse_scale(args.scale)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {input_path}")
    print(f"Output to:  {output_path}")
    print()

    # Parse ScanImage metadata first
    print("Checking for ScanImage metadata...")
    si_metadata = parse_scanimage_metadata(str(input_path))

    if si_metadata.get("is_scanimage"):
        print("ScanImage file detected!")
        print(format_scanimage_metadata_report(si_metadata))
        print()

        # Auto-detect slices per volume if not specified
        if args.slices_per_volume is None:
            args.slices_per_volume = si_metadata.get("slices_per_volume", 1)
            if args.slices_per_volume > 1:
                print(f"Auto-detected {args.slices_per_volume} slices per volume")

        # Auto-detect frames per slice if not specified
        if args.frames_per_slice == 1 and si_metadata.get("frames_per_slice", 1) > 1:
            args.frames_per_slice = si_metadata["frames_per_slice"]
            print(
                f"Auto-detected {args.frames_per_slice} frames per slice for averaging"
            )
    else:
        print("Not a ScanImage file or metadata not found")

    # Validate slices per volume
    if args.slices_per_volume is None:
        print("Error: Could not auto-detect slices per volume.", file=sys.stderr)
        print("Please specify with --slices-per-volume", file=sys.stderr)
        return 1

    # Open the input file using 2D reader
    print("\nOpening input file...")
    try:
        reader = ReshapeTIFFReader(
            str(input_path),
            buffer_size=100,  # Smaller buffer for 3D data
            bin_size=1,  # No binning
            deinterleave=1,  # Handle deinterleaving if needed
        )
    except Exception as e:
        print(f"Error opening input file: {e}", file=sys.stderr)
        return 1

    print(f"Input shape: {reader.shape}")
    print(f"Data type: {reader.dtype}")
    print(f"Total frames: {reader.frame_count}")

    if isinstance(reader, ReshapeTIFFReader) and reader.prefers_series_reader():
        axes = reader.series_axes or "unknown"
        print(f"\nDetected multi-axis layout (axes={axes})")
        print("Using series-aware channel reader to preserve multi-channel data")

    # Calculate volume structure
    slices_per_volume = args.slices_per_volume
    frames_per_slice = args.frames_per_slice

    # Total frames should be divisible by (slices_per_volume * frames_per_slice)
    frames_per_volume = slices_per_volume * frames_per_slice
    total_volumes = reader.frame_count // frames_per_volume

    if reader.frame_count % frames_per_volume != 0:
        print(f"\nWarning: Total frames ({reader.frame_count}) not evenly divisible by")
        print(f"         frames per volume ({frames_per_volume})")
        print("         Last incomplete volume will be discarded")

    print("\nDetected structure:")
    print(f"  {total_volumes} volumes")
    print(f"  {slices_per_volume} slices per volume")
    if frames_per_slice > 1:
        print(f"  {frames_per_slice} frames per slice (will be averaged)")
    print(f"  {reader.height} x {reader.width} pixels")
    print(f"  {reader.n_channels} channel(s)")
    base_zyx = (slices_per_volume, reader.height, reader.width, reader.n_channels)
    target_zyx = _target_shape(base_zyx, scale)
    if scale is not None:
        print(
            f"  Scaling (X, Y, Z): {scale} -> output per volume "
            f"(Z={target_zyx[0]}, Y={target_zyx[1]}, X={target_zyx[2]})"
        )
    else:
        print(
            f"  Output per volume: (Z={target_zyx[0]}, Y={target_zyx[1]}, X={target_zyx[2]})"
        )

    # Apply volume selection
    start_vol = args.start_volume if args.start_volume is not None else 0
    end_vol = args.end_volume if args.end_volume is not None else total_volumes
    stride = args.volume_stride

    # Validate range
    start_vol = max(0, min(start_vol, total_volumes - 1))
    end_vol = max(start_vol + 1, min(end_vol, total_volumes))

    # Calculate selected volumes
    selected_volumes = list(range(start_vol, end_vol, stride))
    n_selected = len(selected_volumes)

    print("\nVolume selection:")
    print(f"  Range: [{start_vol}, {end_vol})")
    print(f"  Stride: {stride}")
    print(f"  Selected: {n_selected} volumes")

    if args.dry_run:
        print("\nDry run - no output written")
        reader.close()
        return 0

    # Process and write data
    print("\nReshaping data...")

    # Create output array shape: (T, Z, Y, X, C)
    output_shape = (
        n_selected,
        target_zyx[0],
        target_zyx[1],
        target_zyx[2],
        reader.n_channels,
    )

    print(f"Output shape: {output_shape}")

    split_channels = bool(args.split_channels)
    if split_channels and reader.n_channels < 2:
        print(
            "Split-channels requested but only 1 channel detected; writing a single output file."
        )
        split_channels = False

    if split_channels:
        channel_paths = [
            output_path.with_name(f"{output_path.stem}_ch{ch}{output_path.suffix}")
            for ch in range(reader.n_channels)
        ]
        print("\nChannel outputs:")
        for ch, path in enumerate(channel_paths):
            print(f"  Channel {ch}: {path}")

        writers = [
            TIFFFileWriter3D(str(path), dim_order=args.output_dim_order)
            for path in channel_paths
        ]
    else:
        # Create writer
        writer = TIFFFileWriter3D(str(output_path), dim_order=args.output_dim_order)

    # Process each selected volume
    for vol_idx, vol_num in enumerate(selected_volumes):
        if args.verbose or (vol_idx % max(1, n_selected // 10) == 0):
            print(
                f"Processing volume {vol_idx + 1}/{n_selected} (original #{vol_num})..."
            )

        # Calculate frame indices for this volume
        frame_start = vol_num * frames_per_volume
        frame_end = frame_start + frames_per_volume

        # Read all frames for this volume
        volume_frames = reader[
            frame_start:frame_end
        ]  # Shape: (frames_per_volume, H, W, C)

        # Ensure correct dtype (reader may return float64)
        if volume_frames.dtype != reader.dtype:
            volume_frames = volume_frames.astype(reader.dtype)

        # Reshape based on frames_per_slice
        if frames_per_slice > 1:
            # Average frames at each Z position
            volume_data = np.zeros(
                (slices_per_volume, reader.height, reader.width, reader.n_channels),
                dtype=np.float32,
            )

            for z in range(slices_per_volume):
                slice_start = z * frames_per_slice
                slice_end = slice_start + frames_per_slice
                # Average frames for this Z slice
                volume_data[z] = volume_frames[slice_start:slice_end].mean(axis=0)

            # Convert back to original dtype
            volume_data = volume_data.astype(reader.dtype)
        else:
            # Direct reshape - frames are Z slices
            volume_data = volume_frames.reshape(
                slices_per_volume, reader.height, reader.width, reader.n_channels
            )

        volume_data = _resize_volume(volume_data, scale, target_zyx)

        if split_channels:
            # Write each channel to its own file, preserving ZYX ordering
            for ch_idx, ch_writer in enumerate(writers):
                ch_writer.write_frames(volume_data[..., ch_idx : ch_idx + 1])
        else:
            # Write this volume (writer expects (T, Z, Y, X, C) or (Z, Y, X, C) for single)
            writer.write_frames(volume_data)

    # Close files
    if split_channels:
        for ch_writer in writers:
            ch_writer.close()
    else:
        writer.close()
    reader.close()

    if split_channels:
        print("\nSuccess! Channel outputs written:")
        for ch, path in enumerate(channel_paths):
            print(f"  Channel {ch}: {path}")
        print(
            f"Final shape per channel file: (T={n_selected}, Z={target_zyx[0]}, "
            f"Y={target_zyx[1]}, X={target_zyx[2]}, C=1)"
        )
    else:
        print(f"\nSuccess! Output written to: {output_path}")
        print(
            f"Final shape: (T={n_selected}, Z={target_zyx[0]}, "
            f"Y={target_zyx[1]}, X={target_zyx[2]}, C={reader.n_channels})"
        )

    return 0


def main():
    """Standalone entry point for testing."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_tiff_reshape_parser(subparsers)

    # Simulate tiff-reshape command
    args = parser.parse_args(["tiff-reshape"] + sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
