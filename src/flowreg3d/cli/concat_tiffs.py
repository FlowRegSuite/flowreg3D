"""
Concatenate per-volume 3D files from a folder into a single TIFF movie.

Each file in the folder is treated as one timepoint containing a 3D volume
(Z, Y, X, [C]). Files are ordered lexicographically, normalized to TZYXC, and
stacked into an ImageJ-compatible 3D TIFF movie.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile

from flowreg3d.util.io.tiff_3d import TIFFFileWriter3D


def _discover_files(folder: Path, pattern: str) -> List[Path]:
    """Return a sorted list of files in ``folder`` matching ``pattern``."""
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def _strip_suffix(name: str, suffix: str) -> Optional[str]:
    """Remove suffix from name if present; return None when it does not match."""
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else None


def _discover_channel_files(folder: Path, suffixes: List[str]) -> List[List[Path]]:
    """Return matched file lists per channel suffix, validating alignment."""
    channel_lists: List[List[Path]] = []

    for suffix in suffixes:
        matched = sorted(path for path in folder.glob(f"*{suffix}") if path.is_file())
        if not matched:
            raise ValueError(f"No files found matching '*{suffix}' in {folder}")
        channel_lists.append(matched)

    lengths = {len(lst) for lst in channel_lists}
    if len(lengths) != 1:
        raise ValueError(
            f"Channel counts differ across suffixes: {[len(lst) for lst in channel_lists]}"
        )

    n_items = lengths.pop()
    for idx in range(n_items):
        bases = []
        for suffix, paths in zip(suffixes, channel_lists):
            base = _strip_suffix(paths[idx].name, suffix)
            if base is None:
                raise ValueError(
                    f"File name '{paths[idx].name}' does not end with expected suffix '{suffix}'"
                )
            bases.append(base)
        if not all(b == bases[0] for b in bases):
            raise ValueError(
                f"File mismatch at index {idx}: {[paths[idx].name for paths in channel_lists]}"
            )

    return channel_lists


def _load_volume(path: Path, dim_order: Optional[str] = None) -> np.ndarray:
    """
    Load a single 3D volume (one timepoint) and normalize to TZYXC.

    Args:
        path: File path to read.
        dim_order: Optional explicit axis order (e.g., ZYX, ZYXC, TZYXC).

    Returns:
        Volume with shape (1, Z, Y, X, C).
    """
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = (
            dim_order
            or (tif.imagej_metadata or {}).get("axes")
            or getattr(series, "axes", "")  # type: ignore[attr-defined]
            or ""
        )

    data = np.asarray(data)
    axes = axes.upper() if axes else ""

    # Some ImageJ exports use "I" as an index axis instead of Z; map when Z is missing.
    if axes and "Z" not in axes and "I" in axes:
        axes = axes.replace("I", "Z")
    if axes and "Z" not in axes and "S" in axes:
        # Some writers use S (samples) where we expect Z; map the first S to Z.
        axes = axes.replace("S", "Z", 1)

    # If axes still doesn't cover all required dims, fall back to inference.
    required_dims = set("TZYXC")
    if axes:
        if not required_dims.issubset(set(axes)):
            axes = ""

    if axes:
        if len(axes) != data.ndim:
            raise ValueError(
                f"Axes '{axes}' do not match data ndim ({data.ndim}) for {path.name}"
            )
    else:
        if data.ndim == 3:
            axes = "ZYX"
        elif data.ndim == 4:
            axes = "ZYXC"
        elif data.ndim == 5:
            axes = "TZYXC"
        else:
            raise ValueError(f"Unable to infer axes for {path.name}: ndim={data.ndim}")

    if "T" not in axes:
        axes = "T" + axes
        data = data[np.newaxis, ...]

    if "C" not in axes:
        axes = axes + "C"
        data = data[..., np.newaxis]

    try:
        transpose_order = [axes.index(dim) for dim in "TZYXC"]
    except ValueError as exc:
        raise ValueError(
            f"Missing required axes in {path.name}: expected T, Z, Y, X, C coverage (got '{axes}')"
        ) from exc

    volume = np.transpose(data, transpose_order)

    if volume.shape[0] != 1:
        raise ValueError(
            f"{path.name} contains multiple timepoints (T={volume.shape[0]}), "
            "expected exactly one per file."
        )

    return volume


def add_concat_tiffs_parser(subparsers):
    """Add the concat-tiffs subcommand to the CLI parser."""
    parser = subparsers.add_parser(
        "concat-tiffs",
        help="Concatenate per-volume 3D files from a folder into a TIFF movie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Concatenate per-volume 3D files from a folder into a single TIFF movie.

Each file is treated as one timepoint containing a 3D volume (Z, Y, X, [C]).
Files are read in sorted order and stacked along the time axis into a TZYXC
ImageJ hyperstack compatible with FlowReg3D and napari.
        """,
        epilog="""
Examples:
  # Concatenate all .tif/.tiff files in a folder
  %(prog)s /data/frames output_movie.tif

  # Use a custom pattern and explicit axis order
  %(prog)s /data/frames output_movie.tif --pattern "frame_*.tif" --dim-order ZYXC

  # Dry run to inspect detected shapes without writing
  %(prog)s /data/frames output_movie.tif --dry-run
        """,
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing per-frame 3D files (each a single timepoint)",
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output TIFF movie path (written as TZYXC ImageJ hyperstack)",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*.tif*",
        help="Glob pattern for input files (default: *.tif*)",
    )

    parser.add_argument(
        "--dim-order",
        type=str,
        default=None,
        help="Axis order of input files if metadata is missing (e.g., ZYX, ZYXC, TZYXC)",
    )

    parser.add_argument(
        "--channel-suffixes",
        nargs="+",
        default=None,
        help=(
            "Treat each suffix as a distinct channel and align files by shared basename, "
            "e.g., --channel-suffixes _ch1.tiff _ch2.tif"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected files and shapes without writing output",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file details during concatenation",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    parser.add_argument(
        "--output-dim-order",
        type=str,
        default="TZYXC",
        help="Dimension order for output file (default: TZYXC)",
    )

    parser.add_argument(
        "--split-channels",
        action="store_true",
        help="Write one output file per channel (appends _ch{index} before the extension)",
    )

    parser.set_defaults(func=concat_tiffs)

    return parser


def concat_tiffs(args):
    """Concatenate 3D frame files into a single TIFF movie."""
    input_dir = Path(args.input_folder)
    if not input_dir.exists() or not input_dir.is_dir():
        print(
            f"Error: Input folder not found or not a directory: {input_dir}",
            file=sys.stderr,
        )
        return 1

    channel_suffixes = getattr(args, "channel_suffixes", None)
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        print(f"Error: Output file exists: {output_path}", file=sys.stderr)
        print("Use --overwrite to replace it", file=sys.stderr)
        return 1

    try:
        if channel_suffixes:
            channel_suffixes = list(channel_suffixes)
            channel_files = _discover_channel_files(input_dir, channel_suffixes)
            n_volumes = len(channel_files[0])

            first_volumes = [
                _load_volume(paths[0], args.dim_order) for paths in channel_files
            ]
            zyx_shape = first_volumes[0].shape[1:]
            dtype = first_volumes[0].dtype
            total_channels = sum(vol.shape[-1] for vol in first_volumes)

            for vol in first_volumes[1:]:
                if vol.shape[1:] != zyx_shape:
                    raise ValueError("Channel volumes have mismatched shapes.")
                dtype = np.promote_types(dtype, vol.dtype)

            print(
                f"Found {n_volumes} volume pairs across {len(channel_suffixes)} channels."
            )
            print(
                f"Detected per-file shape: (Z={zyx_shape[0]}, Y={zyx_shape[1]}, X={zyx_shape[2]}, C_total={total_channels})"
            )

            if args.verbose:
                for idx in range(n_volumes):
                    names = [paths[idx].name for paths in channel_files]
                    print(f"  [{idx:03d}] {', '.join(names)}")
        else:
            files = _discover_files(input_dir, args.pattern)
            if not files:
                print(
                    f"Error: No files found in {input_dir} matching pattern '{args.pattern}'",
                    file=sys.stderr,
                )
                return 1

            first_volume = _load_volume(files[0], args.dim_order)
            zyx_shape = first_volume.shape[1:]
            dtype = first_volume.dtype
            total_channels = zyx_shape[3]
            n_volumes = len(files)

            print(f"Found {n_volumes} files to concatenate.")
            if args.verbose:
                for idx, path in enumerate(files):
                    print(f"  [{idx:03d}] {path.name}")

        print(f"Data type: {dtype}")

        if args.dry_run:
            print("\nDry run - no output written")
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle split-channels option
        split_channels = bool(getattr(args, "split_channels", False))
        if split_channels and total_channels < 2:
            print(
                "Split-channels requested but only 1 channel detected; writing a single output file."
            )
            split_channels = False

        if split_channels:
            channel_paths = [
                output_path.with_name(f"{output_path.stem}_ch{ch}{output_path.suffix}")
                for ch in range(total_channels)
            ]
            print("\nChannel outputs:")
            for ch, path in enumerate(channel_paths):
                print(f"  Channel {ch}: {path}")

            writers = [
                TIFFFileWriter3D(
                    str(path),
                    dim_order=args.output_dim_order,
                    imagej=True,
                    expected_frames=n_volumes,
                )
                for path in channel_paths
            ]
        else:
            writer = TIFFFileWriter3D(
                str(output_path),
                dim_order=args.output_dim_order,
                imagej=True,
                expected_frames=n_volumes,
            )

        if channel_suffixes:
            for idx in range(n_volumes):
                if args.verbose:
                    names = [paths[idx].name for paths in channel_files]
                    print(f"Reading {', '.join(names)}...")

                vols = []
                for paths in channel_files:
                    vol = _load_volume(paths[idx], args.dim_order)
                    if vol.shape[1:] != zyx_shape:
                        raise ValueError(
                            f"Shape mismatch for {paths[idx].name}: expected {zyx_shape}, got {vol.shape[1:]}"
                        )
                    if vol.dtype != dtype:
                        vol = vol.astype(dtype)
                    vols.append(vol)

                combined = np.concatenate(vols, axis=-1)

                if split_channels:
                    # Write each channel to its own file
                    for ch_idx, ch_writer in enumerate(writers):
                        ch_writer.write_frames(combined[..., ch_idx : ch_idx + 1])
                else:
                    writer.write_frames(combined)

                if args.verbose:
                    print(f"Appended volume {idx + 1}/{n_volumes}")
        else:
            for idx, path in enumerate(files):
                if args.verbose:
                    print(f"Reading {path.name}...")

                volume = _load_volume(path, args.dim_order)

                if volume.shape[1:] != zyx_shape:
                    raise ValueError(
                        f"Shape mismatch for {path.name}: expected {zyx_shape}, got {volume.shape[1:]}"
                    )

                if volume.dtype != dtype:
                    volume = volume.astype(dtype)

                if split_channels:
                    # Write each channel to its own file
                    for ch_idx, ch_writer in enumerate(writers):
                        ch_writer.write_frames(volume[..., ch_idx : ch_idx + 1])
                else:
                    writer.write_frames(volume)

                if args.verbose:
                    print(f"Appended volume {idx + 1}/{n_volumes}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Close writer(s) if they exist
        if "writers" in locals():
            for ch_writer in writers:
                ch_writer.close()
        elif "writer" in locals():
            writer.close()

    if split_channels:
        print("\nSuccess! Channel outputs written:")
        for ch, path in enumerate(channel_paths):
            print(f"  Channel {ch}: {path}")
        print(
            f"Final shape per channel file: (T={n_volumes}, Z={zyx_shape[0]}, "
            f"Y={zyx_shape[1]}, X={zyx_shape[2]}, C=1)"
        )
    else:
        print(f"\nSuccess! Output written to: {output_path}")
        print(
            f"Final shape: (T={n_volumes}, Z={zyx_shape[0]}, "
            f"Y={zyx_shape[1]}, X={zyx_shape[2]}, C={total_channels})"
        )

    return 0
