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

    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        print(f"Error: Output file exists: {output_path}", file=sys.stderr)
        print("Use --overwrite to replace it", file=sys.stderr)
        return 1

    files = _discover_files(input_dir, args.pattern)
    if not files:
        print(
            f"Error: No files found in {input_dir} matching pattern '{args.pattern}'",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(files)} files to concatenate.")
    if args.verbose:
        for idx, path in enumerate(files):
            print(f"  [{idx:03d}] {path.name}")

    first_volume = _load_volume(files[0], args.dim_order)
    zyx_shape = first_volume.shape[1:]
    dtype = first_volume.dtype

    print(
        f"Detected volume shape per file: (Z={zyx_shape[0]}, Y={zyx_shape[1]}, X={zyx_shape[2]}, C={zyx_shape[3]})"
    )
    print(f"Data type: {dtype}")

    if args.dry_run:
        print("\nDry run - no output written")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = TIFFFileWriter3D(str(output_path), dim_order=args.output_dim_order)

    try:
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

            writer.write_frames(volume)

            if args.verbose:
                print(f"Appended volume {idx + 1}/{len(files)}")
    finally:
        writer.close()

    print(f"\nSuccess! Output written to: {output_path}")
    print(
        f"Final shape: (T={len(files)}, Z={zyx_shape[0]}, "
        f"Y={zyx_shape[1]}, X={zyx_shape[2]}, C={zyx_shape[3]})"
    )

    return 0
