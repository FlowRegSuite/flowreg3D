"""
Main entry point for FlowReg3D CLI.

Provides subcommands for various 3D data processing tasks.
"""

import argparse
import sys


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="flowreg3d",
        description="FlowReg3D - 3D Motion Correction and Data Processing Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reshape flat TIFF to 3D volumes (auto-detect from metadata)
  flowreg3d tiff-reshape input.tif output.tif

  # Specify slices per volume manually
  flowreg3d tiff-reshape input.tif output.tif --slices-per-volume 30

  # Extract specific volume range
  flowreg3d tiff-reshape input.tif output.tif --start-volume 10 --end-volume 50

  # Use stride to sample every Nth volume
  flowreg3d tiff-reshape input.tif output.tif --volume-stride 2
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Import and register subcommands
    from flowreg3d.cli.tiff_reshape import add_tiff_reshape_parser
    from flowreg3d.cli.concat_tiffs import add_concat_tiffs_parser

    add_tiff_reshape_parser(subparsers)
    add_concat_tiffs_parser(subparsers)

    # Future commands can be added here:
    # from flowreg3d.cli.motion_correct import add_motion_correct_parser
    # add_motion_correct_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if hasattr(args, "func"):
        try:
            return args.func(args)
        except Exception as e:
            if hasattr(args, "verbose") and args.verbose:
                import traceback

                traceback.print_exc()
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
