"""
3D TIFF file reader and writer with support for volumetric time series.

Handles various dimension orderings (TXYZC, TZYXC, etc.) and provides
seamless integration with the 3D motion correction pipeline.
"""

import os
import warnings
from typing import Union, List, Optional
import numpy as np

try:
    import tifffile

    TIFF_SUPPORTED = True
except ImportError:
    TIFF_SUPPORTED = False
    warnings.warn("tifffile not installed. TIFF support unavailable.")

from flowreg3d.util.io._base_3d import VideoReader3D, VideoWriter3D


class TIFFFileReader3D(VideoReader3D):
    """
    3D TIFF reader supporting various dimension orderings.

    Supports:
    - Standard multi-page TIFFs with volumetric data
    - ImageJ hyperstacks with metadata
    - Flexible dimension interpretation via dim_order parameter
    - Memory-mapped reading for large files
    """

    def __init__(
        self,
        file_path: str,
        buffer_size: int = 10,
        bin_size: int = 1,
        dim_order: str = "TZYXC",
        **kwargs,
    ):
        """
        Initialize 3D TIFF reader.

        Args:
            file_path: Path to TIFF file
            buffer_size: Number of volumes per batch
            bin_size: Temporal binning factor
            dim_order: Dimension ordering in file (e.g., 'TZYXC', 'TXYZC', 'ZTXYC')
                      Must contain T, X, Y, Z. C is optional (assumes 1 if missing)
        """
        if not TIFF_SUPPORTED:
            raise ImportError("tifffile library required for TIFF support")

        super().__init__()

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.dim_order = dim_order.upper()

        # Validate dimension order
        required_dims = set("TXYZ")
        if not required_dims.issubset(set(self.dim_order)):
            raise ValueError(f"dim_order must contain T, X, Y, Z. Got: {dim_order}")

        # Internal state
        self._tiff_file = None
        self._data_array = None
        self._metadata = {}

        # Validate file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"TIFF file not found: {file_path}")

    def _initialize(self):
        """Open TIFF and parse dimensions."""
        try:
            # Open with tifffile
            self._tiff_file = tifffile.TiffFile(self.file_path)

            # Read as array - tifffile handles most formats automatically
            self._data_array = self._tiff_file.asarray()

            # Parse ImageJ metadata if available and update dim_order
            if (
                hasattr(self._tiff_file, "imagej_metadata")
                and self._tiff_file.imagej_metadata
            ):
                ij_meta = self._tiff_file.imagej_metadata
                self._metadata["imagej"] = ij_meta

                # Report detected structure
                if "frames" in ij_meta and "slices" in ij_meta:
                    print(
                        f"ImageJ hyperstack detected: {ij_meta.get('frames')} frames, "
                        f"{ij_meta.get('slices')} slices, {ij_meta.get('channels', 1)} channels"
                    )

                # Check if axes information is available in metadata
                if "axes" in ij_meta:
                    # Use the axes order from metadata
                    self.dim_order = ij_meta["axes"]
                elif (
                    "frames" in ij_meta
                    and "slices" in ij_meta
                    and "channels" in ij_meta
                ):
                    # ImageJ hyperstack is typically in TZCYX order when written by our writer
                    self.dim_order = "TZCYX"

            # Parse dimensions based on dim_order
            self._parse_dimensions()

            # Set dtype
            self.dtype = self._data_array.dtype

        except Exception as e:
            raise IOError(f"Failed to open 3D TIFF file: {e}")

    def _parse_dimensions(self):
        """Parse array dimensions according to dim_order."""
        shape = self._data_array.shape
        ndim = len(shape)

        # Handle case where C is implicit (single channel)
        if "C" not in self.dim_order:
            if ndim == len(self.dim_order):
                # Dimensions match exactly, add implicit C=1
                self.dim_order = self.dim_order + "C"
                self._data_array = self._data_array[..., np.newaxis]
                shape = self._data_array.shape
            elif ndim == len(self.dim_order) + 1:
                # Has channel dimension, update dim_order
                self.dim_order = self.dim_order + "C"
            else:
                raise ValueError(
                    f"Cannot parse dimensions. Array shape {shape} "
                    f"doesn't match dim_order '{self.dim_order}'"
                )
        else:
            # C is in dim_order but might be missing from actual data
            if ndim == len(self.dim_order) - 1:
                # Array is missing channel dimension, add it
                c_axis = self.dim_order.index("C")
                self._data_array = np.expand_dims(self._data_array, axis=c_axis)
                shape = self._data_array.shape

        # Verify dimension count matches
        if len(shape) != len(self.dim_order):
            raise ValueError(
                f"Dimension mismatch: array has {len(shape)} dims, "
                f"dim_order '{self.dim_order}' expects {len(self.dim_order)}"
            )

        # Create dimension mapping
        dim_map = {dim: idx for idx, dim in enumerate(self.dim_order)}

        # Extract dimensions
        self.frame_count = shape[dim_map["T"]]
        self.depth = shape[dim_map["Z"]]
        self.height = shape[dim_map["Y"]]
        self.width = shape[dim_map["X"]]
        self.n_channels = shape[dim_map["C"]] if "C" in dim_map else 1

        # Transpose to standard TZYXC order if needed
        target_order = "TZYXC"
        if self.dim_order != target_order:
            # Build transpose indices
            transpose_idx = []
            for dim in target_order:
                if dim in dim_map:
                    transpose_idx.append(dim_map[dim])

            self._data_array = np.transpose(self._data_array, transpose_idx)
            self.dim_order = target_order

        print(
            f"Parsed 3D TIFF: T={self.frame_count}, Z={self.depth}, "
            f"Y={self.height}, X={self.width}, C={self.n_channels}"
        )

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read specified timepoints.

        Returns:
            Array with shape (T, Z, Y, X, C)
        """
        if isinstance(frame_indices, slice):
            return self._data_array[frame_indices].copy()
        else:  # List of indices
            return self._data_array[frame_indices].copy()

    def close(self):
        """Close TIFF file."""
        if self._tiff_file is not None:
            self._tiff_file.close()
            self._tiff_file = None
        self._data_array = None


class TIFFFileWriter3D(VideoWriter3D):
    """
    3D TIFF writer for volumetric time series.

    Writes data in ImageJ hyperstack format for compatibility.
    """

    _METADATA_PADDING = 1000

    def __init__(
        self,
        file_path: str,
        dim_order: str = "TZYXC",
        compression: Optional[str] = None,
        bigtiff: bool = True,
        imagej: bool = True,
        expected_frames: Optional[int] = None,
        ome: bool = False,
        metadata: Optional[dict] = None,
        compression_level: int = 6,
    ):
        """
        Initialize 3D TIFF writer.

        Args:
            file_path: Output file path
            dim_order: Dimension ordering for output (default: 'TZYXC')
            compression: Compression type ('none', 'lzw', 'zlib', 'jpeg')
            bigtiff: Use BigTIFF format for files >4GB (default: True)
            imagej: Write ImageJ-compatible metadata (default: False)
            expected_frames: Optional total frame count (T). If provided, metadata uses
                this value so viewers keep the T dimension when streaming.
            ome: Write OME-TIFF metadata (default: False)
            metadata: Additional metadata dict to include
            compression_level: Compression level for zlib (0-9)
        """
        if not TIFF_SUPPORTED:
            raise ImportError("tifffile library required for TIFF support")

        super().__init__()

        self.file_path = file_path
        self.dim_order = dim_order.upper()
        self.frames_written = 0
        self.bigtiff = bigtiff
        self.imagej = imagej
        self.expected_frames = expected_frames
        self.ome = ome
        self.metadata = metadata or {}
        self.compression_level = compression_level

        # Compression mapping (mirror 2D writer)
        self._compression_map = {
            "none": None,
            "lzw": "lzw",
            "zlib": "zlib",
            "deflate": "zlib",
            "jpeg": "jpeg",
        }
        self.compression = (compression or "none").lower()

        # Track axes for current write (used for metadata like 2D writer)
        self._current_axes = None

        # Write parameters (set up on first write)
        self._file_kwargs = {}
        self._frame_kwargs = {}
        self._base_metadata = {"Software": "flowreg3d"}
        self._base_metadata.update(self.metadata)
        self._tif_writer: Optional[tifffile.TiffWriter] = None
        self._first_write_done = False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    def write_frames(self, frames: np.ndarray):
        """
        Write volumes to file.

        Args:
            frames: Array with shape (T, Z, Y, X, C) or (Z, Y, X, C) for single volume
        """
        if frames.ndim == 4:  # Single volume
            frames = frames[np.newaxis, ...]  # Add time dimension

        if frames.ndim != 5:
            raise ValueError(f"Expected 4D or 5D array, got {frames.ndim}D")

        # Track axes for metadata (matching 2D writer pattern)
        self._current_axes = "TZYX" if frames.shape[-1] == 1 else "TZCYX"

        # Initialize on first write
        if not self.initialized:
            self.init(frames)
            self._create_file()

        # Write frames to disk
        self._write_frames_to_file(frames)
        self.frames_written += frames.shape[0]

    def _create_file(self):
        """Initialize file for writing."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        self._setup_write_params()

        self._tif_writer = tifffile.TiffWriter(
            self.file_path,
            bigtiff=self.bigtiff,
            # Always manage ImageJ metadata manually to avoid tifffile validation
            imagej=False,
            ome=self.ome,
        )

    def _setup_write_params(self):
        """Set up parameters for writing (exactly like 2D version)."""
        compression = self._compression_map.get(self.compression, None)
        if compression == "zlib":
            compression = (compression, self.compression_level)

        self._file_kwargs = {
            "bigtiff": self.bigtiff,
            "imagej": False,
            "ome": self.ome,
        }

        self._frame_kwargs = {
            "compression": compression,
            "contiguous": True,
        }

    def _write_frames_to_file(self, frames: np.ndarray):
        """
        Write volumes to TIFF file (exactly like 2D version writes frames).

        Args:
            frames: Array with shape (T, Z, Y, X, C)
        """
        T, Z, Y, X, C = frames.shape

        # Only supply metadata on the first write to avoid append errors and
        # to let viewers recover axes. Even with imagej=False we emit an axes
        # tag for the first write.
        write_metadata = None
        description = None
        if not self._first_write_done:
            write_metadata = self._base_metadata.copy()
            axes_meta = "ZCYX" if C > 1 else "ZYX"

            if self.imagej:
                frames_meta = (
                    self.expected_frames if self.expected_frames is not None else 0
                )
                images_meta = frames_meta * self.depth * C if frames_meta else 0
                padding = (
                    " " * self._METADATA_PADDING if self.expected_frames is None else ""
                )
                description = (
                    "ImageJ=1.53c\n"
                    f"images={images_meta}\n"
                    f"channels={C}\n"
                    f"slices={self.depth}\n"
                    f"frames={frames_meta}\n"
                    "hyperstack=true\n"
                    "mode=composite\n"
                    "loop=false\n"
                    f"{padding}"
                )
            else:
                write_metadata.update({"axes": axes_meta})

        # Write each volume individually (like 2D writes each frame)
        if C == 1:
            # Single channel: (Z, Y, X, C) -> (Z, Y, X) like 2D does (T, H, W)
            for idx, volume in enumerate(frames):
                volume = volume[:, :, :, 0]  # Remove channel dimension
                md = write_metadata if idx == 0 else None
                self._tif_writer.write(
                    volume,
                    metadata=md,
                    description=description if idx == 0 else None,
                    **self._frame_kwargs,
                )
        else:
            # Multi-channel: (Z, Y, X, C) like 2D does (H, W, C)
            for idx, volume in enumerate(frames):
                md = write_metadata if idx == 0 else None
                # Write with planarconfig="contig" like 2D
                self._tif_writer.write(
                    volume.transpose(0, 3, 1, 2),
                    planarconfig="contig",
                    metadata=md,
                    description=description if idx == 0 else None,
                    **self._frame_kwargs,
                )

        self._first_write_done = True

    def close(self):
        """Close the TIFF file (following 2D pattern)."""
        # Close writer if open
        if self._tif_writer is not None:
            self._tif_writer.close()
            self._tif_writer = None

        # If we streamed without a known frame count, fix the header now
        if (
            self.imagej
            and self.expected_frames is None
            and self.frames_written > 0
            and self.initialized
        ):
            try:
                with tifffile.TiffFile(self.file_path, mode="r+") as tif:
                    if 270 in tif.pages[0].tags:
                        ij_str = (
                            "ImageJ=1.53c\n"
                            f"images={self.frames_written * self.depth * self.n_channels}\n"
                            f"channels={self.n_channels}\n"
                            f"slices={self.depth}\n"
                            f"frames={self.frames_written}\n"
                            "hyperstack=true\n"
                            "mode=composite\n"
                            "loop=false\n"
                        )
                        tif.pages[0].tags[270].overwrite(ij_str)
                        print(
                            f"  [Header Updated] Corrected frame count to {self.frames_written}"
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"  [Warning] Failed to update TIFF header: {exc}")

        # Print summary
        if self.frames_written > 0:
            print(f"3D TIFF file written: {self.file_path}")
            print(f"  Volumes: {self.frames_written}")
            print(
                f"  Shape per volume: (Z={self.depth}, Y={self.height}, X={self.width})"
            )
            print(f"  Channels: {self.n_channels}")
            print(f"  Compression: {self.compression or 'none'}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
