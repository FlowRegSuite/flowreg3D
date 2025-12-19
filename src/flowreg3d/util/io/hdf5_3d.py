import os
from typing import Union, List

import h5py
import numpy as np

from flowreg3d.util.io._base_3d import VideoReader3D, VideoWriter3D
from flowreg3d.util.io._ds_io_3d import DSFileReader3D, DSFileWriter3D


class HDF5FileReader3D(DSFileReader3D, VideoReader3D):
    """HDF5 3D volumetric file reader with dataset discovery."""

    def __init__(
        self, file_path: str, buffer_size: int = 500, bin_size: int = 1, **kwargs
    ):
        # Initialize parent classes
        DSFileReader3D.__init__(self)
        VideoReader3D.__init__(self)

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.h5file = None

        # Dataset-specific options
        self.dataset_names = kwargs.get("dataset_names")
        self.dimension_ordering = kwargs.get("dimension_ordering")

    def _initialize(self):
        """Open file and set up properties."""
        try:
            self.h5file = h5py.File(self.file_path, "r")
        except Exception as e:
            raise IOError(f"Cannot open HDF5 file: {e}")

        # Use DSFileReader mixin to find datasets
        if not self.dataset_names:
            datasets_info = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets_info.append((name, obj.shape))

            self.h5file.visititems(visitor)
            self.dataset_names = self._find_datasets(datasets_info)

        if not self.dataset_names:
            raise ValueError("No suitable datasets found")

        # Set properties from first dataset
        first_ds = self.h5file[self.dataset_names[0]]
        shape = first_ds.shape

        # Detect dimension ordering for 3D data
        # Assume (T, Z, Y, X) for 4D or (T, Z, Y, X, C) for 5D
        if len(shape) == 4:
            self.frame_count, self.depth, self.height, self.width = shape
            self.n_channels = len(self.dataset_names)
        elif len(shape) == 5:
            self.frame_count, self.depth, self.height, self.width, self.n_channels = (
                shape
            )

        self.dtype = first_ds.dtype

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """Read raw frames from HDF5 file."""
        # Convert list to slice if contiguous
        if isinstance(frame_indices, list):
            if len(frame_indices) == 0:
                return np.empty(
                    (0, self.depth, self.height, self.width, self.n_channels),
                    dtype=self.dtype,
                )

            # Check if contiguous
            if len(frame_indices) > 1:
                diffs = np.diff(frame_indices)
                if np.all(diffs == 1):
                    frame_indices = slice(frame_indices[0], frame_indices[-1] + 1)

        # Allocate output
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            indices = range(start, stop, step)
        else:
            indices = frame_indices

        n_frames = len(indices)
        output = np.zeros(
            (n_frames, self.depth, self.height, self.width, self.n_channels),
            dtype=self.dtype,
        )

        # Read from each dataset/channel
        for ch_idx, ds_name in enumerate(self.dataset_names):
            dataset = self.h5file[ds_name]

            if isinstance(frame_indices, slice):
                # Efficient slicing for contiguous frames
                data = dataset[frame_indices, :, :, :]
            else:
                # Fancy indexing for non-contiguous
                data = dataset[indices, :, :, :]

            output[:, :, :, :, ch_idx] = data

        return output

    def close(self):
        """Close HDF5 file."""
        if self.h5file:
            self.h5file.close()
            self.h5file = None


class HDF5FileWriter3D(DSFileWriter3D, VideoWriter3D):
    """
    HDF5 3D volumetric file writer with MATLAB compatibility.

    Accepts volumes in Python format (T, Z, Y, X, C) but stores them
    in MATLAB-compatible format as separate 4D datasets per channel
    with configurable dimension ordering.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initialize HDF5 writer.

        Args:
            file_path: Output file path
            dataset_names: Optional dataset naming pattern or list
                          Default: 'ch*' (produces ch1, ch2, etc.)
            dimension_ordering: Storage order for MATLAB compatibility
                               Default: (0, 1, 2) for (H, W, T) storage
            compression: HDF5 compression ('gzip', 'lzf', or None)
            compression_level: Compression level for gzip (1-9)
            chunk_size: Chunk size for temporal dimension (default: 1)
        """
        # Initialize parent classes
        DSFileWriter3D.__init__(self, **kwargs)
        VideoWriter3D.__init__(self)

        self.file_path = file_path
        self._h5file = None
        self._datasets = {}
        self._frame_counter = 0

        # MATLAB compatibility options
        # Default (1, 2, 3, 0) means store as (T, Z, Y, X) which MATLAB reads as (Z, Y, X, T)
        self.dimension_ordering = kwargs.get("dimension_ordering", (1, 2, 3, 0))

        # Compression options
        self.compression = kwargs.get("compression", None)
        self.compression_level = kwargs.get("compression_level", 4)
        self.chunk_temporal = kwargs.get("chunk_size", 1)

        # Dataset naming - default to MATLAB convention
        if not self.dataset_names:
            self.dataset_names = "ch*"  # Will produce ch1, ch2, etc.

    def _create_datasets(self):
        """Create HDF5 datasets for each channel."""
        # Remove existing file if it exists
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        self._h5file = h5py.File(self.file_path, "w")

        # Define initial shape and max shape for expandable datasets
        # We store in MATLAB format: separate 4D datasets per channel
        initial_shape = [None, None, None, None]
        initial_shape[self.dimension_ordering[0]] = self.depth
        initial_shape[self.dimension_ordering[1]] = self.height
        initial_shape[self.dimension_ordering[2]] = self.width
        initial_shape[self.dimension_ordering[3]] = 0  # Start with 0 frames
        initial_shape = tuple(initial_shape)

        max_shape = [None, None, None, None]
        max_shape[self.dimension_ordering[0]] = self.depth
        max_shape[self.dimension_ordering[1]] = self.height
        max_shape[self.dimension_ordering[2]] = self.width
        max_shape[self.dimension_ordering[3]] = None  # Unlimited frames
        max_shape = tuple(max_shape)

        # Chunking for efficient I/O
        chunk_shape = [None, None, None, None]
        chunk_shape[self.dimension_ordering[0]] = self.depth
        chunk_shape[self.dimension_ordering[1]] = self.height
        chunk_shape[self.dimension_ordering[2]] = self.width
        chunk_shape[self.dimension_ordering[3]] = self.chunk_temporal
        chunk_shape = tuple(chunk_shape)

        # Create a dataset for each channel
        for ch_idx in range(self.n_channels):
            ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)

            # Create expandable dataset
            if self.compression:
                if self.compression == "gzip":
                    ds = self._h5file.create_dataset(
                        name=ds_name,
                        shape=initial_shape,
                        maxshape=max_shape,
                        dtype=self.dtype,
                        chunks=chunk_shape,
                        compression="gzip",
                        compression_opts=self.compression_level,
                    )
                else:
                    ds = self._h5file.create_dataset(
                        name=ds_name,
                        shape=initial_shape,
                        maxshape=max_shape,
                        dtype=self.dtype,
                        chunks=chunk_shape,
                        compression=self.compression,
                    )
            else:
                ds = self._h5file.create_dataset(
                    name=ds_name,
                    shape=initial_shape,
                    maxshape=max_shape,
                    dtype=self.dtype,
                    chunks=chunk_shape,
                )

            self._datasets[ds_name] = ds

            # Add MATLAB-friendly attributes
            ds.attrs["dimension_ordering"] = self.dimension_ordering
            ds.attrs["original_shape_TZYXC"] = (
                0,
                self.depth,
                self.height,
                self.width,
                self.n_channels,
            )

    def write_frames(self, frames: np.ndarray):
        """
        Write volumes to HDF5 file.

        Args:
            frames: Array with shape (T, Z, Y, X, C) or (T, Z, Y, X) or (Z, Y, X)
        """
        # Normalize input to 5D (T, Z, Y, X, C)
        if frames.ndim == 3:  # Single volume, single channel (Z, Y, X)
            frames = frames[np.newaxis, :, :, :, np.newaxis]
        elif frames.ndim == 4:
            if (
                frames.shape[0] == self.depth
                and frames.shape[1] == self.height
                and frames.shape[2] == self.width
            ):
                # Single volume, multiple channels (Z, Y, X, C)
                frames = frames[np.newaxis, :, :, :, :]
            else:
                # Multiple volumes, single channel (T, Z, Y, X)
                frames = frames[:, :, :, :, np.newaxis]
        elif frames.ndim != 5:
            raise ValueError(f"Expected 3D, 4D or 5D input, got {frames.ndim}D")

        # Initialize on first write
        if not self.initialized:
            T, Z, Y, X, C = frames.shape
            self.depth = Z
            self.height = Y
            self.width = X
            self.n_channels = C
            self.dtype = frames.dtype
            self.initialized = True
            self._create_datasets()

        # Validate shape
        T, Z, Y, X, C = frames.shape
        if Z != self.depth or Y != self.height or X != self.width:
            raise ValueError(
                f"Volume size mismatch. Expected ({self.depth}, {self.height}, {self.width}), "
                f"got ({Z}, {Y}, {X})"
            )
        if C != self.n_channels:
            raise ValueError(
                f"Channel count mismatch. Expected {self.n_channels}, got {C}"
            )

        # Write each channel separately for MATLAB compatibility
        for ch_idx in range(self.n_channels):
            ds_name = self.get_ds_name(ch_idx + 1, self.n_channels)
            dataset = self._datasets[ds_name]

            # Extract channel data: (T, Z, Y, X)
            channel_data = frames[:, :, :, :, ch_idx]

            # Transpose to MATLAB storage order
            # From (T, Z, Y, X) to dimension_ordering
            if self.dimension_ordering != (3, 0, 1, 2):
                # Create mapping from current (T=0, Z=1, Y=2, X=3) to target ordering
                # Default MATLAB is (Z=0, Y=1, X=2, T=3)
                perm = [None, None, None, None]
                perm[self.dimension_ordering[0]] = 1  # Z position
                perm[self.dimension_ordering[1]] = 2  # Y position
                perm[self.dimension_ordering[2]] = 3  # X position
                perm[self.dimension_ordering[3]] = 0  # T position
                channel_data = np.transpose(channel_data, perm)

            # Determine where to write in the dataset
            current_frames = dataset.shape[self.dimension_ordering[3]]
            new_total_frames = current_frames + T

            # Resize dataset along time dimension
            new_shape = list(dataset.shape)
            new_shape[self.dimension_ordering[3]] = new_total_frames
            dataset.resize(new_shape)

            # Create slice objects for writing
            slices = [slice(None), slice(None), slice(None), slice(None)]
            slices[self.dimension_ordering[3]] = slice(current_frames, new_total_frames)

            # Write the data
            dataset[tuple(slices)] = channel_data

            # Update attributes
            dataset.attrs["original_shape_TZYXC"] = (
                new_total_frames,
                Z,
                Y,
                X,
                self.n_channels,
            )

        self._frame_counter = new_total_frames

        # Flush to ensure data is written
        if self._h5file:
            self._h5file.flush()

    def close(self):
        """Close the HDF5 file."""
        if self._h5file:
            # Write final metadata for MATLAB compatibility
            if self._datasets:
                # Add file-level attributes
                self._h5file.attrs["n_channels"] = self.n_channels
                self._h5file.attrs["frame_count"] = self._frame_counter
                self._h5file.attrs["depth"] = self.depth
                self._h5file.attrs["height"] = self.height
                self._h5file.attrs["width"] = self.width
                self._h5file.attrs["dimension_ordering"] = self.dimension_ordering
                self._h5file.attrs["format"] = "flowreg3d_hdf5_v1"

                # Store dataset names as attribute for easy discovery
                dataset_names_list = list(self._datasets.keys())
                self._h5file.attrs["dataset_names"] = dataset_names_list

            self._h5file.close()
            self._h5file = None
            self._datasets = {}
            print(f"HDF5 file written: {self.file_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    import numpy as np
    from pathlib import Path
    from mdf import MDFFileReader
    import cv2
    from flowreg3d.util.io.hdf5_3d import HDF5FileWriter3D, HDF5FileReader3D

    filename = r"D:\2025_OIST\Shinobu\RFPonly\190403_001.MDF"
    out_path = Path(filename + ".hdf")

    mdf = MDFFileReader(filename, buffer_size=500, bin_size=1)

    with HDF5FileWriter3D(str(out_path)) as w:
        # for i in range(5 * 8200, 5 * 9200):
        for i in range(5 * 8200, 5 * 8300):
            frame = mdf[i]
            w.write_frames(frame[np.newaxis])

    h5 = HDF5FileReader3D(str(out_path), buffer_size=500, bin_size=5)
    h5_b5 = h5[0:20]
    h5.close()
    mdf.close()

    mdf2 = MDFFileReader(filename, buffer_size=500, bin_size=5)
    mdf_b5 = mdf2[8200 : 8200 + 20]
    mdf2.close()

    counter = 0
    while True:
        frame = np.concatenate([h5_b5[counter], mdf_b5[counter]], axis=0)
        counter = (counter + 1) % h5_b5.shape[0]
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frame[..., 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        key = cv2.waitKey(1)
        if key == 27:
            break

    if not np.array_equal(h5_b5, mdf_b5):
        d = h5_b5.astype(np.int64) - mdf_b5.astype(np.int64)
        print(int(np.abs(d).max()))
        print("Frames are not equal!")
    else:
        print(f"OK {out_path}")


def reader_main():
    import cv2
    from flowreg3d.util.io.hdf5_3d import HDF5FileReader3D

    filename = r"D:\2025_OIST\Shinobu\RFPonly\test.hdf"
    reader = HDF5FileReader3D(filename, buffer_size=500, bin_size=1)
    print(f"Number of frames: {len(reader)}")
    for i in range(len(reader)):
        frame = reader[i]
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frame[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
