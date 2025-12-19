"""
Array-based readers and writers for 3D volumetric data.
Provides in-memory I/O for motion correction pipeline.
"""

from typing import Union, List
import numpy as np

from flowreg3d.util.io._base_3d import VideoReader3D, VideoWriter3D


class ArrayReader3D(VideoReader3D):
    """
    Reader for numpy arrays treated as 3D video sources.
    Useful for processing in-memory data without file I/O.
    """

    def __init__(self, array: np.ndarray, buffer_size: int = 10, bin_size: int = 1):
        """
        Initialize array reader for 3D volumetric data.

        Args:
            array: Input array with shape (T, Z, Y, X, C) or (Z, Y, X, C) for single volume
            buffer_size: Number of volumes to read at once
            bin_size: Temporal binning factor (not used for arrays)
        """
        super().__init__()

        # Ensure 5D array (T, Z, Y, X, C)
        if array.ndim == 4:  # Single volume (Z, Y, X, C)
            array = array[np.newaxis, ...]  # Add time dimension
        elif array.ndim == 3:  # Single volume, single channel (Z, Y, X)
            array = array[np.newaxis, ..., np.newaxis]  # Add time and channel
        elif array.ndim == 5:  # Already (T, Z, Y, X, C)
            pass
        else:
            raise ValueError(f"Array must be 3D, 4D or 5D, got shape {array.shape}")

        self.array = array
        self.buffer_size = buffer_size
        self.bin_size = bin_size

        # Set dimensions
        self.frame_count = array.shape[0]
        self.depth = array.shape[1]
        self.height = array.shape[2]
        self.width = array.shape[3]
        self.n_channels = array.shape[4]
        self.dtype = array.dtype

        self._initialized = True
        self._current_idx = 0

    def _initialize(self):
        """Already initialized in __init__."""
        pass

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read specified volumes from array.

        Args:
            frame_indices: Slice or list of volume indices

        Returns:
            Array with shape (T, Z, Y, X, C)
        """
        return self.array[frame_indices].copy()

    def close(self):
        """No cleanup needed for array reader."""
        pass


class ArrayWriter3D(VideoWriter3D):
    """
    Writer that accumulates 3D volumes in memory.
    Useful for getting results without file I/O.
    """

    def __init__(self):
        """Initialize array writer for 3D volumetric data."""
        super().__init__()
        self._vid = []

    def write_frames(self, frames: np.ndarray):
        """
        Add volumes to internal list.

        Args:
            frames: Array with shape (T, Z, Y, X, C) or (Z, Y, X, C) for single volume
        """
        if frames.ndim == 3:
            return frames[np.newaxis, ..., np.newaxis]
        if frames.ndim == 4:  # Single volume
            frames = frames[np.newaxis, ...]  # Add time dimension
        if frames.ndim != 5:
            raise ValueError(f"Expected 3D, 4D or 5D array, got {frames.ndim}D")

        # Initialize on first write
        if not self.initialized:
            self.init(frames)

        self._vid.append(frames)

    def get_array(self) -> np.ndarray:
        """
        Get concatenated array of all written volumes.

        Returns:
            Array with shape (T, Z, Y, X, C) containing all written volumes
        """
        if not self._vid:
            return None

        return np.concatenate(self._vid, axis=0)

    def close(self):
        """Finalize array by concatenating all frames."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Test array reader and writer for 3D data."""
    print("Testing ArrayReader3D and ArrayWriter3D...")

    # Create test 3D data
    test_data = np.random.randint(0, 255, (10, 20, 64, 64, 2), dtype=np.uint8)
    print(f"Test data shape: {test_data.shape}")

    # Test reader
    print("\nTesting ArrayReader3D...")
    reader = ArrayReader3D(test_data, buffer_size=3)

    print(f"  Frame count: {reader.frame_count}")
    print(f"  Depth: {reader.depth}")
    print(f"  Dimensions: {reader.height}x{reader.width}")
    print(f"  Channels: {reader.n_channels}")

    # Read all volumes
    all_volumes = reader[:]
    assert all_volumes.shape == test_data.shape, "Shape mismatch after reading"
    print("  ✓ Read all volumes successfully")

    # Read subset
    subset = reader[2:5]
    assert subset.shape == (3, 20, 64, 64, 2), "Subset shape mismatch"
    print("  ✓ Read subset successfully")

    # Test writer
    print("\nTesting ArrayWriter3D...")
    writer = ArrayWriter3D()

    # Write in batches
    writer.write_frames(test_data[:5])
    writer.write_frames(test_data[5:8])
    writer.write_frames(test_data[8:])

    # Get result
    result = writer.get_array()
    assert result.shape == test_data.shape, "Shape mismatch after writing"
    np.testing.assert_array_equal(
        result, test_data, err_msg="Data mismatch after write"
    )
    print("  ✓ Write and retrieve successful")

    # Test single volume
    print("\nTesting single volume handling...")
    single_volume = test_data[0]  # Shape (20, 64, 64, 2)

    reader_single = ArrayReader3D(single_volume)
    assert reader_single.frame_count == 1, "Single volume should have frame_count=1"

    writer_single = ArrayWriter3D()
    writer_single.write_frames(single_volume)
    result_single = writer_single.get_array()
    assert result_single.shape == (1, 20, 64, 64, 2), "Single volume shape mismatch"
    print("  ✓ Single volume handling successful")

    print("\n✓ All ArrayReader3D/ArrayWriter3D tests passed!")


if __name__ == "__main__":
    main()
