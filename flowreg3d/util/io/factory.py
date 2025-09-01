"""
Factory functions for creating video readers and writers based on file format.
Simplified version for 3D with TIFF and array support.
"""

import os
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

import numpy as np

from flowreg3d.util.io._base_3d import VideoReader3D, VideoWriter3D


def get_video_file_reader(input_source: Union[str, Path, np.ndarray, VideoReader3D, List[str]], 
                          buffer_size: int = 10,  # Smaller default for 3D volumes
                          bin_size: int = 1, **kwargs) -> VideoReader3D:
    """
    Factory function to create appropriate 3D reader based on input type.
    
    Args:
        input_source: Path to video file, numpy array, VideoReader3D instance, 
                     list of paths for multichannel, or folder for images
        buffer_size: Buffer size for reading (smaller for 3D due to memory)
        bin_size: Temporal binning factor
        **kwargs: Additional reader-specific arguments
    
    Returns:
        Appropriate VideoReader3D subclass instance
    """
    from pathlib import Path

    # Handle numpy arrays
    if isinstance(input_source, np.ndarray):
        from flowreg3d.util.io._arr_3d import ArrayReader3D
        return ArrayReader3D(input_source, buffer_size, bin_size)
    
    # Handle VideoReader3D instances (already initialized)
    if isinstance(input_source, VideoReader3D):
        return input_source

    # Import readers here to avoid circular imports
    from flowreg3d.util.io.tiff_3d import TIFFFileReader3D
    
    # TODO: Add these when implemented:
    # from flowreg3d.util.io.hdf5_3d import HDF5FileReader3D
    # from flowreg3d.util.io.mat_3d import MATFileReader3D
    # from flowreg3d.util.io.multifile_wrappers_3d import MULTICHANNELFileReader3D

    # Handle multichannel input (list of files)
    if isinstance(input_source, list):
        # TODO: Implement MULTICHANNELFileReader3D
        raise NotImplementedError("Multichannel 3D reading not yet implemented")
    
    # From here on, treat as file path
    file_path = input_source
    path = Path(file_path)
    
    # Handle folder input (image sequence)
    if path.is_dir():
        # TODO: Implement image stack reading for 3D
        raise NotImplementedError("3D image folder reading not yet implemented. Use TIFF stacks instead.")
    
    # Handle file input
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()

    readers = {
        '.tif': TIFFFileReader3D,
        '.tiff': TIFFFileReader3D,
        # TODO: Add more formats as they're implemented
        # '.h5': HDF5FileReader3D,
        # '.hdf5': HDF5FileReader3D,
        # '.hdf': HDF5FileReader3D,
        # '.mat': MATFileReader3D,
    }

    reader_class = readers.get(ext)
    if reader_class:
        return reader_class(str(file_path), buffer_size, bin_size, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for 3D: {ext}. Currently only TIFF is supported.")


def get_video_file_writer(file_path: Optional[str], output_format: str, **kwargs) -> VideoWriter3D:
    """
    Factory function to create appropriate 3D writer based on output format.
    
    Args:
        file_path: Output file path (can be None for ARRAY format)
        output_format: Output format string (e.g., 'TIFF', 'HDF5', 'ARRAY')
        **kwargs: Additional writer-specific arguments
    
    Returns:
        Appropriate VideoWriter3D subclass instance
    """
    from pathlib import Path

    # Import writers here to avoid circular imports
    from flowreg3d.util.io.tiff_3d import TIFFFileWriter3D
    
    # TODO: Add these when implemented:
    # from flowreg3d.util.io.hdf5_3d import HDF5FileWriter3D
    # from flowreg3d.util.io.mat_3d import MATFileWriter3D
    # from flowreg3d.util.io.multifile_wrappers_3d import MULTIFILEFileWriter3D

    # Special handling for memory formats
    if output_format == 'ARRAY':
        from flowreg3d.util.io._arr_3d import ArrayWriter3D
        return ArrayWriter3D()

    # Ensure we have a file path for file-based formats
    if file_path is None:
        raise ValueError(f"file_path required for output format: {output_format}")

    # Handle different output formats
    if output_format == 'TIFF':
        return TIFFFileWriter3D(file_path, **kwargs)
    elif output_format == 'HDF5':
        # TODO: Implement HDF5FileWriter3D
        raise NotImplementedError("HDF5 3D writing not yet implemented")
    elif output_format == 'MAT':
        # TODO: Implement MATFileWriter3D
        raise NotImplementedError("MAT 3D writing not yet implemented")
    elif output_format.startswith('MULTIFILE'):
        # TODO: Implement multifile writers for 3D
        raise NotImplementedError("Multifile 3D writing not yet implemented")
    else:
        raise ValueError(f"Unsupported 3D output format: {output_format}")


def main():
    """Test factory functions with 3D data."""
    import tempfile
    
    # Create test 3D data (T, Z, Y, X, C)
    test_volumes = np.random.randint(0, 255, (5, 20, 64, 64, 2), dtype=np.uint8)
    
    print("Testing 3D factory functions...")
    
    # Test array reader
    print("\nTesting ArrayReader3D creation...")
    reader = get_video_file_reader(test_volumes)
    assert reader.shape == (5, 20, 64, 64, 2), f"Shape mismatch: {reader.shape}"
    print(f"✓ ArrayReader3D created with shape: {reader.shape}")
    
    # Test array writer
    print("\nTesting ArrayWriter3D creation...")
    writer = get_video_file_writer(None, 'ARRAY')
    writer.write_frames(test_volumes)
    result = writer.get_array()
    assert result.shape == test_volumes.shape, f"Written shape mismatch: {result.shape}"
    print(f"✓ ArrayWriter3D created and tested")
    
    # Test TIFF writer
    print("\nTesting TIFFFileWriter3D creation...")
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        writer = get_video_file_writer(tmp.name, 'TIFF')
        writer.write_frames(test_volumes)
        writer.close()
        
        # Read back
        reader = get_video_file_reader(tmp.name)
        read_data = reader[:]
        reader.close()
        
        assert read_data.shape == test_volumes.shape, f"TIFF round-trip shape mismatch: {read_data.shape}"
        print(f"✓ TIFF 3D round-trip successful with shape: {read_data.shape}")
        
        # Clean up
        os.unlink(tmp.name)
    
    print("\n✓ All 3D factory tests passed!")


if __name__ == "__main__":
    main()