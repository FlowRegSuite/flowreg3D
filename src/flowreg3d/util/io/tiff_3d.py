"""
3D TIFF file reader and writer with support for volumetric time series.

Handles various dimension orderings (TXYZC, TZYXC, etc.) and provides
seamless integration with the 3D motion correction pipeline.
"""

import os
import warnings
from typing import Union, List, Optional, Tuple
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
    
    def __init__(self, file_path: str, buffer_size: int = 10, bin_size: int = 1,
                 dim_order: str = 'TZYXC', **kwargs):
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
        required_dims = set('TXYZ')
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
            
            # Parse ImageJ metadata if available
            if hasattr(self._tiff_file, 'imagej_metadata') and self._tiff_file.imagej_metadata:
                ij_meta = self._tiff_file.imagej_metadata
                self._metadata['imagej'] = ij_meta
                
                # Report detected structure
                if 'frames' in ij_meta and 'slices' in ij_meta:
                    print(f"ImageJ hyperstack detected: {ij_meta.get('frames')} frames, "
                          f"{ij_meta.get('slices')} slices, {ij_meta.get('channels', 1)} channels")
            
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
        if 'C' not in self.dim_order:
            if ndim == len(self.dim_order):
                # Dimensions match exactly, add implicit C=1
                self.dim_order = self.dim_order + 'C'
                self._data_array = self._data_array[..., np.newaxis]
                shape = self._data_array.shape
            elif ndim == len(self.dim_order) + 1:
                # Has channel dimension, update dim_order
                self.dim_order = self.dim_order + 'C'
            else:
                raise ValueError(f"Cannot parse dimensions. Array shape {shape} "
                                f"doesn't match dim_order '{self.dim_order}'")
        else:
            # C is in dim_order but might be missing from actual data
            if ndim == len(self.dim_order) - 1:
                # Array is missing channel dimension, add it
                c_axis = self.dim_order.index('C')
                self._data_array = np.expand_dims(self._data_array, axis=c_axis)
                shape = self._data_array.shape

        # Verify dimension count matches
        if len(shape) != len(self.dim_order):
            raise ValueError(f"Dimension mismatch: array has {len(shape)} dims, "
                           f"dim_order '{self.dim_order}' expects {len(self.dim_order)}")

        # Create dimension mapping
        dim_map = {dim: idx for idx, dim in enumerate(self.dim_order)}
        
        # Extract dimensions
        self.frame_count = shape[dim_map['T']]
        self.depth = shape[dim_map['Z']]
        self.height = shape[dim_map['Y']]
        self.width = shape[dim_map['X']]
        self.n_channels = shape[dim_map['C']] if 'C' in dim_map else 1
        
        # Transpose to standard TZYXC order if needed
        target_order = 'TZYXC'
        if self.dim_order != target_order:
            # Build transpose indices
            transpose_idx = []
            for dim in target_order:
                if dim in dim_map:
                    transpose_idx.append(dim_map[dim])
            
            self._data_array = np.transpose(self._data_array, transpose_idx)
            self.dim_order = target_order
        
        print(f"Parsed 3D TIFF: T={self.frame_count}, Z={self.depth}, "
              f"Y={self.height}, X={self.width}, C={self.n_channels}")
    
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
    
    def __init__(self, file_path: str, dim_order: str = 'TZYXC'):
        """
        Initialize 3D TIFF writer.
        
        Args:
            file_path: Output file path
            dim_order: Dimension ordering for output (default: 'TZYXC')
        """
        if not TIFF_SUPPORTED:
            raise ImportError("tifffile library required for TIFF support")
        
        super().__init__()
        
        self.file_path = file_path
        self.dim_order = dim_order.upper()
        self.frames_written = 0
        self._frames = []
        
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
        
        # Initialize on first write
        if not self.initialized:
            self.init(frames)
        
        self._frames.append(frames)
        self.frames_written += frames.shape[0]
    
    def close(self):
        """Write accumulated frames and close file."""
        if self._frames:
            # Concatenate all frames
            all_frames = np.concatenate(self._frames, axis=0)  # (T,Z,Y,X,C)
            T, Z, Y, X, C = all_frames.shape

            # Transpose to canonical ImageJ order (T,Z,C,Y,X)
            tzcyx = all_frames.transpose(0, 1, 4, 2, 3)  # (T,Z,Y,X,C) -> (T,Z,C,Y,X)

            # Write with ImageJ metadata
            # tifffile will handle the internal flattening to pages
            tifffile.imwrite(
                self.file_path,
                tzcyx,
                imagej=True,
                metadata={'axes': 'TZCYX', 'frames': T, 'slices': Z, 'channels': C}
            )

            print(f"Wrote 3D TIFF: {self.file_path} (T={T}, Z={Z}, Y={Y}, X={X}, C={C})")

        self._frames = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()