"""
Threading executor for 3D volumes - processes volumes in parallel using threads.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Tuple, Optional
import numpy as np
from .base_3d import BaseExecutor3D


class ThreadingExecutor3D(BaseExecutor3D):
    """
    Threading executor that processes 3D volumes in parallel using threads.
    
    Good for I/O-bound operations or when the GIL is released (e.g., NumPy operations).
    Less efficient than multiprocessing for pure Python CPU-bound operations.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize threading executor.
        
        Args:
            n_workers: Number of worker threads. If None, uses RuntimeContext default.
        """
        super().__init__(n_workers)
        self.executor = None
    
    def setup(self):
        """Create the thread pool executor."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
    
    def cleanup(self):
        """Shutdown the thread pool executor."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
    
    def _process_volume(
        self,
        t: int,
        volume: np.ndarray,
        volume_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str,
        flow_params: dict
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Process a single 3D volume.
        
        Args:
            t: Volume index
            volume: Raw volume to register
            volume_proc: Preprocessed volume
            reference_raw: Raw reference volume
            reference_proc: Preprocessed reference volume
            w_init: Initial 3D flow field
            get_displacement_func: Function to compute 3D optical flow
            imregister_func: Function to apply 3D flow field
            interpolation_method: Interpolation method
            flow_params: Dictionary of flow computation parameters
            
        Returns:
            Tuple of (volume_index, registered_volume, flow_field)
        """
        # Compute 3D optical flow with all parameters
        flow = get_displacement_func(
            reference_proc,
            volume_proc,
            uvw=w_init.copy(),
            **flow_params
        )
        
        # Apply 3D flow field to register the volume
        reg_volume = imregister_func(
            volume,
            flow[..., 0],  # u (x) displacement
            flow[..., 1],  # v (y) displacement
            flow[..., 2],  # w (z) displacement
            reference_raw,
            interpolation_method=interpolation_method
        )
        
        return t, reg_volume, flow.astype(np.float32, copy=False)
    
    def process_batch(
        self,
        batch: np.ndarray,
        batch_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str = 'cubic',
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process 3D volumes in parallel using threads.
        
        Args:
            batch: Raw volumes to register, shape (T, Z, Y, X, C)
            batch_proc: Preprocessed volumes for flow computation, shape (T, Z, Y, X, C)
            reference_raw: Raw reference volume, shape (Z, Y, X, C)
            reference_proc: Preprocessed reference volume, shape (Z, Y, X, C)
            w_init: Initial 3D flow field, shape (Z, Y, X, 3)
            get_displacement_func: Function to compute 3D optical flow
            imregister_func: Function to apply 3D flow field for registration
            interpolation_method: Interpolation method for registration
            **kwargs: Additional parameters including 'flow_params' dict
            
        Returns:
            Tuple of (registered_volumes, flow_fields)
        """
        T, Z, Y, X, C = batch.shape
        
        # Get flow parameters from kwargs
        flow_params = kwargs.get('flow_params', {})
        
        # Initialize output arrays (use empty instead of zeros for performance)
        registered = np.empty_like(batch)
        flow_fields = np.empty((T, Z, Y, X, 3), dtype=np.float32)
        
        # Ensure executor is created
        if self.executor is None:
            self.setup()
        
        # Submit all volumes for processing
        futures = []
        for t in range(T):
            future = self.executor.submit(
                self._process_volume,
                t,
                batch[t],
                batch_proc[t],
                reference_raw,
                reference_proc,
                w_init,
                get_displacement_func,
                imregister_func,
                interpolation_method,
                flow_params
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            t, reg_volume, flow = future.result()
            
            # Store results
            flow_fields[t] = flow
            
            # Handle case where registered volume might have fewer channels
            if reg_volume.ndim < registered.ndim - 1:
                registered[t, ..., 0] = reg_volume
            else:
                registered[t] = reg_volume
            
            # Call progress callback for this volume
            if progress_callback is not None:
                progress_callback(1)
        
        return registered, flow_fields
    
    def get_info(self) -> dict:
        """Get information about this executor."""
        info = super().get_info()
        info.update({
            'parallel': True,
            'description': f'Threaded 3D parallel processing with {self.n_workers} workers'
        })
        return info


# Register this executor with RuntimeContext on import
ThreadingExecutor3D.register()