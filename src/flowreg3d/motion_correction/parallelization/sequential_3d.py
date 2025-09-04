"""
Sequential executor for 3D volumes - processes volumes one by one without parallelization.
"""

from typing import Callable, Tuple
import numpy as np
from .base_3d import BaseExecutor3D


class SequentialExecutor3D(BaseExecutor3D):
    """
    Sequential executor that processes 3D volumes one at a time.
    
    This is the simplest executor and serves as a reference implementation.
    It's also the most memory-efficient as it only processes one volume at a time.
    """
    
    def __init__(self, n_workers: int = 1):
        """
        Initialize sequential executor.
        
        Args:
            n_workers: Ignored for sequential executor, always uses 1.
        """
        super().__init__(n_workers=1)
    
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
        progress_callback: Callable = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process 3D volumes sequentially.
        
        Args:
            batch: Raw volumes to register, shape (T, Z, Y, X, C)
            batch_proc: Preprocessed volumes for flow computation, shape (T, Z, Y, X, C)
            reference_raw: Raw reference volume, shape (Z, Y, X, C)
            reference_proc: Preprocessed reference volume, shape (Z, Y, X, C)
            w_init: Initial flow field, shape (Z, Y, X, 3) with [u, v, w] components
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
        
        # Process each volume sequentially
        for t in range(T):
            # Compute 3D optical flow for this volume with all parameters
            flow = get_displacement_func(
                reference_proc, 
                batch_proc[t], 
                uvw=w_init.copy(),
                **flow_params
            )
            
            # Apply 3D flow field to register the volume
            reg_volume = imregister_func(
                batch[t],
                flow[..., 0],  # u (x) displacement
                flow[..., 1],  # v (y) displacement
                flow[..., 2],  # w (z) displacement
                reference_raw,
                interpolation_method=interpolation_method
            )
            
            # Store results
            flow_fields[t] = flow.astype(np.float32, copy=False)
            
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
            'parallel': False,
            'description': 'Sequential 3D volume-by-volume processing'
        })
        return info


# Register this executor with RuntimeContext on import
SequentialExecutor3D.register()