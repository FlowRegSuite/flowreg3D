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
        flow_params_all = kwargs.get('flow_params', {})
        
        # Extract CC parameters and remove them from flow_params
        use_cc = bool(flow_params_all.get('cc_initialization', False))
        cc_hw = flow_params_all.get('cc_hw', 256)
        cc_up = int(flow_params_all.get('cc_up', 10))
        
        # Create flow_params without CC parameters
        flow_params = {k: v for k, v in flow_params_all.items() 
                      if k not in ['cc_initialization', 'cc_hw', 'cc_up']}
        
        # Initialize output arrays (use empty instead of zeros for performance)
        registered = np.empty_like(batch)
        flow_fields = np.empty((T, Z, Y, X, 3), dtype=np.float32)
        
        # Import prealignment functions if needed
        from flowreg3d.util.xcorr_prealignment import estimate_rigid_xcorr_3d
        
        # Setup CC parameters
        target_hw = cc_hw
        if isinstance(target_hw, int):
            target_hw = (target_hw, target_hw)
        up = cc_up
        weight = flow_params.get('weight', None)
        
        # Process each volume sequentially
        for t in range(T):
            if use_cc:
                # Step 1: Backward warp mov by w_init to get partially aligned
                mov_partial = imregister_func(
                    batch_proc[t],
                    w_init[..., 0],  # dx
                    w_init[..., 1],  # dy
                    w_init[..., 2],  # dz
                    reference_proc,
                    interpolation_method='linear'
                )
                
                # Ensure shape consistency for xcorr (handle single channel case)
                ref_for_cc = reference_proc
                mov_for_cc = mov_partial
                if reference_proc.ndim == 4 and reference_proc.shape[3] == 1:
                    ref_for_cc = reference_proc[..., 0]
                if mov_partial.ndim == 3 and reference_proc.ndim == 4:
                    mov_for_cc = mov_partial
                elif mov_partial.ndim == 4 and mov_partial.shape[3] == 1:
                    mov_for_cc = mov_partial[..., 0]
                
                # Step 2: Estimate rigid residual between ref and partially aligned mov
                w_cross = estimate_rigid_xcorr_3d(ref_for_cc, mov_for_cc, target_hw=target_hw, up=up, weight=weight)
                
                # Step 3: Combine w_init + w_cross
                w_combined = w_init.copy()
                w_combined[..., 0] += w_cross[0]
                w_combined[..., 1] += w_cross[1]
                w_combined[..., 2] += w_cross[2]
                
                # Step 4: Backward warp original mov by combined field
                mov_aligned = imregister_func(
                    batch_proc[t],
                    w_combined[..., 0],
                    w_combined[..., 1],
                    w_combined[..., 2],
                    reference_proc,
                    interpolation_method='linear'
                )
                
                # Ensure mov_aligned has channel dimension (imregister_wrapper strips it for single channel)
                if mov_aligned.ndim == 3:
                    mov_aligned = mov_aligned[..., np.newaxis]
                
                # Step 5: Get residual non-rigid displacement
                w_residual = get_displacement_func(reference_proc, mov_aligned, uvw=np.zeros_like(w_init), **flow_params)
                
                # Step 6: Total flow is w_init + w_cross + w_residual
                flow = (w_combined + w_residual).astype(np.float32, copy=False)
            else:
                # Compute 3D optical flow without prealignment
                flow = get_displacement_func(
                    reference_proc, 
                    batch_proc[t], 
                    uvw=w_init.copy(),
                    **flow_params
                ).astype(np.float32, copy=False)
            
            # Apply 3D flow field to register the volume
            reg_volume = imregister_func(
                batch[t],
                flow[..., 0],  # u (dx) displacement
                flow[..., 1],  # v (dy) displacement
                flow[..., 2],  # w (dz) displacement
                reference_raw,
                interpolation_method=interpolation_method
            )
            
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
            'parallel': False,
            'description': 'Sequential 3D volume-by-volume processing'
        })
        return info


# Register this executor with RuntimeContext on import
SequentialExecutor3D.register()