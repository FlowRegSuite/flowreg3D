"""
Multiprocessing executor for 3D volumes - processes volumes in parallel using shared memory.
"""

from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Tuple, Optional, Dict
import numpy as np
from .base_3d import BaseExecutor3D


# Global dictionary to store shared memory references in worker processes
_SHM: Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]] = {}


def _init_shared(shm_specs: Dict[str, Tuple[str, tuple, str]]):
    """
    Initialize shared memory in worker process.

    Args:
        shm_specs: Dictionary mapping names to (shm_name, shape, dtype_str) tuples
    """
    # Optional: Limit thread usage in numerical libraries to avoid oversubscription
    # Uncomment if you experience performance issues with nested parallelism
    # os.environ.update({
    #     'OMP_NUM_THREADS': '1',
    #     'MKL_NUM_THREADS': '1',
    #     'OPENBLAS_NUM_THREADS': '1',
    #     'NUMEXPR_NUM_THREADS': '1'
    # })

    global _SHM
    _SHM = {}
    for key, (name, shape, dtype_str) in shm_specs.items():
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        _SHM[key] = (shm, arr)


def _process_volume_worker(
    t: int, interpolation_method: str, flow_param_scalars: dict
) -> int:
    """
    Worker function to process a single 3D volume using shared memory.

    Args:
        t: Volume index
        interpolation_method: Interpolation method for registration
        flow_param_scalars: Dictionary of scalar flow parameters (non-array)

    Returns:
        Volume index (for tracking completion)
    """
    # Import functions inside worker to avoid pickling issues with Numba
    from flowreg3d.core.optical_flow_3d import get_displacement, imregister_wrapper

    # Get arrays from shared memory
    batch = _SHM["batch"][1]
    batch_proc = _SHM["batch_proc"][1]
    registered = _SHM["registered"][1]
    w_out = _SHM["flow_fields"][1]
    ref_proc = _SHM["ref_proc"][1]
    ref_raw = _SHM["ref_raw"][1]
    w_init = _SHM["w_init"][1]

    # Extract CC parameters and remove them from flow_params
    use_cc = bool(flow_param_scalars.get("cc_initialization", False))
    cc_hw = flow_param_scalars.get("cc_hw", 256)
    cc_up = int(flow_param_scalars.get("cc_up", 10))

    # Import prealignment function only if needed
    if use_cc:
        from flowreg3d.util.xcorr_prealignment import estimate_rigid_xcorr_3d

    # Create flow_params without CC parameters
    flow_params = {
        k: v
        for k, v in flow_param_scalars.items()
        if k not in ["cc_initialization", "cc_hw", "cc_up"]
    }
    if "weight" in _SHM:
        flow_params["weight"] = _SHM["weight"][1]

    # Check if cross-correlation initialization is enabled
    if use_cc:
        target_hw = cc_hw
        if isinstance(target_hw, int):
            target_hw = (target_hw, target_hw)
        weight = _SHM["weight"][1] if "weight" in _SHM else None

        # Step 1: Backward warp mov by w_init to get partially aligned
        mov_partial = imregister_wrapper(
            batch_proc[t],
            w_init[..., 0],  # dx
            w_init[..., 1],  # dy
            w_init[..., 2],  # dz
            ref_proc,
            interpolation_method="linear",
        )

        # Ensure shape consistency for xcorr (handle single channel case)
        ref_for_cc = ref_proc
        mov_for_cc = mov_partial
        if ref_proc.ndim == 4 and ref_proc.shape[3] == 1:
            ref_for_cc = ref_proc[..., 0]
        if mov_partial.ndim == 3 and ref_proc.ndim == 4:
            mov_for_cc = mov_partial
        elif mov_partial.ndim == 4 and mov_partial.shape[3] == 1:
            mov_for_cc = mov_partial[..., 0]

        # Step 2: Estimate rigid residual between ref and partially aligned mov
        w_cross = estimate_rigid_xcorr_3d(
            ref_for_cc, mov_for_cc, target_hw=target_hw, up=cc_up, weight=weight
        )

        # Step 3: Combine w_init + w_cross
        w_combined = w_init.copy()
        w_combined[..., 0] += w_cross[0]
        w_combined[..., 1] += w_cross[1]
        w_combined[..., 2] += w_cross[2]

        # Step 4: Backward warp original mov by combined field
        mov_aligned = imregister_wrapper(
            batch_proc[t],
            w_combined[..., 0],
            w_combined[..., 1],
            w_combined[..., 2],
            ref_proc,
            interpolation_method="linear",
        )

        # Ensure mov_aligned has channel dimension (imregister_wrapper strips it for single channel)
        if mov_aligned.ndim == 3:
            mov_aligned = mov_aligned[..., np.newaxis]

        # Step 5: Get residual non-rigid displacement
        w_residual = get_displacement(
            ref_proc, mov_aligned, uvw=np.zeros_like(w_init), **flow_params
        )

        # Step 6: Total flow is w_init + w_cross + w_residual
        flow = (w_combined + w_residual).astype(np.float32, copy=False)
    else:
        flow = get_displacement(
            ref_proc, batch_proc[t], uvw=w_init.copy(), **flow_params
        ).astype(np.float32, copy=False)

    # Apply 3D flow field to register the volume
    reg_volume = imregister_wrapper(
        batch[t],
        flow[..., 0],  # u (dx) displacement
        flow[..., 1],  # v (dy) displacement
        flow[..., 2],  # w (dz) displacement
        ref_raw,
        interpolation_method=interpolation_method,
    )

    # Store results directly in shared memory
    w_out[t] = flow

    # Handle case where registered volume might have fewer channels
    if reg_volume.ndim < registered.ndim - 1:
        registered[t, ..., 0] = reg_volume
    else:
        registered[t] = reg_volume

    return t


class MultiprocessingExecutor3D(BaseExecutor3D):
    """
    Multiprocessing executor using shared memory for zero-copy data sharing.

    This is the most efficient executor for CPU-bound operations as it:
    1. Uses multiple CPU cores in parallel
    2. Avoids data serialization overhead with shared memory
    3. Bypasses the GIL completely
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize multiprocessing executor.

        Args:
            n_workers: Number of worker processes. If None, uses RuntimeContext default.
        """
        super().__init__(n_workers)
        self.shm_handles = {}
        self.executor = None

    def setup(self):
        """Create the process pool executor."""
        if self.executor is None:
            # We'll create the executor with initializer when processing
            pass

    def cleanup(self):
        """Cleanup shared memory and shutdown executor."""
        # Cleanup shared memory
        for shm in self.shm_handles.values():
            shm.close()
            shm.unlink()
        self.shm_handles = {}

        # Shutdown executor
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _create_shared_input(self, name: str, arr: np.ndarray, shm_specs: dict):
        """
        Create shared memory for input array.

        Args:
            name: Name for the shared memory
            arr: Array to share
            shm_specs: Dictionary to store shared memory specifications
        """
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shared_arr[:] = arr
        shm_specs[name] = (shm.name, arr.shape, str(arr.dtype))
        self.shm_handles[name] = shm

    def _create_shared_output(
        self, name: str, shape: tuple, dtype: np.dtype, shm_specs: dict
    ) -> np.ndarray:
        """
        Create shared memory for output array.

        Args:
            name: Name for the shared memory
            shape: Shape of the array
            dtype: Data type of the array
            shm_specs: Dictionary to store shared memory specifications

        Returns:
            Numpy array view of the shared memory
        """
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shm_specs[name] = (shm.name, shape, str(np.dtype(dtype)))
        self.shm_handles[name] = shm
        return arr

    def process_batch(
        self,
        batch: np.ndarray,
        batch_proc: np.ndarray,
        reference_raw: np.ndarray,
        reference_proc: np.ndarray,
        w_init: np.ndarray,
        get_displacement_func: Callable,
        imregister_func: Callable,
        interpolation_method: str = "cubic",
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process 3D volumes in parallel using multiprocessing with shared memory.

        Args:
            batch: Raw volumes to register, shape (T, Z, Y, X, C)
            batch_proc: Preprocessed volumes for flow computation, shape (T, Z, Y, X, C)
            reference_raw: Raw reference volume, shape (Z, Y, X, C)
            reference_proc: Preprocessed reference volume, shape (Z, Y, X, C)
            w_init: Initial 3D flow field, shape (Z, Y, X, 3)
            get_displacement_func: Ignored (functions imported in worker)
            imregister_func: Ignored (functions imported in worker)
            interpolation_method: Interpolation method for registration
            **kwargs: Additional parameters including 'flow_params' dict

        Returns:
            Tuple of (registered_volumes, flow_fields)
        """
        T, Z, Y, X, C = batch.shape

        # Get flow parameters from kwargs
        flow_params = kwargs.get("flow_params", {})

        # Create shared memory for all arrays
        shm_specs = {}

        # Input arrays (read-only in workers)
        self._create_shared_input("batch", batch, shm_specs)
        self._create_shared_input("batch_proc", batch_proc, shm_specs)
        self._create_shared_input("ref_raw", reference_raw, shm_specs)
        self._create_shared_input("ref_proc", reference_proc, shm_specs)
        self._create_shared_input("w_init", w_init.astype(np.float32), shm_specs)

        # Handle weight array separately if present in flow_params
        if isinstance(flow_params.get("weight", None), np.ndarray):
            self._create_shared_input("weight", flow_params["weight"], shm_specs)
            # Create scalar-only params dict (without weight array)
            flow_param_scalars = {k: v for k, v in flow_params.items() if k != "weight"}
        else:
            flow_param_scalars = dict(flow_params)

        # Output arrays (written by workers)
        reg_arr = self._create_shared_output(
            "registered", batch.shape, batch.dtype, shm_specs
        )
        flow_arr = self._create_shared_output(
            "flow_fields", (T, Z, Y, X, 3), np.float32, shm_specs
        )

        # Create process pool with shared memory initialization
        with ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_init_shared, initargs=(shm_specs,)
        ) as executor:
            # Submit all volumes for processing
            futures = [
                executor.submit(
                    _process_volume_worker, t, interpolation_method, flow_param_scalars
                )
                for t in range(T)
            ]

            # Wait for all volumes to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions that occurred

        # Copy results from shared memory (important to copy before cleanup!)
        registered = np.array(reg_arr, copy=True)
        flow_fields = np.array(flow_arr, copy=True)

        # Call progress callback for entire batch (multiprocessing processes batch in parallel)
        if progress_callback is not None:
            progress_callback(T)

        # Cleanup shared memory
        for shm in self.shm_handles.values():
            shm.close()
            shm.unlink()
        self.shm_handles = {}

        return registered, flow_fields

    def get_info(self) -> dict:
        """Get information about this executor."""
        info = super().get_info()
        info.update(
            {
                "parallel": True,
                "description": f"3D multiprocessing with shared memory, {self.n_workers} workers",
                "features": ["zero-copy", "shared-memory", "true-parallelism"],
            }
        )
        return info


# Register this executor with RuntimeContext on import
MultiprocessingExecutor3D.register()
