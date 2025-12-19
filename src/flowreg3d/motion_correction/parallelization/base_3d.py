"""
Base executor abstract class for 3D parallelization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from flowreg3d._runtime import RuntimeContext


class BaseExecutor3D(ABC):
    """
    Abstract base class for 3D parallelization executors.

    All executors must implement the process_batch method which takes:
    - Batch of 3D volumes to process
    - Preprocessed batch
    - Reference volumes (raw and preprocessed)
    - Initial 3D flow field
    - Options and parameters

    And returns:
    - Registered volumes
    - Computed 3D flow fields
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize the executor.

        Args:
            n_workers: Number of workers to use. If None, uses RuntimeContext default.
        """
        self.n_workers = n_workers or RuntimeContext.get("max_workers", 1)
        self.name = self.__class__.__name__.replace("Executor", "").lower()

    @abstractmethod
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
        Process a batch of 3D volumes for motion correction.

        Args:
            batch: Raw volumes to register, shape (T, Z, Y, X, C)
            batch_proc: Preprocessed volumes for flow computation, shape (T, Z, Y, X, C)
            reference_raw: Raw reference volume, shape (Z, Y, X, C)
            reference_proc: Preprocessed reference volume, shape (Z, Y, X, C)
            w_init: Initial 3D flow field, shape (Z, Y, X, 3)
            get_displacement_func: Function to compute 3D optical flow
            imregister_func: Function to apply 3D flow field for registration
            interpolation_method: Interpolation method for registration
            progress_callback: Optional callback for per-volume progress (volumes_completed)
            **kwargs: Additional parameters

        Returns:
            Tuple of (registered_volumes, flow_fields) where:
                registered_volumes: shape (T, Z, Y, X, C)
                flow_fields: shape (T, Z, Y, X, 3)
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def setup(self):
        """
        Setup method called before processing.
        Override in subclasses if needed.
        """
        pass

    def cleanup(self):
        """
        Cleanup method called after processing.
        Override in subclasses if needed.
        """
        pass

    @classmethod
    def register(cls):
        """Register this executor with the RuntimeContext."""
        # Keep 3D suffix to avoid confusion with 2D version
        instance_name = cls.__name__.replace(
            "Executor", ""
        ).lower()  # e.g., SequentialExecutor3D -> sequential3d
        RuntimeContext.register_parallelization_executor(instance_name, cls)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this executor.

        Returns:
            Dictionary with executor information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "n_workers": self.n_workers,
        }
