"""
Random seed management for reproducibility across multiple frameworks.
Implements 2025 best practices for fixing random seeds in numpy, torch, tensorflow, and jax.
"""

import os
import random
import warnings
from typing import Optional

# Framework availability flags
HAS_NUMPY = False
HAS_TORCH = False
HAS_TENSORFLOW = False
HAS_JAX = False

# Try importing each framework and set flags
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    pass

try:
    import jax
    HAS_JAX = True
except ImportError:
    pass


def fix_seed(seed: int = 1, deterministic: bool = False, verbose: bool = False) -> None:
    """
    Fix random seed across all available frameworks for reproducibility.
    
    Args:
        seed: The random seed to use (default: 1)
        deterministic: Whether to enforce fully deterministic behavior (may impact performance)
        verbose: Whether to print which frameworks were seeded
    """
    # Set environment-level seed (affects Python's hash functions)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Always set Python's built-in random seed
    random.seed(seed)
    
    if verbose:
        print(f"Setting random seed to {seed} for available frameworks:")
        print(f"  - Python random: ✓")
    
    # NumPy
    if HAS_NUMPY:
        np.random.seed(seed)
        # Also set the new numpy random generator seed (numpy >= 1.17)
        np.random.default_rng(seed)
        if verbose:
            print(f"  - NumPy: ✓")
    
    # PyTorch
    if HAS_TORCH:
        torch.manual_seed(seed)
        
        # Handle CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            
            if deterministic:
                # Ensure deterministic behavior (may reduce performance)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Use deterministic algorithms where available (PyTorch >= 1.7)
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except RuntimeError:
                        # Some operations might not have deterministic implementations
                        if verbose:
                            warnings.warn("Some PyTorch operations may not be deterministic")
        
        if verbose:
            cuda_status = " (with CUDA)" if torch.cuda.is_available() else ""
            print(f"  - PyTorch{cuda_status}: ✓")
    
    # TensorFlow
    if HAS_TENSORFLOW:
        tf.random.set_seed(seed)
        
        if deterministic:
            # Enable deterministic ops (TensorFlow >= 2.0)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            
            # Set threading for reproducibility (may reduce performance)
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except RuntimeError:
                # Might fail if already initialized
                if verbose:
                    warnings.warn("Could not set TensorFlow threading parameters")
        
        if verbose:
            print(f"  - TensorFlow: ✓")
    
    # JAX
    if HAS_JAX:
        # JAX uses a different approach with explicit keys rather than global seeds
        # We can still set an initial key for consistency
        try:
            from jax import random as jax_random
            # Store the key as a module variable for later use
            global JAX_KEY
            JAX_KEY = jax_random.PRNGKey(seed)
            if verbose:
                print(f"  - JAX: ✓ (key stored as JAX_KEY)")
        except ImportError:
            if verbose:
                warnings.warn("JAX found but jax.random not available")
    
    if verbose:
        if deterministic:
            print(f"  Deterministic mode: ENABLED (may impact performance)")
        else:
            print(f"  Deterministic mode: DISABLED")


def get_numpy_generator(seed: Optional[int] = None) -> 'np.random.Generator':
    """
    Get a numpy random generator with optional seed.
    Uses the modern numpy.random.Generator API (numpy >= 1.17).
    
    Args:
        seed: Optional seed for the generator. If None, uses system entropy.
    
    Returns:
        A numpy random Generator object
    
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy is not installed")
    
    return np.random.default_rng(seed)


def get_torch_generator(seed: Optional[int] = None) -> 'torch.Generator':
    """
    Get a PyTorch random generator with optional seed.
    
    Args:
        seed: Optional seed for the generator
    
    Returns:
        A PyTorch Generator object
    
    Raises:
        ImportError: If PyTorch is not available
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed")
    
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def get_jax_key(seed: Optional[int] = None) -> 'jax.random.PRNGKey':
    """
    Get a JAX PRNG key with optional seed.
    
    Args:
        seed: Optional seed for the key. If None, uses the global JAX_KEY if available.
    
    Returns:
        A JAX PRNGKey
    
    Raises:
        ImportError: If JAX is not available
    """
    if not HAS_JAX:
        raise ImportError("JAX is not installed")
    
    from jax import random as jax_random
    
    if seed is not None:
        return jax_random.PRNGKey(seed)
    elif 'JAX_KEY' in globals():
        return JAX_KEY
    else:
        # Default seed if no global key exists
        return jax_random.PRNGKey(42)


# Module initialization: print available frameworks
def _check_frameworks():
    """Check which frameworks are available."""
    frameworks = []
    if HAS_NUMPY:
        frameworks.append("numpy")
    if HAS_TORCH:
        frameworks.append("torch")
    if HAS_TENSORFLOW:
        frameworks.append("tensorflow")
    if HAS_JAX:
        frameworks.append("jax")
    return frameworks


# Store available frameworks for reference
AVAILABLE_FRAMEWORKS = _check_frameworks()