"""
PyTorch implementation of image processing utilities with exact parity to NumPy version.
Provides normalization and Gaussian filtering functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, Union
from collections import deque
from functools import lru_cache


def normalize(
    arr: Union[np.ndarray, torch.Tensor],
    ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
    channel_normalization: Literal["together", "separate"] = "together",
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize array to [0,1] range with exact parity to NumPy version.

    Args:
        arr: Array to normalize, shape (Z,Y,X,C) or (T,Z,Y,X,C)
        ref: Optional reference for normalization ranges
        channel_normalization: 'separate' for per-channel, 'together' for global
        eps: Small value to avoid division by zero

    Returns:
        Normalized array in [0,1] range (float64 for parity)
    """
    # Handle input types
    if isinstance(arr, np.ndarray):
        input_numpy = True
        arr_tensor = torch.from_numpy(arr)
    else:
        input_numpy = False
        arr_tensor = arr

    if ref is not None:
        if isinstance(ref, np.ndarray):
            ref_tensor = torch.from_numpy(ref)
        else:
            ref_tensor = ref
    else:
        ref_tensor = None

    device = arr_tensor.device

    if channel_normalization == "separate":
        # Per-channel normalization - matches NumPy's float64 allocation
        result = torch.zeros_like(arr_tensor, dtype=torch.float64, device=device)

        if arr_tensor.ndim == 4:  # (Z,Y,X,C)
            for c in range(arr_tensor.shape[3]):
                if ref_tensor is not None and ref_tensor.ndim >= 4:
                    # Use reference's min/max for this channel
                    ref_channel = ref_tensor[..., c]
                    min_val = ref_channel.min()
                    max_val = ref_channel.max()
                else:
                    # Use array's own min/max
                    channel = arr_tensor[..., c]
                    min_val = channel.min()
                    max_val = channel.max()

                # Avoid division by zero
                value_range = max_val - min_val
                if value_range > 0:
                    result[..., c] = (arr_tensor[..., c] - min_val) / value_range
                else:
                    result[..., c] = arr_tensor[..., c] - min_val  # All same value

        elif arr_tensor.ndim == 5:  # (T,Z,Y,X,C)
            for c in range(arr_tensor.shape[4]):
                if ref_tensor is not None and ref_tensor.ndim >= 4:
                    ref_channel = ref_tensor[..., c]
                    min_val = ref_channel.min()
                    max_val = ref_channel.max()
                else:
                    channel = arr_tensor[..., c]
                    min_val = channel.min()
                    max_val = channel.max()

                # Avoid division by zero
                value_range = max_val - min_val
                if value_range > 0:
                    result[..., c] = (arr_tensor[..., c] - min_val) / value_range
                else:
                    result[..., c] = arr_tensor[..., c] - min_val  # All same value
        else:
            # 3D or unsupported, use global normalization
            if ref_tensor is not None:
                min_val = ref_tensor.min()
                max_val = ref_tensor.max()
            else:
                min_val = arr_tensor.min()
                max_val = arr_tensor.max()
            value_range = max_val - min_val
            if value_range > 0:
                result = (arr_tensor - min_val) / value_range
            else:
                result = arr_tensor - min_val

            # Ensure float64 for parity
            result = result.to(torch.float64)

            # Convert back to appropriate type
            if input_numpy:
                return result.cpu().numpy()
            else:
                return result

        # Convert back for separate channel case (float64 result)
        if input_numpy:
            return result.cpu().numpy()
        else:
            return result

    else:
        # Global normalization - must return float64 for parity
        if ref_tensor is not None:
            min_val = ref_tensor.min()
            max_val = ref_tensor.max()
        else:
            min_val = arr_tensor.min()
            max_val = arr_tensor.max()

        # Match NumPy version which uses eps in denominator
        result = (arr_tensor - min_val) / (max_val - min_val + eps)

        # Force float64 for exact parity with NumPy version
        result = result.to(torch.float64)

        if input_numpy:
            return result.cpu().numpy()
        else:
            return result


@lru_cache(maxsize=128)
def _get_gaussian_kernel_1d_cpu(
    sigma: float, truncate: float, dtype_str: str
) -> np.ndarray:
    """
    Create 1D Gaussian kernel on CPU matching scipy's implementation.
    Cached on CPU to avoid device duplication.

    Args:
        sigma: Standard deviation
        truncate: Truncate at this many standard deviations
        dtype_str: 'float32' or 'float64'

    Returns:
        1D Gaussian kernel as numpy array
    """
    dtype = np.float64 if dtype_str == "float64" else np.float32

    if sigma <= 0:
        kernel = np.array([1.0], dtype=dtype)
    else:
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = (kernel / kernel.sum()).astype(dtype)

    return kernel


def _apply_separable_gaussian_3d(
    arr: torch.Tensor, sigmas: tuple, truncate: float = 4.0
) -> torch.Tensor:
    """
    Apply separable 3D Gaussian filter using 1D convolutions.

    Args:
        arr: Input tensor shape (Z, Y, X) or (T, Z, Y, X) for batched
        sigmas: Tuple of (sz, sy, sx) in scipy order
        truncate: Truncation parameter

    Returns:
        Filtered tensor
    """
    sz, sy, sx = sigmas
    dtype_str = "float64" if arr.dtype == torch.float64 else "float32"
    device = arr.device

    # Ensure contiguous
    arr = arr.contiguous()

    # Handle both 3D and 4D (batched) inputs
    if arr.ndim == 3:
        # Add batch and channel dims for conv3d: (1, 1, Z, Y, X)
        x = arr.unsqueeze(0).unsqueeze(0)
        batched = False
    else:  # arr.ndim == 4, already (T, Z, Y, X)
        # Add channel dim: (T, 1, Z, Y, X)
        x = arr.unsqueeze(1)
        batched = True

    # Apply separable 1D convolutions
    # Z-axis
    if sz > 0:
        kernel_np = _get_gaussian_kernel_1d_cpu(sz, truncate, dtype_str)
        kernel_z = torch.from_numpy(kernel_np).to(device=device, dtype=arr.dtype)
        radius_z = (len(kernel_z) - 1) // 2
        if radius_z > 0:
            x = F.pad(x, (0, 0, 0, 0, radius_z, radius_z), mode="reflect")
        weight_z = kernel_z.view(1, 1, -1, 1, 1)
        x = F.conv3d(x, weight_z, stride=1, padding=0)

    # Y-axis
    if sy > 0:
        kernel_np = _get_gaussian_kernel_1d_cpu(sy, truncate, dtype_str)
        kernel_y = torch.from_numpy(kernel_np).to(device=device, dtype=arr.dtype)
        radius_y = (len(kernel_y) - 1) // 2
        if radius_y > 0:
            x = F.pad(x, (0, 0, radius_y, radius_y, 0, 0), mode="reflect")
        weight_y = kernel_y.view(1, 1, 1, -1, 1)
        x = F.conv3d(x, weight_y, stride=1, padding=0)

    # X-axis
    if sx > 0:
        kernel_np = _get_gaussian_kernel_1d_cpu(sx, truncate, dtype_str)
        kernel_x = torch.from_numpy(kernel_np).to(device=device, dtype=arr.dtype)
        radius_x = (len(kernel_x) - 1) // 2
        if radius_x > 0:
            x = F.pad(x, (radius_x, radius_x, 0, 0, 0, 0), mode="reflect")
        weight_x = kernel_x.view(1, 1, 1, 1, -1)
        x = F.conv3d(x, weight_x, stride=1, padding=0)

    # Remove added dimensions
    if batched:
        return x.squeeze(1)  # Remove channel dim
    else:
        return x.squeeze(0).squeeze(0)  # Remove batch and channel dims


def apply_gaussian_filter(
    arr: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    mode: str = "reflect",
    truncate: float = 4.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply Gaussian filtering with exact parity to NumPy/SciPy version.

    Args:
        arr: Input array, shape (Z,Y,X,C) or (T,Z,Y,X,C)
        sigma: Standard deviation for Gaussian kernel
               - array shape (4,): [sx, sy, sz, st] for all channels
               - array shape (n_channels, 4): per-channel sigmas
        mode: Boundary handling mode (only 'reflect' supported)
        truncate: Truncate filter at this many standard deviations

    Returns:
        Filtered array (always float64 to match NumPy version)
    """
    # Enforce mode='reflect' for parity
    if mode != "reflect":
        raise ValueError("Only 'reflect' mode is supported to match CPU behavior.")

    # Handle input conversion
    if isinstance(arr, np.ndarray):
        input_numpy = True
        arr_tensor = torch.from_numpy(arr)
    else:
        input_numpy = False
        arr_tensor = arr

    if isinstance(sigma, torch.Tensor):
        sigma_np = sigma.cpu().numpy()
    else:
        sigma_np = np.asarray(sigma)

    device = arr_tensor.device

    # Allocate float64 result to match NumPy version
    result = torch.zeros_like(arr_tensor, dtype=torch.float64, device=device)

    if arr_tensor.ndim == 4:  # (Z,Y,X,C) - 3D spatial only
        for c in range(arr_tensor.shape[3]):
            if sigma_np.ndim == 2:  # Per-channel sigmas
                s = sigma_np[min(c, len(sigma_np) - 1), :3]  # Use spatial components
            else:
                s = sigma_np[:3] if len(sigma_np) >= 3 else sigma_np

            # Reorder from (sx, sy, sz) to (sz, sy, sx) for consistency
            if len(s) == 3:
                s_3d = (s[2], s[1], s[0])
            else:
                s_3d = tuple(s) if len(s) > 0 else (0, 0, 0)

            # Apply separable Gaussian filter
            channel_data = arr_tensor[..., c].to(torch.float64)
            result[..., c] = _apply_separable_gaussian_3d(channel_data, s_3d, truncate)

    elif arr_tensor.ndim == 5:  # (T,Z,Y,X,C) - 4D spatiotemporal
        for c in range(arr_tensor.shape[4]):
            if sigma_np.ndim == 2:  # Per-channel sigmas
                s = sigma_np[min(c, len(sigma_np) - 1)]
                # Reorder from (sx, sy, sz, st) to (st, sz, sy, sx)
                if len(s) == 4:
                    st, sz, sy, sx = s[3], s[2], s[1], s[0]
                else:
                    st = sz = sy = sx = 0.0
            else:
                if len(sigma_np) == 4:
                    st, sz, sy, sx = sigma_np[3], sigma_np[2], sigma_np[1], sigma_np[0]
                else:
                    st = sz = sy = sx = 0.0

            # Get channel data
            channel_data = arr_tensor[..., c].to(torch.float64)  # (T, Z, Y, X)

            # Spatial filtering - batched over time dimension
            if sz > 0 or sy > 0 or sx > 0:
                # Apply batched 3D spatial filtering (treat T as batch)
                channel_data = _apply_separable_gaussian_3d(
                    channel_data, (sz, sy, sx), truncate
                )

            # Temporal filtering along T dimension
            if st > 0:
                # Apply 1D convolution along time dimension
                kernel_np = _get_gaussian_kernel_1d_cpu(st, truncate, "float64")
                kernel_t = torch.from_numpy(kernel_np).to(
                    device=device, dtype=torch.float64
                )
                radius_t = (len(kernel_t) - 1) // 2

                # Reshape for 1D conv: (1, 1, T, Z*Y*X)
                T, Z, Y, X = channel_data.shape
                data_reshaped = channel_data.reshape(1, 1, T, -1)

                if radius_t > 0:
                    data_reshaped = F.pad(
                        data_reshaped, (0, 0, radius_t, radius_t), mode="reflect"
                    )

                # Apply temporal convolution
                weight_t = kernel_t.view(1, 1, -1, 1)
                data_filtered = F.conv2d(data_reshaped, weight_t, stride=1, padding=0)

                # Reshape back
                channel_data = data_filtered.reshape(T, Z, Y, X)

            result[..., c] = channel_data

    else:
        # Unsupported dimensionality - apply directly (ensure float64)
        arr_float = arr_tensor.to(torch.float64)
        result = _apply_separable_gaussian_3d(
            arr_float, tuple(sigma_np) if len(sigma_np) >= 3 else (0, 0, 0), truncate
        )

    # Return with same type as input
    if input_numpy:
        return result.cpu().numpy()
    else:
        return result


def gaussian_filter_1d_half_kernel(
    buffer: deque, sigma_t: float, mode: str = "reflect", truncate: float = 4.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Fast 1D Gaussian filter using half kernel for temporal dimension.
    Exact parity with NumPy version.

    Args:
        buffer: deque of 2D filtered frames [oldest...newest]
        sigma_t: Temporal standard deviation
        mode: Boundary handling mode (unused, kept for API consistency)
        truncate: Truncate filter at this many standard deviations

    Returns:
        Temporally filtered current frame (last in buffer)
    """
    if not buffer or len(buffer) == 0:
        return None

    if len(buffer) == 1:
        return (
            buffer[-1].copy()
            if isinstance(buffer[-1], np.ndarray)
            else buffer[-1].clone()
        )

    # No temporal filtering if sigma is 0
    if sigma_t <= 0:
        return (
            buffer[-1].copy()
            if isinstance(buffer[-1], np.ndarray)
            else buffer[-1].clone()
        )

    # Determine if we're working with numpy or torch
    input_numpy = isinstance(buffer[-1], np.ndarray)

    # Create half Gaussian kernel (matching NumPy implementation)
    kernel_radius = int(truncate * sigma_t + 0.5)
    kernel_size = min(kernel_radius + 1, len(buffer))  # Half kernel including center

    # Generate half Gaussian kernel weights
    if input_numpy:
        x = np.arange(kernel_size, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma_t) ** 2)
        kernel = kernel / kernel.sum()

        # Apply weighted average using half kernel
        result = np.zeros_like(buffer[-1], dtype=np.float64)

        for i in range(kernel_size):
            # Index from the end of buffer
            frame_idx = -(i + 1)
            result += kernel[i] * buffer[frame_idx]

        return result.astype(buffer[-1].dtype)
    else:
        # PyTorch version
        device = buffer[-1].device
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x / sigma_t) ** 2)
        kernel = kernel / kernel.sum()

        # Apply weighted average
        result = torch.zeros_like(buffer[-1], dtype=torch.float64)

        for i in range(kernel_size):
            frame_idx = -(i + 1)
            result += kernel[i] * buffer[frame_idx].to(torch.float64)

        return result.to(buffer[-1].dtype)
