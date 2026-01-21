"""
PyTorch implementation of resize_util_3D with exact numerical parity to NumPy version.
Implements fused Gauss+cubic interpolation with reflect boundary conditions.
"""

import torch
import numpy as np
from functools import lru_cache
from typing import Tuple, Union


# Keys cubic parameter (exact same as NumPy version)
A = -0.75


def _cubic_weight_np(x: np.ndarray) -> np.ndarray:
    """Compute cubic interpolation weight (Keys cubic) in NumPy."""
    ax = np.abs(x)
    weight = np.zeros_like(x)

    # |x| < 1
    mask1 = ax < 1.0
    weight[mask1] = (A + 2.0) * ax[mask1] ** 3 - (A + 3.0) * ax[mask1] ** 2 + 1.0

    # 1 <= |x| < 2
    mask2 = (ax >= 1.0) & (ax < 2.0)
    weight[mask2] = (
        A * ax[mask2] ** 3 - 5.0 * A * ax[mask2] ** 2 + 8.0 * A * ax[mask2] - 4.0 * A
    )

    return weight


def _reflect_idx_np(j: np.ndarray, n: int) -> np.ndarray:
    """Apply reflect boundary condition using closed-form solution."""
    if n <= 1:
        return np.zeros_like(j, dtype=np.int64)

    period = 2 * n - 2
    r = j % period
    r = np.where(r < 0, r + period, r)
    return np.where(r < n, r, period - r).astype(np.int64)


def _reflect_idx_torch(j: torch.Tensor, n: int) -> torch.Tensor:
    """Apply reflect boundary condition using closed-form solution (torch)."""
    if n <= 1:
        return torch.zeros_like(j, dtype=torch.long)

    period = 2 * n - 2
    r = j.remainder(period)
    r = torch.where(r < 0, r + period, r)
    return torch.where(r < n, r, period - r).to(torch.long)


@lru_cache(maxsize=64)
def _precompute_fused_gauss_cubic_cpu(
    in_len: int, out_len: int, sigma_rounded: float, dtype_str: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute sampling indices and weights for fused Gauss+cubic interpolation.
    Computed on CPU in NumPy for cache efficiency, matching NumPy version exactly.

    Args:
        in_len: Input dimension length
        out_len: Output dimension length
        sigma_rounded: Gaussian sigma (rounded to avoid cache misses)
        dtype_str: 'float32' or 'float64'

    Returns:
        idx: int64 array of shape (out_len, P)
        wt: dtype array of shape (out_len, P)
    """
    sigma = sigma_rounded
    scale = out_len / in_len
    dtype = np.float64 if dtype_str == "float64" else np.float32

    # Gaussian kernel setup (same as NumPy version)
    if sigma <= 0.0:
        R = 0
        g = np.array([1.0], dtype=dtype)
    else:
        R = int(np.ceil(2.0 * sigma))
        x = np.arange(-R, R + 1, dtype=dtype)
        g = np.exp(-0.5 * (x / sigma) ** 2).astype(dtype)
        g /= g.sum()

    P = 2 * R + 4
    idx = np.zeros((out_len, P), dtype=np.int64)
    wt = np.zeros((out_len, P), dtype=dtype)

    # Vectorized weight computation
    u = np.arange(-R, R + 1, dtype=dtype) if R > 0 else np.array([0.0], dtype=dtype)

    for i in range(out_len):
        x = (i + 0.5) / scale - 0.5
        left = int(np.floor(x - 2.0)) - R

        # Compute indices with reflection
        j = left + np.arange(P)
        idx[i, :] = _reflect_idx_np(j, in_len)

        # Compute weights with Gaussian convolution (vectorized)
        d = x - j  # shape (P,)
        if R > 0:
            # Convolve Gaussian with cubic: sum over u dimension
            # d[:, None] - u[None, :] has shape (P, 2R+1)
            cubic_vals = _cubic_weight_np(d[:, None] - u[None, :])  # (P, 2R+1)
            w = np.sum(g[None, :] * cubic_vals, axis=1)  # (P,)
        else:
            # No Gaussian blur, just cubic
            w = _cubic_weight_np(d)

        # Normalize weights
        wt[i, :] = w / w.sum()

    return idx, wt


def _resize_x3d_torch(
    src: torch.Tensor, idx: torch.Tensor, wt: torch.Tensor
) -> torch.Tensor:
    """Resize along X axis using gather."""
    src = src.contiguous()

    if src.ndim == 3:
        D, H, W = src.shape
        ow, P = wt.shape

        # Gather along W dimension
        src_expanded = src.unsqueeze(2).expand(D, H, ow, W)  # (D, H, ow, W)
        idx_expanded = (
            idx.unsqueeze(0).unsqueeze(0).expand(D, H, ow, P)
        )  # (D, H, ow, P)

        gathered = torch.gather(
            src_expanded, dim=3, index=idx_expanded
        )  # (D, H, ow, P)
        dst = (gathered * wt.unsqueeze(0).unsqueeze(0)).sum(dim=3)  # (D, H, ow)

    else:  # Batched: (N, D, H, W)
        N, D, H, W = src.shape
        ow, P = wt.shape

        # Reshape to merge batch and depth
        src_flat = src.reshape(N * D, H, W).contiguous()

        src_expanded = src_flat.unsqueeze(2).expand(N * D, H, ow, W)
        idx_expanded = idx.unsqueeze(0).unsqueeze(0).expand(N * D, H, ow, P)

        gathered = torch.gather(src_expanded, dim=3, index=idx_expanded)
        dst_flat = (gathered * wt.unsqueeze(0).unsqueeze(0)).sum(dim=3)

        dst = dst_flat.reshape(N, D, H, ow)

    return dst


def _resize_y3d_torch(
    src: torch.Tensor, idx: torch.Tensor, wt: torch.Tensor
) -> torch.Tensor:
    """Resize along Y axis using gather."""
    src = src.contiguous()

    if src.ndim == 3:
        D, H, W = src.shape
        oh, P = wt.shape

        # Gather along H dimension
        src_expanded = src.unsqueeze(1).expand(D, oh, H, W)  # (D, oh, H, W)
        idx_expanded = (
            idx.unsqueeze(0).unsqueeze(3).expand(D, oh, P, W)
        )  # (D, oh, P, W)

        gathered = torch.gather(
            src_expanded, dim=2, index=idx_expanded
        )  # (D, oh, P, W)
        dst = (gathered * wt.unsqueeze(0).unsqueeze(3)).sum(dim=2)  # (D, oh, W)

    else:  # Batched
        N, D, H, W = src.shape
        oh, P = wt.shape

        # Merge batch
        src_flat = src.reshape(N * D, H, W).contiguous()

        src_expanded = src_flat.unsqueeze(1).expand(N * D, oh, H, W)
        idx_expanded = idx.unsqueeze(0).unsqueeze(3).expand(N * D, oh, P, W)

        gathered = torch.gather(src_expanded, dim=2, index=idx_expanded)
        dst_flat = (gathered * wt.unsqueeze(0).unsqueeze(3)).sum(dim=2)

        dst = dst_flat.reshape(N, D, oh, W)

    return dst


def _resize_z3d_torch(
    src: torch.Tensor, idx: torch.Tensor, wt: torch.Tensor
) -> torch.Tensor:
    """Resize along Z axis using gather - fully vectorized."""
    src = src.contiguous()

    if src.ndim == 3:
        D, H, W = src.shape
        od, P = wt.shape

        # Gather along D dimension
        src_expanded = src.unsqueeze(0).expand(od, D, H, W)  # (od, D, H, W)
        idx_expanded = idx.view(od, P, 1, 1).expand(od, P, H, W)  # (od, P, H, W)

        gathered = torch.gather(
            src_expanded, dim=1, index=idx_expanded
        )  # (od, P, H, W)
        dst = (gathered * wt.view(od, P, 1, 1)).sum(dim=1)  # (od, H, W)

    else:  # Batched - fully vectorized
        N, D, H, W = src.shape
        od, P = wt.shape

        # Transpose to put D first for gathering
        src_transposed = src.transpose(0, 1).contiguous()  # (D, N, H, W)
        src_expanded = src_transposed.unsqueeze(0).expand(
            od, D, N, H, W
        )  # (od, D, N, H, W)

        # Expand indices
        idx_expanded = idx.view(od, P, 1, 1, 1).expand(
            od, P, N, H, W
        )  # (od, P, N, H, W)

        # Gather and apply weights
        gathered = torch.gather(
            src_expanded, dim=1, index=idx_expanded
        )  # (od, P, N, H, W)
        dst = (gathered * wt.view(od, P, 1, 1, 1)).sum(dim=1)  # (od, N, H, W)

        # Transpose back
        dst = dst.transpose(0, 1).contiguous()  # (N, od, H, W)

    return dst


def imresize_fused_gauss_cubic3D(
    img: Union[np.ndarray, torch.Tensor],
    size: Tuple[int, int, int],
    sigma_coeff: float = 0.6,
    per_axis: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    3D resize with fused Gaussian+cubic interpolation.
    Exact parity with NumPy version including dtype handling.

    Args:
        img: Input image (D,H,W) or (D,H,W,C)
        size: Output size (od, oh, ow)
        sigma_coeff: Gaussian blur coefficient
        per_axis: Whether to compute sigma per axis

    Returns:
        Resized image with same dtype as input
    """
    # Determine compute dtype based on input (force float32 like NumPy backend)
    if isinstance(img, np.ndarray):
        input_numpy = True
        input_dtype = img.dtype
        compute_dtype = np.float32
        img_tensor = torch.from_numpy(img.astype(compute_dtype))
        compute_torch_dtype = torch.float32
    else:
        input_numpy = False
        input_dtype = img.dtype
        compute_torch_dtype = torch.float32
        img_tensor = img.to(compute_torch_dtype)

    device = img_tensor.device
    dtype_str = "float64" if compute_torch_dtype == torch.float64 else "float32"
    od, oh, ow = size[:3]

    with torch.no_grad():
        # Handle 4D input (with channels)
        if img_tensor.ndim == 4:
            D, H, W, C = img_tensor.shape
            # Reshape to (C, D, H, W) for batch processing
            x = img_tensor.permute(3, 0, 1, 2).contiguous()
        else:
            x = img_tensor.unsqueeze(0)  # Add batch dimension

        # Compute sigmas (same logic as NumPy)
        sz = od / x.shape[1]  # D is at dim 1 now
        sy = oh / x.shape[2]
        sx = ow / x.shape[3]

        if per_axis:
            sigx = sigma_coeff / sx if sx < 1.0 else 0.0
            sigy = sigma_coeff / sy if sy < 1.0 else 0.0
            sigz = sigma_coeff / sz if sz < 1.0 else 0.0
        else:
            s = min(sx, sy, sz)
            val = (sigma_coeff / s) if s < 1.0 else 0.0
            sigx = sigy = sigz = val

        # Get precomputed tables from CPU cache (round sigma to avoid cache misses)
        idx_cpu_x, wt_cpu_x = _precompute_fused_gauss_cubic_cpu(
            x.shape[3], ow, round(sigx, 6), dtype_str
        )
        idx_cpu_y, wt_cpu_y = _precompute_fused_gauss_cubic_cpu(
            x.shape[2], oh, round(sigy, 6), dtype_str
        )
        idx_cpu_z, wt_cpu_z = _precompute_fused_gauss_cubic_cpu(
            x.shape[1], od, round(sigz, 6), dtype_str
        )

        # Move tables to device
        idx_x = torch.from_numpy(idx_cpu_x).to(device)
        wt_x = torch.from_numpy(wt_cpu_x).to(device=device, dtype=compute_torch_dtype)

        idx_y = torch.from_numpy(idx_cpu_y).to(device)
        wt_y = torch.from_numpy(wt_cpu_y).to(device=device, dtype=compute_torch_dtype)

        idx_z = torch.from_numpy(idx_cpu_z).to(device)
        wt_z = torch.from_numpy(wt_cpu_z).to(device=device, dtype=compute_torch_dtype)

        # Three-pass separable resize
        tmp1 = _resize_x3d_torch(x, idx_x, wt_x)  # (C, D, H, ow)
        tmp2 = _resize_y3d_torch(tmp1, idx_y, wt_y)  # (C, D, oh, ow)
        y = _resize_z3d_torch(tmp2, idx_z, wt_z)  # (C, od, oh, ow)

        # Reshape back to original layout
        if img_tensor.ndim == 4:
            # (C, od, oh, ow) -> (od, oh, ow, C)
            y = y.permute(1, 2, 3, 0).contiguous()
        else:
            # Remove batch dimension
            y = y.squeeze(0)

        # Convert back to original dtype with integer-safe rounding/clipping
        if input_numpy:
            y_np = y.cpu().numpy()
            if np.issubdtype(input_dtype, np.integer):
                info = np.iinfo(input_dtype)
                y_np = np.rint(y_np)
                np.clip(y_np, info.min, info.max, out=y_np)
                return y_np.astype(input_dtype)
            return y_np.astype(input_dtype, copy=False)
        else:
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if input_dtype in integer_dtypes:
                info = torch.iinfo(input_dtype)
                y = y.round()
                y = torch.clamp(y, min=info.min, max=info.max)
                return y.to(dtype=input_dtype)
            return y.to(dtype=input_dtype)


def imresize2d_gauss_cubic(
    img2d: Union[np.ndarray, torch.Tensor],
    out_hw: Tuple[int, int],
    sigma_coeff: float = 0.6,
) -> Union[np.ndarray, torch.Tensor]:
    """
    2D resize wrapper using 3D implementation.
    """
    # Add singleton Z dimension
    if isinstance(img2d, np.ndarray):
        img3d = img2d[np.newaxis, ...]
    else:
        img3d = img2d.unsqueeze(0)

    # Resize with singleton Z
    y = imresize_fused_gauss_cubic3D(
        img3d,
        (1, int(out_hw[0]), int(out_hw[1])),
        sigma_coeff=sigma_coeff,
        per_axis=True,
    )

    # Remove singleton Z dimension
    if isinstance(y, np.ndarray):
        return y[0]
    else:
        return y.squeeze(0)
