import numpy as np
from numba import njit


A = -0.75


@njit(fastmath=True, cache=True)
def _resize_x3d(src, idx, wt):
    D, H, W = src.shape
    ow, P = wt.shape
    dst = np.empty((D, H, ow), src.dtype)
    for z in range(D):
        for y in range(H):
            for i in range(ow):
                s = 0.0
                for p in range(P):
                    s += src[z, y, idx[i, p]] * wt[i, p]
                dst[z, y, i] = s
    return dst


@njit(fastmath=True, cache=True)
def _resize_y3d(src, idx, wt):
    D, H, W = src.shape
    oh, P = wt.shape
    dst = np.empty((D, oh, W), src.dtype)
    for z in range(D):
        for x in range(W):
            for i in range(oh):
                s = 0.0
                for p in range(P):
                    s += src[z, idx[i, p], x] * wt[i, p]
                dst[z, i, x] = s
    return dst


@njit(fastmath=True, cache=True)
def _resize_z3d(src, idx, wt):
    D, H, W = src.shape
    od, P = wt.shape
    dst = np.empty((od, H, W), src.dtype)
    for y in range(H):
        for x in range(W):
            for i in range(od):
                s = 0.0
                for p in range(P):
                    s += src[idx[i, p], y, x] * wt[i, p]
                dst[i, y, x] = s
    return dst


@njit(inline="always")
def _cubic(x):
    ax = abs(x)
    if ax < 1.0:
        return (A + 2.0) * ax**3 - (A + 3.0) * ax**2 + 1.0
    elif ax < 2.0:
        return A * ax**3 - 5.0 * A * ax**2 + 8.0 * A * ax - 4.0 * A
    else:
        return 0.0


@njit(inline="always")
def _reflect_idx(j, n):
    if n <= 1:
        return 0
    while j < 0 or j >= n:
        if j < 0:
            j = -j - 1
        else:
            j = 2 * n - 1 - j
    return j


@njit(fastmath=True, cache=True)
def _fill_tables_fused_gauss_cubic_reflect(idx, wt, in_len, out_len, scale, g, R):
    P = 2 * R + 4
    for i in range(out_len):
        x = (i + 0.5) / scale - 0.5
        left = int(np.floor(x - 2.0)) - R
        ssum = 0.0
        for p in range(P):
            j = left + p
            jj = _reflect_idx(j, in_len)
            idx[i, p] = jj
            d = x - j
            acc = 0.0
            for u in range(-R, R + 1):
                acc += g[u + R] * _cubic(d - u)
            wt[i, p] = acc
            ssum += acc
        inv = 1.0 / ssum
        for p in range(P):
            wt[i, p] *= inv


def _precompute_fused_gauss_cubic(in_len, out_len, sigma):
    scale = out_len / in_len
    if sigma <= 0.0:
        R = 0
        g = np.array([1.0], dtype=np.float32)
    else:
        R = int(np.ceil(2.0 * sigma))
        x = np.arange(-R, R + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        g /= g.sum()
    idx = np.empty((out_len, 2 * R + 4), np.int32)
    wt = np.empty((out_len, 2 * R + 4), np.float32)
    _fill_tables_fused_gauss_cubic_reflect(idx, wt, in_len, out_len, scale, g, R)
    return idx, wt


def imresize_fused_gauss_cubic3D(img, size, sigma_coeff=0.6, per_axis=False):
    od, oh, ow = size[:3]
    x = img.astype(np.float32, copy=False)
    sz = od / x.shape[0]
    sy = oh / x.shape[1]
    sx = ow / x.shape[2]
    if per_axis:
        sigx = sigma_coeff / sx if sx < 1.0 else 0.0
        sigy = sigma_coeff / sy if sy < 1.0 else 0.0
        sigz = sigma_coeff / sz if sz < 1.0 else 0.0
    else:
        s = sx
        if sy < s:
            s = sy
        if sz < s:
            s = sz
        val = (sigma_coeff / s) if s < 1.0 else 0.0
        sigx = sigy = sigz = val
    idx_x, wt_x = _precompute_fused_gauss_cubic(x.shape[2], ow, sigx)
    idx_y, wt_y = _precompute_fused_gauss_cubic(x.shape[1], oh, sigy)
    idx_z, wt_z = _precompute_fused_gauss_cubic(x.shape[0], od, sigz)
    if x.ndim == 3:
        tmp1 = _resize_x3d(x, idx_x, wt_x)
        tmp2 = _resize_y3d(tmp1, idx_y, wt_y)
        y = _resize_z3d(tmp2, idx_z, wt_z)
    elif x.ndim == 4:
        c = x.shape[3]
        y = np.empty((od, oh, ow, c), np.float32)
        for k in range(c):
            tmp1 = _resize_x3d(x[:, :, :, k], idx_x, wt_x)
            tmp2 = _resize_y3d(tmp1, idx_y, wt_y)
            y[:, :, :, k] = _resize_z3d(tmp2, idx_z, wt_z)
    else:
        raise ValueError("img must be 3D or 4D with channels-last")

    # Preserve integer semantics: round then clip to dtype range before casting.
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        y = np.rint(y)
        np.clip(y, info.min, info.max, out=y)
        return y.astype(img.dtype)

    return y.astype(img.dtype, copy=False)


def imresize2d_gauss_cubic(img2d, out_hw, sigma_coeff=0.6):
    y = imresize_fused_gauss_cubic3D(
        img2d[None, ...],
        (1, int(out_hw[0]), int(out_hw[1])),
        sigma_coeff=sigma_coeff,
        per_axis=True,
    )
    return y[0]
