from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from flowreg3d.core.level_solver_3d import level_solver_rbgs3d_torch
from flowreg3d.util.torch.resize_util_3D import imresize_fused_gauss_cubic3D

Tensor = torch.Tensor
resize = imresize_fused_gauss_cubic3D


# --- helpers (kept outside the public API order) ---------------------------------

def _median3d(x: Tensor, k: int = 5) -> Tensor:
    p = k // 2
    x4 = F.pad(x[None, None], (p, p, p, p, p, p), mode="replicate")
    u = x4.unfold(2, k, 1).unfold(3, k, 1).unfold(4, k, 1)
    u = u.contiguous().view(1, 1, u.size(2), u.size(3), u.size(4), -1)
    return u.median(dim=-1).values[0, 0]


def _normalize_weight_like_numpy(weight: Union[float, Tensor], C: int, shape3, device, dtype) -> Tensor:
    if isinstance(weight, torch.Tensor):
        w = weight.to(device=device, dtype=dtype)
        if w.ndim == 1:
            if w.numel() < C:
                pad = torch.full((C,), 1.0 / C, dtype=dtype, device=device)
                pad[: w.numel()] = w
                w = pad
            elif w.numel() > C:
                w = w[:C]
            w = w / w.sum()
            return w.view(1, 1, 1, -1).expand(*shape3, -1)
        if w.ndim == 3:
            return w.unsqueeze(-1).expand(*shape3, C)
        if w.ndim == 0:
            return w.view(1, 1, 1, 1).expand(*shape3, C)
        return w
    t = torch.as_tensor(weight, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.view(1, 1, 1, 1).expand(*shape3, C)
    if t.ndim == 1:
        if t.numel() < C:
            pad = torch.full((C,), 1.0 / C, dtype=dtype, device=device)
            pad[: t.numel()] = t
            t = pad
        elif t.numel() > C:
            t = t[:C]
        t = t / t.sum()
        return t.view(1, 1, 1, -1).expand(*shape3, -1)
    if t.ndim == 3:
        return t.unsqueeze(-1).expand(*shape3, C)
    raise ValueError("Unsupported weight shape")


# --- public API functions (order/names mirror original NumPy file) ---------------

def matlab_gradient(f: Tensor, spacing: float) -> Tensor:
    g = torch.zeros_like(f)
    g[1:-1] = (f[2:] - f[:-2]) / (2.0 * spacing)
    g[0] = (f[1] - f[0]) / spacing
    g[-1] = (f[-1] - f[-2]) / spacing
    return g


def imregister_wrapper(f2_level: Tensor, u: Tensor, v: Tensor, w: Tensor, f1_level: Tensor,
                       interpolation_method: str = "cubic") -> Tensor:
    if f2_level.ndim == 3:
        f2_level = f2_level.unsqueeze(-1)
        f1_level = f1_level.unsqueeze(-1)
    D, H, W, C = f2_level.shape
    z = torch.arange(D, device=f2_level.device, dtype=u.dtype).view(D, 1, 1)
    y = torch.arange(H, device=f2_level.device, dtype=u.dtype).view(1, H, 1)
    x = torch.arange(W, device=f2_level.device, dtype=u.dtype).view(1, 1, W)
    map_x = x + u
    map_y = y + v
    map_z = z + w
    oob = (map_x < 0) | (map_x > (W - 1)) | (map_y < 0) | (map_y > (H - 1)) | (map_z < 0) | (map_z > (D - 1))
    gx = 2.0 * (map_x / (W - 1.0)) - 1.0
    gy = 2.0 * (map_y / (H - 1.0)) - 1.0
    gz = 2.0 * (map_z / (D - 1.0)) - 1.0
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
    x_in = f2_level.permute(3, 0, 1, 2).unsqueeze(0)
    sampled = F.grid_sample(x_in, grid, mode="bicubic" if interpolation_method.lower() == "cubic" else "bilinear",
        padding_mode="border", align_corners=True, ).squeeze(0).permute(1, 2, 3, 0)
    if f1_level.ndim == 3:
        f1_level = f1_level.unsqueeze(-1)
    sampled[oob] = f1_level[oob]
    return sampled[..., 0] if C == 1 else sampled


def warpingDepth(eta: float, levels: int, p: int, m: int, n: int) -> int:
    min_dim = min(p, m, n)
    depth = 0
    for _ in range(levels):
        depth += 1
        min_dim *= eta
        if round(min_dim) < 10:
            break
    return depth


def add_boundary(f: Tensor) -> Tensor:
    return F.pad(f, (1, 1, 1, 1, 1, 1), mode="replicate")


def get_motion_tensor_gc(f1: Tensor, f2: Tensor, hz: float, hy: float, hx: float):
    f1p = F.pad(f1, (1, 1, 1, 1, 1, 1), mode="replicate")
    f2p = F.pad(f2, (1, 1, 1, 1, 1, 1), mode="replicate")
    gz1 = matlab_gradient(f1p, hz)
    gy1 = matlab_gradient(f1p.transpose(0, 1), hy).transpose(0, 1)
    gx1 = matlab_gradient(f1p.transpose(0, 2), hx).transpose(0, 2)
    gz2 = matlab_gradient(f2p, hz)
    gy2 = matlab_gradient(f2p.transpose(0, 1), hy).transpose(0, 1)
    gx2 = matlab_gradient(f2p.transpose(0, 2), hx).transpose(0, 2)
    fx = 0.5 * (gx1 + gx2)
    fy = 0.5 * (gy1 + gy2)
    fz = 0.5 * (gz1 + gz2)
    ft = f2p - f1p
    fx = F.pad(fx[1:-1, 1:-1, 1:-1], (1, 1, 1, 1, 1, 1), mode="replicate")
    fy = F.pad(fy[1:-1, 1:-1, 1:-1], (1, 1, 1, 1, 1, 1), mode="replicate")
    fz = F.pad(fz[1:-1, 1:-1, 1:-1], (1, 1, 1, 1, 1, 1), mode="replicate")
    ft = F.pad(ft[1:-1, 1:-1, 1:-1], (1, 1, 1, 1, 1, 1), mode="replicate")

    def _second_diff(f: Tensor, hx_: float, hy_: float, hz_: float):
        fxx = torch.zeros_like(f)
        fyy = torch.zeros_like(f)
        fzz = torch.zeros_like(f)
        fxx[:, :, 1:-1] = (f[:, :, 0:-2] - 2.0 * f[:, :, 1:-1] + f[:, :, 2:]) / (hx_ * hx_)
        fyy[:, 1:-1, :] = (f[:, 0:-2, :] - 2.0 * f[:, 1:-1, :] + f[:, 2:, :]) / (hy_ * hy_)
        fzz[1:-1, :, :] = (f[0:-2, :, :] - 2.0 * f[1:-1, :, :] + f[2:, :, :]) / (hz_ * hz_)
        return fxx, fyy, fzz

    dfx_z = matlab_gradient(fx, hz)
    dfx_y = matlab_gradient(fx.transpose(0, 1), hy).transpose(0, 1)
    dfy_z = matlab_gradient(fy, hz)
    dfy_y = matlab_gradient(fy.transpose(0, 1), hy).transpose(0, 1)
    dfx_x = matlab_gradient(fx.transpose(0, 2), hx).transpose(0, 2)
    dfy_x = matlab_gradient(fy.transpose(0, 2), hx).transpose(0, 2)
    dfz_z = matlab_gradient(fz, hz)
    dfz_y = matlab_gradient(fz.transpose(0, 1), hy).transpose(0, 1)
    dfz_x = matlab_gradient(fz.transpose(0, 2), hx).transpose(0, 2)
    dft_z = matlab_gradient(ft, hz)
    dft_y = matlab_gradient(ft.transpose(0, 1), hy).transpose(0, 1)
    dft_x = matlab_gradient(ft.transpose(0, 2), hx).transpose(0, 2)
    fxy = dfx_y
    fxz = dfx_z
    fyz = dfy_z
    fzt, fyt, fxt = dft_z, dft_y, dft_x
    fxx1, fyy1, fzz1 = _second_diff(f1p, hx, hy, hz)
    fxx2, fyy2, fzz2 = _second_diff(f2p, hx, hy, hz)
    fxx = 0.5 * (fxx1 + fxx2)
    fyy = 0.5 * (fyy1 + fyy2)
    fzz = 0.5 * (fzz1 + fzz2)
    reg_x = 1.0 / (torch.sqrt(fxx * fxx + fxy * fxy + fxz * fxz) ** 2 + 1e-6)
    reg_y = 1.0 / (torch.sqrt(fxy * fxy + fyy * fyy + fyz * fyz) ** 2 + 1e-6)
    reg_z = 1.0 / (torch.sqrt(fxz * fxz + fyz * fyz + fzz * fzz) ** 2 + 1e-6)
    J11 = reg_x * fxx * fxx + reg_y * fxy * fxy + reg_z * fxz * fxz
    J22 = reg_x * fxy * fxy + reg_y * fyy * fyy + reg_z * fyz * fyz
    J33 = reg_x * fxz * fxz + reg_y * fyz * fyz + reg_z * fzz * fzz
    J12 = reg_x * fxx * fxy + reg_y * fxy * fyy + reg_z * fxz * fyz
    J13 = reg_x * fxx * fxz + reg_y * fxy * fyz + reg_z * fxz * fzz
    J23 = reg_x * fxy * fxz + reg_y * fyy * fyz + reg_z * fyz * fzz
    J14 = reg_x * fxx * fxt + reg_y * fxy * fyt + reg_z * fxz * fzt
    J24 = reg_x * fxy * fxt + reg_y * fyy * fyt + reg_z * fyz * fzt
    J34 = reg_x * fxz * fxt + reg_y * fyz * fyt + reg_z * fzz * fzt
    J44 = reg_x * fxt * fxt + reg_y * fyt * fyt + reg_z * fzt * fzt
    for arr in (J11, J22, J33, J44, J12, J13, J23, J14, J24, J34):
        arr[:, :, 0] = 0
        arr[:, :, -1] = 0
        arr[:, 0, :] = 0
        arr[:, -1, :] = 0
        arr[0, :, :] = 0
        arr[-1, :, :] = 0
    return J11, J22, J33, J44, J12, J13, J23, J14, J24, J34


def level_solver(J11: Tensor, J22: Tensor, J33: Tensor, J44: Tensor, J12: Tensor, J13: Tensor, J23: Tensor, J14: Tensor,
                 J24: Tensor, J34: Tensor, weight: Union[float, Tensor], u: Tensor, v: Tensor, w: Tensor,
                 alpha: Tuple[float, float, float], iterations: int, update_lag: int, verbose,
                 a_data: Union[float, Tensor], a_smooth: float, hx: float, hy: float, hz: float):
    du, dv, dw = level_solver_rbgs3d_torch(J11, J22, J33, J44, J12, J13, J23, J14, J24, J34, weight, u, v, w, alpha,
        iterations, update_lag, a_data, a_smooth, hx, hy, hz, )
    return du, dv, dw


def get_displacement(fixed: Tensor, moving: Tensor, alpha: Tuple[float, float, float] = (2.0, 2.0, 2.0),
                     update_lag: int = 10, iterations: int = 20, min_level: int = 0, levels: int = 50, eta: float = 0.8,
                     a_smooth: float = 0.5, a_data: Union[float, Tensor] = 0.45, const_assumption: str = "gc",
                     uvw: Optional[Tensor] = None, weight: Optional[Union[float, Tensor]] = None) -> Tensor:
    assert const_assumption == "gc"
    # dtype parity with NumPy path
    fixed = fixed.to(torch.float64)
    moving = moving.to(torch.float64)

    if fixed.ndim == 3:
        p, m, n = fixed.shape
        fixed_ = fixed.unsqueeze(-1)
        moving_ = moving.unsqueeze(-1)
        C = 1
    else:
        p, m, n, C = fixed.shape
        fixed_ = fixed
        moving_ = moving

    if uvw is not None:
        u_init = uvw[..., 0].to(torch.float64)
        v_init = uvw[..., 1].to(torch.float64)
        w_init = uvw[..., 2].to(torch.float64)
    else:
        u_init = torch.zeros((p, m, n), dtype=torch.float64, device=fixed.device)
        v_init = torch.zeros_like(u_init)
        w_init = torch.zeros_like(u_init)

    shape3 = (p, m, n)
    if weight is None:
        weight_ = torch.ones((*shape3, C), dtype=torch.float64, device=fixed.device) / float(C)
    else:
        weight_ = _normalize_weight_like_numpy(weight, C, shape3, fixed.device, torch.float64)

    a_data_vec = (torch.full((C,), float(a_data), dtype=torch.float64, device=fixed.device) if isinstance(a_data,
                                                                                                          (float,
                                                                                                           int)) else torch.as_tensor(
        a_data, dtype=torch.float64, device=fixed.device))

    f1_low = fixed_
    f2_low = moving_

    max_level_z = warpingDepth(eta, levels, p, m, n)
    max_level_y = warpingDepth(eta, levels, m, n, p)
    max_level_x = warpingDepth(eta, levels, n, p, m)
    max_level = min(max_level_x, max_level_y, max_level_z) * 4
    max_level_z = min(max_level_z, max_level)
    max_level_y = min(max_level_y, max_level)
    max_level_x = min(max_level_x, max_level)
    if max(max_level_x, max_level_y, max_level_z) <= min_level:
        min_level = max(max_level_x, max_level_y, max_level_z) - 1
    if min_level < 0:
        min_level = 0

    u = v = w = None
    top = max(max_level_x, max_level_y, max_level_z)

    for i in range(top, min_level - 1, -1):
        lvl_D = int(round(p * (eta ** min(i, max_level_z))))
        lvl_H = int(round(m * (eta ** min(i, max_level_y))))
        lvl_W = int(round(n * (eta ** min(i, max_level_x))))
        level_size = (lvl_D, lvl_H, lvl_W)

        f1_level = resize(f1_low, level_size)
        f2_level = resize(f2_low, level_size)
        if f1_level.ndim == 3:
            f1_level = f1_level.unsqueeze(-1)
            f2_level = f2_level.unsqueeze(-1)

        cur_hz = float(p) / f1_level.shape[0]
        cur_hy = float(m) / f1_level.shape[1]
        cur_hx = float(n) / f1_level.shape[2]

        if i == top:
            u = add_boundary(resize(u_init, level_size))
            v = add_boundary(resize(v_init, level_size))
            w = add_boundary(resize(w_init, level_size))
            tmp = f2_level
        else:
            u = add_boundary(resize(u[1:-1, 1:-1, 1:-1], level_size))
            v = add_boundary(resize(v[1:-1, 1:-1, 1:-1], level_size))
            w = add_boundary(resize(w[1:-1, 1:-1, 1:-1], level_size))
            tmp = imregister_wrapper(f2_level, u[1:-1, 1:-1, 1:-1] / cur_hx, v[1:-1, 1:-1, 1:-1] / cur_hy,
                                               w[1:-1, 1:-1, 1:-1] / cur_hz, f1_level, interpolation_method="cubic", )
        if tmp.ndim == 3:
            tmp = tmp.unsqueeze(-1)

        u = u.contiguous()
        v = v.contiguous()
        w = w.contiguous()

        J_shape = (f1_level.shape[0] + 2, f1_level.shape[1] + 2, f1_level.shape[2] + 2, f1_level.shape[3])
        J11 = torch.zeros(J_shape, dtype=torch.float64, device=fixed.device)
        J22 = torch.zeros_like(J11)
        J33 = torch.zeros_like(J11)
        J44 = torch.zeros_like(J11)
        J12 = torch.zeros_like(J11)
        J13 = torch.zeros_like(J11)
        J23 = torch.zeros_like(J11)
        J14 = torch.zeros_like(J11)
        J24 = torch.zeros_like(J11)
        J34 = torch.zeros_like(J11)

        for ch in range(f1_level.shape[3]):
            Jc = get_motion_tensor_gc(f1_level[:, :, :, ch], tmp[:, :, :, ch], cur_hz, cur_hy, cur_hx)
            J11[:, :, :, ch] = Jc[0]
            J22[:, :, :, ch] = Jc[1]
            J33[:, :, :, ch] = Jc[2]
            J44[:, :, :, ch] = Jc[3]
            J12[:, :, :, ch] = Jc[4]
            J13[:, :, :, ch] = Jc[5]
            J23[:, :, :, ch] = Jc[6]
            J14[:, :, :, ch] = Jc[7]
            J24[:, :, :, ch] = Jc[8]
            J34[:, :, :, ch] = Jc[9]

        weight_level = resize(weight_, f1_level.shape[:3])
        if weight_level.ndim < 4:
            weight_level = weight_level.unsqueeze(-1)
        weight_level = F.pad(weight_level, (0, 0, 1, 1, 1, 1, 1, 1), value=0.0)

        alpha_scaling = 1.0 if i == min_level else (eta ** (-0.5 * i))
        alpha_i = (alpha_scaling * alpha[0], alpha_scaling * alpha[1], alpha_scaling * alpha[2])

        du, dv, dw = level_solver(J11.contiguous(), J22.contiguous(), J33.contiguous(), J44.contiguous(),
            J12.contiguous(), J13.contiguous(), J23.contiguous(), J14.contiguous(), J24.contiguous(), J34.contiguous(),
            weight_level.contiguous(), u, v, w, alpha_i, iterations, update_lag, None, a_data_vec, a_smooth, cur_hx,
            cur_hy, cur_hz, )

        if min(level_size) > 5:
            core = (slice(1, -1), slice(1, -1), slice(1, -1))
            du[core] = _median3d(du[core], 5)
            dv[core] = _median3d(dv[core], 5)
            dw[core] = _median3d(dw[core], 5)

        u = u + du
        v = v + dv
        w = w + dw

    flow = torch.zeros((u.shape[0] - 2, u.shape[1] - 2, u.shape[2] - 2, 3), dtype=torch.float64, device=fixed.device)
    flow[..., 0] = u[1:-1, 1:-1, 1:-1]
    flow[..., 1] = v[1:-1, 1:-1, 1:-1]
    flow[..., 2] = w[1:-1, 1:-1, 1:-1]

    if min_level > 0:
        flow_resized = torch.zeros((p, m, n, 3), dtype=torch.float64, device=fixed.device)
        flow_resized[..., 0] = resize(flow[..., 0], (p, m, n))
        flow_resized[..., 1] = resize(flow[..., 1], (p, m, n))
        flow_resized[..., 2] = resize(flow[..., 2], (p, m, n))
        flow = flow_resized

    return flow
