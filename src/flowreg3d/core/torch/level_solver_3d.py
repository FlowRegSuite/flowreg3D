import torch


def _set_boundary3d_(A):
    A[:, 0, :] = A[:, 1, :]
    A[:, -1, :] = A[:, -2, :]
    A[:, :, 0] = A[:, :, 1]
    A[:, :, -1] = A[:, :, -2]
    A[0, :, :] = A[1, :, :]
    A[-1, :, :] = A[-2, :, :]


def _as_W_like(weight_level, like):
    if isinstance(weight_level, (float, int)):
        return torch.as_tensor(weight_level, dtype=like.dtype, device=like.device).view(
            1, 1, 1, 1
        )
    if not isinstance(weight_level, torch.Tensor):
        weight_level = torch.as_tensor(
            weight_level, dtype=like.dtype, device=like.device
        )
    if weight_level.dim() == 0:
        return weight_level.to(dtype=like.dtype, device=like.device).view(1, 1, 1, 1)
    if weight_level.dim() == 1:
        return weight_level.to(dtype=like.dtype, device=like.device).view(1, 1, 1, -1)
    if weight_level.dim() == 4 and tuple(weight_level.shape) == tuple(like.shape):
        return weight_level
    raise ValueError(
        "weight_level must be (p,m,n,C), (C,), scalar, or convertible to such"
    )


def level_solver_rbgs3d_torch(
    J11,
    J22,
    J33,
    J44,
    J12,
    J13,
    J23,
    J14,
    J24,
    J34,
    weight_level,
    u,
    v,
    w,
    alpha,
    iterations,
    update_lag,
    a_data,
    a_smooth,
    hx,
    hy,
    hz,
    omega=1.95,
    eps=1e-6,
):
    with torch.no_grad():
        P, M, N, C = J11.shape
        du = torch.zeros_like(u)
        dv = torch.zeros_like(v)
        dw = torch.zeros_like(w)
        ax = torch.as_tensor(alpha[0], dtype=u.dtype, device=u.device) / (hx * hx)
        ay = torch.as_tensor(alpha[1], dtype=u.dtype, device=u.device) / (hy * hy)
        az = torch.as_tensor(alpha[2], dtype=u.dtype, device=u.device) / (hz * hz)
        epsv = torch.as_tensor(eps, dtype=u.dtype, device=u.device)

        if isinstance(a_data, (float, int)):
            A_vec = torch.full(
                (1, 1, 1, C), float(a_data), dtype=J11.dtype, device=J11.device
            )
        else:
            A_vec = torch.as_tensor(a_data, dtype=J11.dtype, device=J11.device).view(
                1, 1, 1, -1
            )

        W = _as_W_like(weight_level, J11)

        psi_data = torch.ones_like(J11)
        psi_smooth = torch.ones_like(u)
        denom_u_data = torch.zeros_like(u)
        denom_v_data = torch.zeros_like(u)
        denom_w_data = torch.zeros_like(u)
        S = W * psi_data

        cK = slice(1, P - 1)
        cJ = slice(1, M - 1)
        cI = slice(1, N - 1)
        kk = torch.arange(1, P - 1, device=u.device).view(-1, 1, 1)
        jj = torch.arange(1, M - 1, device=u.device).view(1, -1, 1)
        ii = torch.arange(1, N - 1, device=u.device).view(1, 1, -1)
        Rmask = (kk + jj + ii) % 2 == 0
        Bmask = ~Rmask
        core = (cK, cJ, cI)

        if update_lag < 1:
            update_lag = 1

        def refresh_boundaries():
            _set_boundary3d_(du)
            _set_boundary3d_(dv)
            _set_boundary3d_(dw)

        def smooth_numerator(field, increment):
            center = field[core]
            return (
                wXm * (field[cK, cJ, 0 : N - 2] + increment[cK, cJ, 0 : N - 2] - center)
                + wXp * (field[cK, cJ, 2:N] + increment[cK, cJ, 2:N] - center)
                + wYm
                * (field[cK, 0 : M - 2, cI] + increment[cK, 0 : M - 2, cI] - center)
                + wYp * (field[cK, 2:M, cI] + increment[cK, 2:M, cI] - center)
                + wZm
                * (field[0 : P - 2, cJ, cI] + increment[0 : P - 2, cJ, cI] - center)
                + wZp * (field[2:P, cJ, cI] + increment[2:P, cJ, cI] - center)
            )

        def sor_update(field, increment, data_numerator, denominator, mask):
            old = increment[core]
            numerator = data_numerator[core] + smooth_numerator(field, increment)
            target = torch.where(
                denominator != 0,
                numerator / denominator,
                torch.zeros_like(denominator),
            )
            proposed = (1 - omega) * old + omega * target
            increment[core] = torch.where(mask, proposed, old)

        for it in range(iterations):
            upd_tick = (it % update_lag) == 0
            if upd_tick:
                du3 = du.unsqueeze(-1)
                dv3 = dv.unsqueeze(-1)
                dw3 = dw.unsqueeze(-1)
                E = (
                    J11 * (du3 * du3)
                    + J22 * (dv3 * dv3)
                    + J33 * (dw3 * dw3)
                    + 2 * J12 * (du3 * dv3)
                    + 2 * J13 * (du3 * dw3)
                    + 2 * J23 * (dv3 * dw3)
                    + 2 * J14 * du3
                    + 2 * J24 * dv3
                    + 2 * J34 * dw3
                    + J44
                )
                E.clamp_min_(0)
                psi_data = A_vec * (E + epsv) ** (A_vec - 1)
                if a_smooth != 1:
                    uc = u + du
                    vc = v + dv
                    wc = w + dw
                    ux = (uc[cK, cJ, 2:N] - uc[cK, cJ, 0 : N - 2]) / (2 * hx)
                    uy = (uc[cK, 2:M, cI] - uc[cK, 0 : M - 2, cI]) / (2 * hy)
                    uz = (uc[2:P, cJ, cI] - uc[0 : P - 2, cJ, cI]) / (2 * hz)
                    vx = (vc[cK, cJ, 2:N] - vc[cK, cJ, 0 : N - 2]) / (2 * hx)
                    vy = (vc[cK, 2:M, cI] - vc[cK, 0 : M - 2, cI]) / (2 * hy)
                    vz = (vc[2:P, cJ, cI] - vc[0 : P - 2, cJ, cI]) / (2 * hz)
                    wx = (wc[cK, cJ, 2:N] - wc[cK, cJ, 0 : N - 2]) / (2 * hx)
                    wy = (wc[cK, 2:M, cI] - wc[cK, 0 : M - 2, cI]) / (2 * hy)
                    wz = (wc[2:P, cJ, cI] - wc[0 : P - 2, cJ, cI]) / (2 * hz)
                    mag = (
                        ux * ux
                        + uy * uy
                        + uz * uz
                        + vx * vx
                        + vy * vy
                        + vz * vz
                        + wx * wx
                        + wy * wy
                        + wz * wz
                    )
                    psi_smooth.zero_()
                    a_s = torch.as_tensor(a_smooth, dtype=u.dtype, device=u.device)
                    psi_smooth[cK, cJ, cI] = a_s * (mag + epsv) ** (a_s - 1)
                    _set_boundary3d_(psi_smooth)
                else:
                    psi_smooth.fill_(1)

                S = W * psi_data
                denom_u_data = (S * J11).sum(3)
                denom_v_data = (S * J22).sum(3)
                denom_w_data = (S * J33).sum(3)

            refresh_boundaries()

            if a_smooth != 1:
                psiC = psi_smooth[cK, cJ, cI]
                wXm = 0.5 * (psiC + psi_smooth[cK, cJ, 0 : N - 2]) * ax
                wXp = 0.5 * (psiC + psi_smooth[cK, cJ, 2:N]) * ax
                wYm = 0.5 * (psiC + psi_smooth[cK, 0 : M - 2, cI]) * ay
                wYp = 0.5 * (psiC + psi_smooth[cK, 2:M, cI]) * ay
                wZm = 0.5 * (psiC + psi_smooth[0 : P - 2, cJ, cI]) * az
                wZp = 0.5 * (psiC + psi_smooth[2:P, cJ, cI]) * az
            else:
                shape = (P - 2, M - 2, N - 2)
                wXm = ax.expand(shape)
                wXp = ax.expand(shape)
                wYm = ay.expand(shape)
                wYp = ay.expand(shape)
                wZm = az.expand(shape)
                wZp = az.expand(shape)

            smooth_denominator = wXm + wXp + wYm + wYp + wZm + wZp
            den_u = denom_u_data[core] + smooth_denominator
            den_v = denom_v_data[core] + smooth_denominator
            den_w = denom_w_data[core] + smooth_denominator

            for color_mask in (Rmask, Bmask):
                nu_data = -(
                    S * (J14 + J12 * dv.unsqueeze(-1) + J13 * dw.unsqueeze(-1))
                ).sum(3)
                sor_update(u, du, nu_data, den_u, color_mask)

                nv_data = -(
                    S * (J24 + J12 * du.unsqueeze(-1) + J23 * dw.unsqueeze(-1))
                ).sum(3)
                sor_update(v, dv, nv_data, den_v, color_mask)

                nw_data = -(
                    S * (J34 + J13 * du.unsqueeze(-1) + J23 * dv.unsqueeze(-1))
                ).sum(3)
                sor_update(w, dw, nw_data, den_w, color_mask)

                refresh_boundaries()

        return du, dv, dw
