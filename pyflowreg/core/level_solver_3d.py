from linecache import cache

import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def set_boundary_2d(f):
    m, n = f.shape
    for i in range(n):
        f[0, i] = f[1, i]
        f[m-1, i] = f[m-2, i]
    for j in range(m):
        f[j, 0] = f[j, 1]
        f[j, n-1] = f[j, n-2]


@njit(fastmath=True, cache=True)
def nonlinearity_smoothness_2d(psi_smooth, u, du, v, dv, m, n, a, hx, hy):
    eps = 0.00001
    u_full = u + du
    v_full = v + dv
    ux = np.zeros((m,n))
    uy = np.zeros((m,n))
    vx = np.zeros((m,n))
    vy = np.zeros((m,n))

    for i in range(n):
        for j in range(m):
            # ux
            if n > 1:
                if i == 0:
                    ux[j,i] = (u_full[j,i+1]-u_full[j,i])/hx
                elif i == n-1:
                    ux[j,i] = (u_full[j,i]-u_full[j,i-1])/hx
                else:
                    ux[j,i] = (u_full[j,i+1]-u_full[j,i-1])/(2.0*hx)
            # vx
            if n > 1:
                if i == 0:
                    vx[j,i] = (v_full[j,i+1]-v_full[j,i])/hx
                elif i == n-1:
                    vx[j,i] = (v_full[j,i]-v_full[j,i-1])/hx
                else:
                    vx[j,i] = (v_full[j,i+1]-v_full[j,i-1])/(2.0*hx)
            # uy
            if m > 1:
                if j == 0:
                    uy[j,i] = (u_full[j+1,i]-u_full[j,i])/hy
                elif j == m-1:
                    uy[j,i] = (u_full[j,i]-u_full[j-1,i])/hy
                else:
                    uy[j,i] = (u_full[j+1,i]-u_full[j-1,i])/(2.0*hy)
            # vy
            if m > 1:
                if j == 0:
                    vy[j,i] = (v_full[j+1,i]-v_full[j,i])/hy
                elif j == m-1:
                    vy[j,i] = (v_full[j,i]-v_full[j-1,i])/hy
                else:
                    vy[j,i] = (v_full[j+1,i]-v_full[j-1,i])/(2.0*hy)

    for i in range(n):
        for j in range(m):
            tmp = ux[j,i]*ux[j,i] + uy[j,i]*uy[j,i] + vx[j,i]*vx[j,i] + vy[j,i]*vy[j,i]
            if tmp < 0.0:
                tmp = 0.0
            psi_smooth[j,i] = a * (tmp+eps)**(a-1.0)

@njit(fastmath=True, cache=True)
def compute_flow(
    J11, J22, J33, J12, J13, J23,
    weight, u, v,
    alpha_x, alpha_y,
    iterations, update_lag,
    a_data, a_smooth, hx, hy
):
    m, n, n_channels = J11.shape
    du = np.zeros((m,n))
    dv = np.zeros((m,n))
    psi = np.ones((m,n,n_channels))
    psi_smooth = np.ones((m,n))

    OMEGA = 1.95
    alpha = np.array([alpha_x, alpha_y], dtype=np.float64)

    for iteration_counter in range(iterations):
        if (iteration_counter+1) % update_lag == 0:
            # Update psi (non-linearities for data term)
            for k in range(n_channels):
                for i in range(n):
                    for j in range(m):
                        val = (J11[j,i,k]*du[j,i]*du[j,i] +
                               J22[j,i,k]*dv[j,i]*dv[j,i] +
                               J23[j,i,k]*dv[j,i] +
                               2.0*J12[j,i,k]*du[j,i]*dv[j,i] +
                               2.0*J13[j,i,k]*du[j,i] +
                               J23[j,i,k]*dv[j,i] +
                               J33[j,i,k])
                        if val < 0.0:
                            val = 0.0
                        psi[j,i,k] = a_data[k]*(val+0.00001)**(a_data[k]-1.0)

            if a_smooth != 1.0:
                nonlinearity_smoothness_2d(psi_smooth, u, du, v, dv, m, n, a_smooth, hx, hy)
            else:
                for i in range(n):
                    for j in range(m):
                        psi_smooth[j,i] = 1.0

        set_boundary_2d(du)
        set_boundary_2d(dv)

        for i in range(1, n-1):
            for j in range(1, m-1):
                denom_u=0.0
                denom_v=0.0
                num_u=0.0
                num_v=0.0

                # neighbors:
                # left = (j, i-1)
                # right = (j, i+1)
                # down = (j+1, i)
                # up = (j-1, i)
                left = (j, i-1)
                right = (j, i+1)
                down = (j+1, i)
                up = (j-1, i)

                if a_smooth != 1.0:
                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[left])*(alpha[0]/(hx*hx))
                    num_u += tmp*(u[left]+du[left] - u[j,i])
                    num_v += tmp*(v[left]+dv[left] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[right])*(alpha[0]/(hx*hx))
                    num_u += tmp*(u[right]+du[right] - u[j,i])
                    num_v += tmp*(v[right]+dv[right] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[down])*(alpha[1]/(hy*hy))
                    num_u += tmp*(u[down]+du[down] - u[j,i])
                    num_v += tmp*(v[down]+dv[down] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = 0.5*(psi_smooth[j,i]+psi_smooth[up])*(alpha[1]/(hy*hy))
                    num_u += tmp*(u[up]+du[up] - u[j,i])
                    num_v += tmp*(v[up]+dv[up] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp
                else:
                    tmp = alpha[0]/(hx*hx)
                    num_u += tmp*(u[left]+du[left] - u[j,i])
                    num_v += tmp*(v[left]+dv[left] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[0]/(hx*hx)
                    num_u += tmp*(u[right]+du[right] - u[j,i])
                    num_v += tmp*(v[right]+dv[right] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1]/(hy*hy)
                    num_u += tmp*(u[down]+du[down] - u[j,i])
                    num_v += tmp*(v[down]+dv[down] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                    tmp = alpha[1]/(hy*hy)
                    num_u += tmp*(u[up]+du[up] - u[j,i])
                    num_v += tmp*(v[up]+dv[up] - v[j,i])
                    denom_u += tmp
                    denom_v += tmp

                for k in range(n_channels):
                    val_u = weight[j,i,k]*psi[j,i,k]*(J13[j,i,k] + J12[j,i,k]*dv[j,i])
                    num_u -= val_u
                    denom_u += weight[j,i,k]*psi[j,i,k]*J11[j,i,k]
                    denom_v += weight[j,i,k]*psi[j,i,k]*J22[j,i,k]

                du_kp1 = num_u/denom_u if denom_u!=0.0 else 0.0
                du[j,i] = (1.0-OMEGA)*du[j,i] + OMEGA*du_kp1

                num_v2 = num_v
                for k in range(n_channels):
                    num_v2 -= weight[j,i,k]*psi[j,i,k]*(J23[j,i,k] + J12[j,i,k]*du[j,i])

                dv_kp1 = num_v2/denom_v if denom_v!=0.0 else 0.0
                dv[j,i] = (1.0-OMEGA)*dv[j,i] + OMEGA*dv_kp1

    flow = np.zeros((m,n,2))
    flow[:,:,0] = du
    flow[:,:,1] = dv
    return flow


@njit(fastmath=True, cache=True)
def set_boundary_3d(f):
    p, m, n = f.shape
    for k in range(p):
        for i in range(n):
            f[k,0,i] = f[k,1,i]
            f[k,m-1,i] = f[k,m-2,i]
        for j in range(m):
            f[k,j,0] = f[k,j,1]
            f[k,j,n-1] = f[k,j,n-2]
    for j in range(m):
        for i in range(n):
            f[0,j,i] = f[1,j,i]
            f[p-1,j,i] = f[p-2,j,i]

@njit(fastmath=True, cache=True)
def nonlinearity_smoothness_3d(psi, u,du, v,dv, w,dw, p,m,n, a, hx,hy,hz):
    eps = 1e-5
    uu=u+du; vv=v+dv; ww=w+dw
    ux=np.zeros((p,m,n)); uy=np.zeros((p,m,n)); uz=np.zeros((p,m,n))
    vx=np.zeros((p,m,n)); vy=np.zeros((p,m,n)); vz=np.zeros((p,m,n))
    wx=np.zeros((p,m,n)); wy=np.zeros((p,m,n)); wz=np.zeros((p,m,n))
    for k in range(p):
        for j in range(m):
            for i in range(n):
                ixm=i-1 if i>0 else 0; ixp=i+1 if i<n-1 else n-1
                jym=j-1 if j>0 else 0; jyp=j+1 if j<m-1 else m-1
                kzm=k-1 if k>0 else 0; kzp=k+1 if k<p-1 else p-1
                ux[k,j,i]=(uu[k,j,ixp]-uu[k,j,ixm])/(2*hx)
                uy[k,j,i]=(uu[k,jyp,i]-uu[k,jym,i])/(2*hy)
                uz[k,j,i]=(uu[kzp,j,i]-uu[kzm,j,i])/(2*hz)
                vx[k,j,i]=(vv[k,j,ixp]-vv[k,j,ixm])/(2*hx)
                vy[k,j,i]=(vv[k,jyp,i]-vv[k,jym,i])/(2*hy)
                vz[k,j,i]=(vv[kzp,j,i]-vv[kzm,j,i])/(2*hz)
                wx[k,j,i]=(ww[k,j,ixp]-ww[k,j,ixm])/(2*hx)
                wy[k,j,i]=(ww[k,jyp,i]-ww[k,jym,i])/(2*hy)
                wz[k,j,i]=(ww[kzp,j,i]-ww[kzm,j,i])/(2*hz)
    for k in range(p):
        for j in range(m):
            for i in range(n):
                g = ux[k,j,i]*ux[k,j,i]+uy[k,j,i]*uy[k,j,i]+uz[k,j,i]*uz[k,j,i] \
                  + vx[k,j,i]*vx[k,j,i]+vy[k,j,i]*vy[k,j,i]+vz[k,j,i]*vz[k,j,i] \
                  + wx[k,j,i]*wx[k,j,i]+wy[k,j,i]*wy[k,j,i]+wz[k,j,i]*wz[k,j,i]
                if g<0.0: g=0.0
                psi[k,j,i] = a * (g+eps)**(a-1.0)

@njit(fastmath=True, cache=True)
def compute_flow_3d(
    J11,J22,J33,J44,J12,J13,J23,J14,J24,J34,
    weight, u, v, w,
    alpha_x, alpha_y, alpha_z,
    iterations, update_lag,
    a_data, a_smooth, hx, hy, hz
):
    p, m, n, C = J11.shape
    du = np.zeros((p,m,n))
    dv = np.zeros((p,m,n))
    dw = np.zeros((p,m,n))
    psi_smooth = np.ones((p,m,n))
    psi = np.ones((p,m,n,C))
    alpha = (alpha_x, alpha_y, alpha_z)
    OMEGA = 1.6
    eps = 1e-6

    for it in range(iterations):
        if a_smooth != 1.0:
            nonlinearity_smoothness_3d(psi_smooth, u,du, v,dv, w,dw, p,m,n, a_smooth, hx,hy,hz)
        if a_data != 1.0 and (it % update_lag == 0):
            for c in range(C):
                for k in range(p):
                    for j in range(m):
                        for i in range(n):
                            val = J44[k,j,i,c]
                            if val<0.0: val=0.0
                            psi[k,j,i,c] = a_data * (val+eps)**(a_data-1.0)

        set_boundary_3d(du)
        set_boundary_3d(dv)
        set_boundary_3d(dw)

        for k in range(1,p-1):
            for j in range(1,m-1):
                for i in range(1,n-1):
                    denom_u=0.0; denom_v=0.0; denom_w=0.0
                    num_u=0.0; num_v=0.0; num_w=0.0

                    km = (k-1,j,i); kp = (k+1,j,i)
                    jm = (k,j-1,i); jp = (k,j+1,i)
                    im = (k,j,i-1); ip = (k,j,i+1)

                    if a_smooth != 1.0:
                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[km])*(alpha[2]/(hz*hz))
                        num_u += tmp*(u[km]+du[km] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[km]+dv[km] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[km]+dw[km] - w[k,j,i]); denom_w += tmp

                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[kp])*(alpha[2]/(hz*hz))
                        num_u += tmp*(u[kp]+du[kp] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[kp]+dv[kp] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[kp]+dw[kp] - w[k,j,i]); denom_w += tmp

                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[jm])*(alpha[1]/(hy*hy))
                        num_u += tmp*(u[jm]+du[jm] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[jm]+dv[jm] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[jm]+dw[jm] - w[k,j,i]); denom_w += tmp

                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[jp])*(alpha[1]/(hy*hy))
                        num_u += tmp*(u[jp]+du[jp] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[jp]+dv[jp] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[jp]+dw[jp] - w[k,j,i]); denom_w += tmp

                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[im])*(alpha[0]/(hx*hx))
                        num_u += tmp*(u[im]+du[im] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[im]+dv[im] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[im]+dw[im] - w[k,j,i]); denom_w += tmp

                        tmp = 0.5*(psi_smooth[k,j,i]+psi_smooth[ip])*(alpha[0]/(hx*hx))
                        num_u += tmp*(u[ip]+du[ip] - u[k,j,i]); denom_u += tmp
                        num_v += tmp*(v[ip]+dv[ip] - v[k,j,i]); denom_v += tmp
                        num_w += tmp*(w[ip]+dw[ip] - w[k,j,i]); denom_w += tmp
                    else:
                        ax = alpha[0]/(hx*hx); ay = alpha[1]/(hy*hy); az = alpha[2]/(hz*hz)
                        num_u += ax*(u[ip]+du[ip] + u[im]+du[im] - 2*u[k,j,i]); denom_u += 2*ax
                        num_v += ax*(v[ip]+dv[ip] + v[im]+dv[im] - 2*v[k,j,i]); denom_v += 2*ax
                        num_w += ax*(w[ip]+dw[ip] + w[im]+dw[im] - 2*w[k,j,i]); denom_w += 2*ax
                        num_u += ay*(u[jp]+du[jp] + u[jm]+du[jm] - 2*u[k,j,i]); denom_u += 2*ay
                        num_v += ay*(v[jp]+dv[jp] + v[jm]+dv[jm] - 2*v[k,j,i]); denom_v += 2*ay
                        num_w += ay*(w[jp]+dw[jp] + w[jm]+dw[jm] - 2*w[k,j,i]); denom_w += 2*ay
                        num_u += az*(u[kp]+du[kp] + u[km]+du[km] - 2*u[k,j,i]); denom_u += 2*az
                        num_v += az*(v[kp]+dv[kp] + v[km]+dv[km] - 2*v[k,j,i]); denom_v += 2*az
                        num_w += az*(w[kp]+dw[kp] + w[km]+dw[km] - 2*w[k,j,i]); denom_w += 2*az

                    for c in range(C):
                        ww = weight[k,j,i,c]
                        if a_data != 1.0: ww *= psi[k,j,i,c]
                        denom_u += ww*J11[k,j,i,c]
                        denom_v += ww*J22[k,j,i,c]
                        denom_w += ww*J33[k,j,i,c]

                    num_u2 = num_u
                    for c in range(C):
                        ww = weight[k,j,i,c]
                        if a_data != 1.0: ww *= psi[k,j,i,c]
                        num_u2 -= ww*(J14[k,j,i,c] + J12[k,j,i,c]*dv[k,j,i] + J13[k,j,i,c]*dw[k,j,i])
                    du_kp1 = num_u2/denom_u if denom_u!=0.0 else 0.0
                    du[k,j,i] = (1.0-OMEGA)*du[k,j,i] + OMEGA*du_kp1

                    num_v2 = num_v
                    for c in range(C):
                        ww = weight[k,j,i,c]
                        if a_data != 1.0: ww *= psi[k,j,i,c]
                        num_v2 -= ww*(J24[k,j,i,c] + J12[k,j,i,c]*du[k,j,i] + J23[k,j,i,c]*dw[k,j,i])
                    dv_kp1 = num_v2/denom_v if denom_v!=0.0 else 0.0
                    dv[k,j,i] = (1.0-OMEGA)*dv[k,j,i] + OMEGA*dv_kp1

                    num_w2 = num_w
                    for c in range(C):
                        ww = weight[k,j,i,c]
                        if a_data != 1.0: ww *= psi[k,j,i,c]
                        num_w2 -= ww*(J34[k,j,i,c] + J13[k,j,i,c]*du[k,j,i] + J23[k,j,i,c]*dv[k,j,i])
                    dw_kp1 = num_w2/denom_w if denom_w!=0.0 else 0.0
                    dw[k,j,i] = (1.0-OMEGA)*dw[k,j,i] + OMEGA*dw_kp1

    flow = np.zeros((p,m,n,3))
    flow[:,:,:,0] = du
    flow[:,:,:,1] = dv
    flow[:,:,:,2] = dw
    return flow
