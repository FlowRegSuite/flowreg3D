import numpy as np
from scipy.ndimage import median_filter, map_coordinates
from skimage.transform import resize

from flowreg3d.core.level_solver_3d import compute_flow_3d
from flowreg3d.util.resize_util_3D import imresize_fused_gauss_cubic3D


def matlab_gradient(f, spacing):
    """Match MATLAB's gradient exactly"""
    grad = np.zeros_like(f)
    # Interior: central differences
    grad[1:-1] = (f[2:] - f[:-2]) / (2 * spacing)
    # Boundaries: one-sided (MATLAB style)
    grad[0] = (f[1] - f[0]) / spacing
    grad[-1] = (f[-1] - f[-2]) / spacing
    return grad


resize = imresize_fused_gauss_cubic3D


def imregister_wrapper(f2_level, u, v, w, f1_level, interpolation_method='cubic'):
    if f2_level.ndim == 3:
        f2_level = f2_level[:, :, :, None]
        f1_level = f1_level[:, :, :, None]
    Z, H, W, C = f2_level.shape
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(Z), np.arange(H), np.arange(W), indexing='ij')
    
    # Apply displacement fields
    map_x = (grid_x + u).astype(np.float32)
    map_y = (grid_y + v).astype(np.float32) 
    map_z = (grid_z + w).astype(np.float32)
    
    # Track out of bounds
    out_of_bounds = (map_x < 0) | (map_x >= W) | (map_y < 0) | (map_y >= H) | (map_z < 0) | (map_z >= Z)
    
    # Clip coordinates for interpolation
    map_x_clipped = np.clip(map_x, 0, W - 1)
    map_y_clipped = np.clip(map_y, 0, H - 1)
    map_z_clipped = np.clip(map_z, 0, Z - 1)
    
    # Determine interpolation order
    if interpolation_method.lower() == 'cubic':
        order = 3
    elif interpolation_method.lower() == 'linear':
        order = 1
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubic'.")
    
    warped = np.empty_like(f2_level, dtype=np.float32)
    for c in range(C):
        warped[:, :, :, c] = map_coordinates(f2_level[:, :, :, c], 
                                            [map_z_clipped, map_y_clipped, map_x_clipped],
                                            order=order, mode='nearest')
    
    # Apply boundary condition - use f1_level for out of bounds
    for c in range(C):
        warped[:, :, :, c][out_of_bounds] = f1_level[:, :, :, c][out_of_bounds]
    
    if warped.shape[3] == 1:
        warped = warped[:, :, :, 0]
    return warped


def warpingDepth(eta, levels, p, m, n):
    min_dim = min(p, m, n)
    warpingdepth = 0
    for _ in range(levels):
        warpingdepth += 1
        min_dim *= eta
        if round(min_dim) < 10:
            break
    return warpingdepth


def add_boundary(f):
    return np.pad(f, 1, mode='edge')


def get_motion_tensor_gc(f1, f2, hz, hy, hx):
    f1p = np.pad(f1, ((1,1),(1,1),(1,1)), mode='symmetric')
    f2p = np.pad(f2, ((1,1),(1,1),(1,1)), mode='symmetric')
    gz1, gy1, gx1 = np.gradient(f1p, hz, hy, hx)
    gz2, gy2, gx2 = np.gradient(f2p, hz, hy, hx)
    fx = 0.5*(gx1+gx2)
    fy = 0.5*(gy1+gy2)
    fz = 0.5*(gz1+gz2)
    ft = f2p - f1p
    fx = np.pad(fx[1:-1,1:-1,1:-1], 1, mode='symmetric')
    fy = np.pad(fy[1:-1,1:-1,1:-1], 1, mode='symmetric')
    fz = np.pad(fz[1:-1,1:-1,1:-1], 1, mode='symmetric')
    ft = np.pad(ft[1:-1,1:-1,1:-1], 1, mode='symmetric')
    
    dfx = np.gradient(fx, hz, hy, hx)
    dfy = np.gradient(fy, hz, hy, hx)
    dfz = np.gradient(fz, hz, hy, hx)
    dft = np.gradient(ft, hz, hy, hx)
    fxy = dfx[1]
    fxz = dfx[0]
    fyz = dfy[0]
    fzt, fyt, fxt = dft
    
    def gradient3(f, hz_, hy_, hx_):
        fxx = np.zeros_like(f)
        fyy = np.zeros_like(f)
        fzz = np.zeros_like(f)
        fxx[:,:,1:-1] = (f[:,:,0:-2] - 2*f[:,:,1:-1] + f[:,:,2:])/(hx_**2)
        fyy[:,1:-1,:] = (f[:,0:-2,:] - 2*f[:,1:-1,:] + f[:,2:,:])/(hy_**2)
        fzz[1:-1,:,:] = (f[0:-2,:,:] - 2*f[1:-1,:,:] + f[2:,:,:])/(hz_**2)
        return fxx, fyy, fzz
    
    fxx1, fyy1, fzz1 = gradient3(f1p, hz, hy, hx)
    fxx2, fyy2, fzz2 = gradient3(f2p, hz, hy, hx)
    fxx = 0.5*(fxx1+fxx2)
    fyy = 0.5*(fyy1+fyy2)
    fzz = 0.5*(fzz1+fzz2)
    
    reg_x = 1.0/((np.sqrt(fxx**2 + fxy**2 + fxz**2)**2) + 1e-6)
    reg_y = 1.0/((np.sqrt(fxy**2 + fyy**2 + fyz**2)**2) + 1e-6)
    reg_z = 1.0/((np.sqrt(fxz**2 + fyz**2 + fzz**2)**2) + 1e-6)
    
    J11 = reg_x*fxx**2 + reg_y*fxy**2 + reg_z*fxz**2
    J22 = reg_x*fxy**2 + reg_y*fyy**2 + reg_z*fyz**2
    J33 = reg_x*fxz**2 + reg_y*fyz**2 + reg_z*fzz**2
    J12 = reg_x*fxx*fxy + reg_y*fxy*fyy + reg_z*fxz*fyz
    J13 = reg_x*fxx*fxz + reg_y*fxy*fyz + reg_z*fxz*fzz
    J23 = reg_x*fxy*fxz + reg_y*fyy*fyz + reg_z*fyz*fzz
    J14 = reg_x*fxx*fxt + reg_y*fxy*fyt + reg_z*fxz*fzt
    J24 = reg_x*fxy*fxt + reg_y*fyy*fyt + reg_z*fyz*fzt
    J34 = reg_x*fxz*fxt + reg_y*fyz*fyt + reg_z*fzz*fzt
    J44 = reg_x*fxt**2 + reg_y*fyt**2 + reg_z*fzt**2
    
    for arr in [J11,J22,J33,J44,J12,J13,J23,J14,J24,J34]:
        arr[:,:,0]=0
        arr[:,:,-1]=0
        arr[:,0,:]=0
        arr[:,-1,:]=0
        arr[0,:,:]=0
        arr[-1,:,:]=0
    return J11,J22,J33,J44,J12,J13,J23,J14,J24,J34


def level_solver(J11, J22, J33, J44, J12, J13, J23, J14, J24, J34, 
                weight, u, v, w, alpha, iterations, update_lag, verbose, 
                a_data, a_smooth, hx, hy, hz):
    result = compute_flow_3d(J11, J22, J33, J44, J12, J13, J23, J14, J24, J34,
                            weight=weight, u=u, v=v, w=w, 
                            alpha_x=alpha[0], alpha_y=alpha[1], alpha_z=alpha[2],
                            iterations=iterations, update_lag=update_lag, 
                            a_data=a_data, a_smooth=a_smooth, 
                            hx=hx, hy=hy, hz=hz)
    du = result[:, :, :, 0]
    dv = result[:, :, :, 1]
    dw = result[:, :, :, 2]
    return du, dv, dw


def get_displacement(fixed, moving, alpha=(2, 2, 2), update_lag=10, iterations=20, min_level=0, levels=50, eta=0.8,
                     a_smooth=0.5, a_data=0.45, const_assumption='gc', uvw=None, weight=None):
    fixed = fixed.astype(np.float64)
    moving = moving.astype(np.float64)
    if fixed.ndim == 4:
        p, m, n, n_channels = fixed.shape
    else:
        p, m, n = fixed.shape
        n_channels = 1
        fixed = fixed[:, :, :, np.newaxis]
        moving = moving[:, :, :, np.newaxis]
    if uvw is not None:
        u_init = uvw[:, :, :, 0]
        v_init = uvw[:, :, :, 1]
        w_init = uvw[:, :, :, 2]
    else:
        u_init = np.zeros((p, m, n), dtype=np.float64)
        v_init = np.zeros((p, m, n), dtype=np.float64)
        w_init = np.zeros((p, m, n), dtype=np.float64)
    if weight is None:
        weight = np.ones((p, m, n, n_channels), dtype=np.float64) / n_channels
    else:
        weight = weight.astype(np.float64)
        if weight.ndim < 4:
            # Handle 1D weight array
            if weight.ndim == 1:
                # If weight has fewer elements than channels, pad with 1/n_channels
                if len(weight) < n_channels:
                    # Use default value for missing channels (MATLAB behavior)
                    default_weight = 1.0 / n_channels
                    weight_expanded = np.full(n_channels, default_weight, dtype=np.float64)
                    weight_expanded[:len(weight)] = weight
                    weight = weight_expanded
                elif len(weight) > n_channels:
                    # Truncate if more weights than channels
                    weight = weight[:n_channels]
                # Normalize weights to sum to 1
                weight = weight / weight.sum()
                # Broadcast to spatial dimensions
                weight = np.ones((p, m, n, n_channels), dtype=np.float64) * weight.reshape(1, 1, 1, -1)
            else:
                # 3D spatial weight - broadcast to all channels
                weight = np.ones((p, m, n, n_channels), dtype=np.float64) * weight[..., np.newaxis]
    if not isinstance(a_data, np.ndarray):
        a_data_arr = np.full(n_channels, a_data, dtype=np.float64)
    else:
        a_data_arr = a_data
    a_data_arr = np.ascontiguousarray(a_data_arr)
    f1_low = fixed
    f2_low = moving
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
    u = None
    v = None
    w = None
    for i in range(max(max_level_x, max_level_y, max_level_z), min_level - 1, -1):
        level_size = (int(round(p * eta ** (min(i, max_level_z)))),
                     int(round(m * eta ** (min(i, max_level_y)))), 
                     int(round(n * eta ** (min(i, max_level_x)))))
        f1_level = resize(f1_low, level_size)
        f2_level = resize(f2_low, level_size)
        if f1_level.ndim == 3:
            f1_level = f1_level[:, :, :, np.newaxis]
            f2_level = f2_level[:, :, :, np.newaxis]
        current_hz = float(p) / f1_level.shape[0]
        current_hy = float(m) / f1_level.shape[1]
        current_hx = float(n) / f1_level.shape[2]
        if i == max(max_level_x, max_level_y, max_level_z):
            u = add_boundary(resize(u_init, level_size))
            v = add_boundary(resize(v_init, level_size))
            w = add_boundary(resize(w_init, level_size))
            tmp = f2_level.copy()
        else:
            # Get previous and new sizes for scaling
            sz_prev = (u.shape[0]-2, u.shape[1]-2, u.shape[2]-2)
            sz_new = level_size
            sx = sz_new[2] / sz_prev[2]
            sy = sz_new[1] / sz_prev[1]
            sz = sz_new[0] / sz_prev[0]
            # Resize and scale the displacement fields to new level
            u = add_boundary(resize(u[1:-1, 1:-1, 1:-1], level_size) * sx)
            v = add_boundary(resize(v[1:-1, 1:-1, 1:-1], level_size) * sy)
            w = add_boundary(resize(w[1:-1, 1:-1, 1:-1], level_size) * sz)
            # Apply warping without division (displacements are in level units)
            tmp = imregister_wrapper(f2_level, 
                                   u[1:-1, 1:-1, 1:-1],
                                   v[1:-1, 1:-1, 1:-1],
                                   w[1:-1, 1:-1, 1:-1],
                                   f1_level)
        if tmp.ndim == 3:
            tmp = tmp[:, :, :, np.newaxis]
        u = np.ascontiguousarray(u)
        v = np.ascontiguousarray(v)
        w = np.ascontiguousarray(w)
        J_size = (f1_level.shape[0] + 2, f1_level.shape[1] + 2, f1_level.shape[2] + 2, n_channels)
        J11 = np.zeros(J_size, dtype=np.float64)
        J22 = np.zeros(J_size, dtype=np.float64)
        J33 = np.zeros(J_size, dtype=np.float64)
        J44 = np.zeros(J_size, dtype=np.float64)
        J12 = np.zeros(J_size, dtype=np.float64)
        J13 = np.zeros(J_size, dtype=np.float64)
        J23 = np.zeros(J_size, dtype=np.float64)
        J14 = np.zeros(J_size, dtype=np.float64)
        J24 = np.zeros(J_size, dtype=np.float64)
        J34 = np.zeros(J_size, dtype=np.float64)
        for ch in range(n_channels):
            J_ch = get_motion_tensor_gc(f1_level[:, :, :, ch], tmp[:, :, :, ch],
                                       current_hz, current_hy, current_hx)
            J11[:, :, :, ch] = J_ch[0]
            J22[:, :, :, ch] = J_ch[1]
            J33[:, :, :, ch] = J_ch[2]
            J44[:, :, :, ch] = J_ch[3]
            J12[:, :, :, ch] = J_ch[4]
            J13[:, :, :, ch] = J_ch[5]
            J23[:, :, :, ch] = J_ch[6]
            J14[:, :, :, ch] = J_ch[7]
            J24[:, :, :, ch] = J_ch[8]
            J34[:, :, :, ch] = J_ch[9]

        weight_level = resize(weight, f1_level.shape[:3])
        weight_level = np.pad(weight_level, ((1,1),(1,1),(1,1),(0,0)), mode='constant', constant_values=0.0)
        if weight_level.ndim < 4:
            weight_level = weight_level[:, :, :, np.newaxis]

        if i == min_level:
            alpha_scaling = 1
        else:
            alpha_scaling = eta ** (-0.5 * i)

        alpha_tmp = [alpha_scaling * alpha[j] for j in range(len(alpha))]

        du, dv, dw = level_solver(np.ascontiguousarray(J11), np.ascontiguousarray(J22), 
                                 np.ascontiguousarray(J33), np.ascontiguousarray(J44),
                                 np.ascontiguousarray(J12), np.ascontiguousarray(J13), 
                                 np.ascontiguousarray(J23), np.ascontiguousarray(J14),
                                 np.ascontiguousarray(J24), np.ascontiguousarray(J34),
                                 np.ascontiguousarray(weight_level), 
                                 u, v, w, alpha_tmp, iterations, update_lag, 0,
                                 a_data_arr, a_smooth, current_hx, current_hy, current_hz)
        if min(level_size) > 5:
            du[1:-1, 1:-1, 1:-1] = median_filter(du[1:-1, 1:-1, 1:-1], size=(5, 5, 5), mode='mirror')
            dv[1:-1, 1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1, 1:-1], size=(5, 5, 5), mode='mirror')
            dw[1:-1, 1:-1, 1:-1] = median_filter(dw[1:-1, 1:-1, 1:-1], size=(5, 5, 5), mode='mirror')
        u = u + du
        v = v + dv
        w = w + dw
    flow = np.zeros((u.shape[0] - 2, u.shape[1] - 2, u.shape[2] - 2, 3), dtype=np.float64)
    flow[:, :, :, 0] = u[1:-1, 1:-1, 1:-1]
    flow[:, :, :, 1] = v[1:-1, 1:-1, 1:-1]
    flow[:, :, :, 2] = w[1:-1, 1:-1, 1:-1]
    if min_level > 0:
        # Resize to original dimensions
        flow_resized = np.zeros((p, m, n, 3), dtype=np.float64)
        for i in range(3):
            flow_resized[:, :, :, i] = resize(flow[:, :, :, i], (p, m, n), order=3)
        flow = flow_resized
    return flow
