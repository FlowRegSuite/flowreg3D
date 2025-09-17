import numpy as np
from skimage.registration import phase_cross_correlation
from flowreg3d.util.resize_util_3D import imresize2d_gauss_cubic
from flowreg3d.core.optical_flow_3d import imregister_wrapper

def _proj_xy(v): return v.mean(axis=0)
def _proj_xz(v): return v.mean(axis=1)

def estimate_rigid_xcorr_3d(ref_vol, mov_vol, target_hw=(256,256), target_z=None, up=10, normalization="phase", disambiguate=True, weight=None):
    if ref_vol.ndim == 4 and ref_vol.shape[3] > 1:
        if weight is not None:
            w = weight.reshape(-1).astype(np.float32)
            w /= w.sum()
            ref_vol = np.tensordot(ref_vol, w, axes=([3],[0]))
            mov_vol = np.tensordot(mov_vol, w, axes=([3],[0]))
        else:
            ref_vol = ref_vol.mean(axis=3)
            mov_vol = mov_vol.mean(axis=3)
    elif ref_vol.ndim == 4:
        ref_vol = ref_vol[...,0]; mov_vol = mov_vol[...,0]

    Z,H,W = ref_vol.shape
    Th = H if target_hw is None else min(H, int(target_hw[0]))
    Tw = W if target_hw is None else min(W, int(target_hw[1]))
    sy = H/Th; sx = W/Tw

    pxy_r = _proj_xy(ref_vol); pxy_m = _proj_xy(mov_vol)
    if (Th,Tw) != (H,W):
        pxy_r = imresize2d_gauss_cubic(pxy_r, (Th,Tw))
        pxy_m = imresize2d_gauss_cubic(pxy_m, (Th,Tw))
    
    # Pre-whiten + Hann window to avoid integer-bin locking on periodic patterns (XY plane)
    pxy_r = pxy_r.astype(np.float32, copy=False)
    pxy_m = pxy_m.astype(np.float32, copy=False)
    pxy_r = pxy_r - pxy_r.mean()
    pxy_m = pxy_m - pxy_m.mean()
    hy = np.hanning(pxy_r.shape[0]).astype(np.float32)
    hx = np.hanning(pxy_r.shape[1]).astype(np.float32)
    win_xy = hy[:, None] * hx[None, :]
    pxy_r = pxy_r * win_xy
    pxy_m = pxy_m * win_xy
    
    s_xy,_,_ = phase_cross_correlation(pxy_r, pxy_m, upsample_factor=up, normalization=normalization, disambiguate=disambiguate)
    dy = float(s_xy[0]) * sy
    dx = float(s_xy[1]) * sx

    Tz = Z if target_z is None else min(Z, int(target_z))
    sz = Z/Tz
    pxz_r = _proj_xz(ref_vol); pxz_m = _proj_xz(mov_vol)
    if Tz != Z or Tw != W:
        pxz_r = imresize2d_gauss_cubic(pxz_r, (Tz, Tw))
        pxz_m = imresize2d_gauss_cubic(pxz_m, (Tz, Tw))
    
    # Pre-whiten + Hann window to avoid integer-bin locking on periodic patterns (XZ plane)
    pxz_r = pxz_r.astype(np.float32, copy=False)
    pxz_m = pxz_m.astype(np.float32, copy=False)
    pxz_r = pxz_r - pxz_r.mean()
    pxz_m = pxz_m - pxz_m.mean()
    hz = np.hanning(pxz_r.shape[0]).astype(np.float32)
    hx = np.hanning(pxz_r.shape[1]).astype(np.float32)
    win_xz = hz[:, None] * hx[None, :]
    pxz_r = pxz_r * win_xz
    pxz_m = pxz_m * win_xz
    
    s_xz,_,_ = phase_cross_correlation(pxz_r, pxz_m, upsample_factor=up, normalization=normalization, disambiguate=disambiguate)
    dz = float(s_xz[0]) * sz

    return -np.array([dx, dy, dz], dtype=np.float32)

if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import shift as ndi_shift
    from scipy.ndimage import rotate
    import time
    from flowreg3d.core.optical_flow_3d import imregister_wrapper
    
    print("Testing rigid cross-correlation alignment...")
    
    # Test 1: Pure translation
    print("\n=== Test 1: Pure Translation ===")
    # Create reference volume
    ref = np.random.rand(40, 128, 128).astype(np.float32)
    z, y, x = np.ogrid[:40, :128, :128]
    ref += 10 * np.exp(-((z-20)**2 + (y-64)**2 + (x-64)**2) / 200)
    
    # True displacement to apply to mov to align with ref
    true_dx_dy_dz = np.array([3.2, -1.5, 2.0], dtype=np.float32)
    
    # Create moved volume by shifting ref
    # We want to create a mov that is displaced from ref
    # Since we're using backward warp convention, negate the shift
    mov = ndi_shift(ref, shift=(true_dx_dy_dz[2], true_dx_dy_dz[1], true_dx_dy_dz[0]), order=1, mode='nearest')
    mov += 0.1 * np.random.randn(*ref.shape)  # Add noise
    
    # Estimate should return the displacement to apply to mov to align with ref
    t0 = time.time()
    est = estimate_rigid_xcorr_3d(ref, mov, target_hw=(128,128), up=20)
    t1 = time.time()
    error = np.abs(est - true_dx_dy_dz)
    
    print(f"True shift [dx,dy,dz]: {true_dx_dy_dz}")
    print(f"Estimated  [dx,dy,dz]: {est}")
    print(f"Error:                 {error}")
    print(f"Max error:             {error.max():.3f}")
    print(f"Estimation time:       {(t1-t0)*1000:.2f} ms")
    
    # Verify the shift works
    t0 = time.time()
    aligned = imregister_wrapper(mov, est[0], est[1], est[2], ref, interpolation_method='linear')
    t1 = time.time()
    alignment_error = np.mean(np.abs(aligned - ref))
    print(f"Alignment error:       {alignment_error:.6f}")
    print(f"Alignment time:        {(t1-t0)*1000:.2f} ms")
    
    assert np.allclose(est, true_dx_dy_dz, atol=0.3), f"Error too large: {error}"
    print("Test 1 passed!")
    
    # Test 2: Translation with rotation
    print("\n=== Test 2: Translation + Rotation ===")
    # Create reference volume with more structured features
    ref2 = np.zeros((50, 128, 128), dtype=np.float32)
    z, y, x = np.ogrid[:50, :128, :128]
    # Add multiple blobs for better rotation detection
    ref2 += 10 * np.exp(-((z-25)**2 + (y-64)**2 + (x-64)**2) / 150)
    ref2 += 8 * np.exp(-((z-15)**2 + (y-40)**2 + (x-90)**2) / 100)
    ref2 += 8 * np.exp(-((z-35)**2 + (y-90)**2 + (x-40)**2) / 100)
    ref2 += 0.1 * np.random.randn(*ref2.shape)
    
    # Apply rotation angles (in degrees)
    angle_z = 5.0  # rotation around Z axis (in XY plane)
    angle_y = 3.0  # rotation around Y axis (in XZ plane)  
    angle_x = 2.0  # rotation around X axis (in YZ plane)
    
    print(f"Applied rotations: X={angle_x}°, Y={angle_y}°, Z={angle_z}°")
    
    # Apply rotations sequentially
    mov2 = rotate(ref2, angle=angle_z, axes=(1, 2), reshape=False, order=1, mode='nearest')
    mov2 = rotate(mov2, angle=angle_y, axes=(0, 2), reshape=False, order=1, mode='nearest')
    mov2 = rotate(mov2, angle=angle_x, axes=(0, 1), reshape=False, order=1, mode='nearest')
    
    # Then apply translation (negative for backward warp convention)
    true_shift2 = np.array([2.5, -1.8, 1.2], dtype=np.float32)
    mov2 = ndi_shift(mov2, shift=(-true_shift2[2], -true_shift2[1], -true_shift2[0]), order=1, mode='nearest')
    mov2 += 0.1 * np.random.randn(*ref2.shape)
    
    # Estimate translation (rotation will remain as residual)
    t0 = time.time()
    est2 = estimate_rigid_xcorr_3d(ref2, mov2, target_hw=(128,128), up=10)
    t1 = time.time()
    error2 = np.abs(est2 - true_shift2)
    
    print(f"True shift [dx,dy,dz]: {true_shift2}")
    print(f"Estimated  [dx,dy,dz]: {est2}")
    print(f"Error:                 {error2}")
    print(f"Max error:             {error2.max():.3f}")
    print(f"Estimation time:       {(t1-t0)*1000:.2f} ms")
    
    # Align using estimated shift
    t0 = time.time()
    aligned2 = imregister_wrapper(mov2, est2[0], est2[1], est2[2], ref2, interpolation_method='linear')
    t1 = time.time()
    
    # Compute alignment errors
    alignment_error2 = np.mean(np.abs(aligned2 - ref2))
    print(f"Alignment error:       {alignment_error2:.6f}")
    print(f"Alignment time:        {(t1-t0)*1000:.2f} ms")
    
    # Estimate residual angular error using cross-correlation of gradients
    from scipy.ndimage import sobel
    # Compute gradients for angular error estimation
    ref2_grad = np.sqrt(sobel(ref2, axis=0)**2 + sobel(ref2, axis=1)**2 + sobel(ref2, axis=2)**2)
    aligned2_grad = np.sqrt(sobel(aligned2, axis=0)**2 + sobel(aligned2, axis=1)**2 + sobel(aligned2, axis=2)**2)
    
    # Normalized cross-correlation as angular similarity measure
    mask = (ref2_grad > 0.5) & (aligned2_grad > 0.5)  # Only consider regions with significant gradients
    if mask.sum() > 0:
        cos_angle = np.sum(ref2_grad[mask] * aligned2_grad[mask]) / (
            np.sqrt(np.sum(ref2_grad[mask]**2)) * np.sqrt(np.sum(aligned2_grad[mask]**2))
        )
        angular_error_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        print(f"Estimated angular error: ~{angular_error_deg:.2f}° (from gradient correlation)")
        print(f"Expected residual:      ~{np.sqrt(angle_x**2 + angle_y**2 + angle_z**2):.2f}° (total rotation)")
    
    print("Note: Higher alignment error expected due to uncompensated rotation")
    
    # More lenient threshold due to rotation
    assert error2.max() < 2.0, f"Translation error too large despite rotation: {error2}"
    print("Test 2 passed!")
