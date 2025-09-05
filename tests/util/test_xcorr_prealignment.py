"""
Tests for cross-correlation based prealignment functionality.
Mirrors the tests from the main block of xcorr_prealignment.py
"""

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import rotate, sobel
import time
from flowreg3d.util.xcorr_prealignment import estimate_rigid_xcorr_3d, prealign_for_flow


def test_pure_translation():
    """Test 3D rigid cross-correlation alignment with pure translation."""
    print("\n=== Test: Pure Translation ===")
    
    # Create reference volume with a 3D Gaussian blob
    ref = np.random.rand(40, 128, 128).astype(np.float32)
    z, y, x = np.ogrid[:40, :128, :128]
    ref += 10 * np.exp(-((z-20)**2 + (y-64)**2 + (x-64)**2) / 200)
    
    # True displacement to apply to mov to align with ref (backward warp convention)
    true_dx_dy_dz = np.array([3.2, -1.5, 2.0], dtype=np.float32)
    
    # Create moved volume by shifting ref
    # We want to create a mov that is displaced from ref
    # Since we're using backward warp convention, negate the shift
    mov = ndi_shift(ref, shift=(-true_dx_dy_dz[2], -true_dx_dy_dz[1], -true_dx_dy_dz[0]), order=1, mode='nearest')
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
    aligned = prealign_for_flow(mov, est, order=1)
    t1 = time.time()
    alignment_error = np.mean(np.abs(aligned - ref))
    print(f"Alignment error:       {alignment_error:.6f}")
    print(f"Alignment time:        {(t1-t0)*1000:.2f} ms")
    
    assert np.allclose(est, true_dx_dy_dz, atol=0.3), f"Error too large: {error}"
    assert alignment_error < 0.5, f"Alignment error too large: {alignment_error}"
    print("✓ Pure translation test passed!")
    

def test_translation_with_rotation():
    """Test alignment with translation and rotation - rotation remains as residual."""
    print("\n=== Test: Translation + Rotation ===")
    
    # Create reference volume with more structured features
    ref = np.zeros((50, 128, 128), dtype=np.float32)
    z, y, x = np.ogrid[:50, :128, :128]
    # Add multiple blobs for better rotation detection
    ref += 10 * np.exp(-((z-25)**2 + (y-64)**2 + (x-64)**2) / 150)
    ref += 8 * np.exp(-((z-15)**2 + (y-40)**2 + (x-90)**2) / 100)
    ref += 8 * np.exp(-((z-35)**2 + (y-90)**2 + (x-40)**2) / 100)
    ref += 0.1 * np.random.randn(*ref.shape)
    
    # Apply rotation angles (in degrees)
    angle_z = 5.0  # rotation around Z axis (in XY plane)
    angle_y = 3.0  # rotation around Y axis (in XZ plane)  
    angle_x = 2.0  # rotation around X axis (in YZ plane)
    
    print(f"Applied rotations: X={angle_x}°, Y={angle_y}°, Z={angle_z}°")
    
    # Apply rotations sequentially
    mov = rotate(ref, angle=angle_z, axes=(1, 2), reshape=False, order=1, mode='nearest')
    mov = rotate(mov, angle=angle_y, axes=(0, 2), reshape=False, order=1, mode='nearest')
    mov = rotate(mov, angle=angle_x, axes=(0, 1), reshape=False, order=1, mode='nearest')
    
    # Then apply translation (negative for backward warp convention)
    true_shift = np.array([2.5, -1.8, 1.2], dtype=np.float32)
    mov = ndi_shift(mov, shift=(-true_shift[2], -true_shift[1], -true_shift[0]), order=1, mode='nearest')
    mov += 0.1 * np.random.randn(*ref.shape)
    
    # Estimate translation (rotation will remain as residual)
    t0 = time.time()
    est = estimate_rigid_xcorr_3d(ref, mov, target_hw=(128,128), up=10)
    t1 = time.time()
    error = np.abs(est - true_shift)
    
    print(f"True shift [dx,dy,dz]: {true_shift}")
    print(f"Estimated  [dx,dy,dz]: {est}")
    print(f"Error:                 {error}")
    print(f"Max error:             {error.max():.3f}")
    print(f"Estimation time:       {(t1-t0)*1000:.2f} ms")
    
    # Align using estimated shift
    t0 = time.time()
    aligned = prealign_for_flow(mov, est, order=1)
    t1 = time.time()
    
    # Compute alignment errors
    alignment_error = np.mean(np.abs(aligned - ref))
    print(f"Alignment error:       {alignment_error:.6f}")
    print(f"Alignment time:        {(t1-t0)*1000:.2f} ms")
    
    # Estimate residual angular error using cross-correlation of gradients
    ref_grad = np.sqrt(sobel(ref, axis=0)**2 + sobel(ref, axis=1)**2 + sobel(ref, axis=2)**2)
    aligned_grad = np.sqrt(sobel(aligned, axis=0)**2 + sobel(aligned, axis=1)**2 + sobel(aligned, axis=2)**2)
    
    # Normalized cross-correlation as angular similarity measure
    mask = (ref_grad > 0.5) & (aligned_grad > 0.5)  # Only consider regions with significant gradients
    if mask.sum() > 0:
        cos_angle = np.sum(ref_grad[mask] * aligned_grad[mask]) / (
            np.sqrt(np.sum(ref_grad[mask]**2)) * np.sqrt(np.sum(aligned_grad[mask]**2))
        )
        angular_error_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        print(f"Estimated angular error: ~{angular_error_deg:.2f}° (from gradient correlation)")
        print(f"Expected residual:      ~{np.sqrt(angle_x**2 + angle_y**2 + angle_z**2):.2f}° (total rotation)")
    
    print("Note: Higher alignment error expected due to uncompensated rotation")
    
    # More lenient threshold due to rotation
    assert error.max() < 2.0, f"Translation error too large despite rotation: {error}"
    print("✓ Translation+rotation test passed!")


def test_multichannel_alignment():
    """Test alignment with multichannel data and weights."""
    print("\n=== Test: Multichannel Alignment ===")
    
    # Create 2-channel reference volume
    ref = np.random.rand(30, 64, 64, 2).astype(np.float32)
    z, y, x = np.ogrid[:30, :64, :64]
    # Different features in each channel
    ref[..., 0] += 5 * np.exp(-((z-15)**2 + (y-32)**2 + (x-32)**2) / 100)
    ref[..., 1] += 5 * np.exp(-((z-15)**2 + (y-20)**2 + (x-40)**2) / 100)
    
    # True displacement (backward warp convention)
    true_dx_dy_dz = np.array([2.5, -1.0, 1.5], dtype=np.float32)
    
    # Create moved multichannel volume
    mov = np.empty_like(ref)
    for c in range(2):
        mov[..., c] = ndi_shift(ref[..., c], 
                                shift=(-true_dx_dy_dz[2], -true_dx_dy_dz[1], -true_dx_dy_dz[0]), 
                                order=1, mode='nearest')
    mov += 0.05 * np.random.randn(*ref.shape)
    
    # Test with equal weights
    weight = np.array([0.5, 0.5], dtype=np.float32)
    
    t0 = time.time()
    est = estimate_rigid_xcorr_3d(ref, mov, target_hw=(64,64), up=10, weight=weight)
    t1 = time.time()
    error = np.abs(est - true_dx_dy_dz)
    
    print(f"True shift [dx,dy,dz]: {true_dx_dy_dz}")
    print(f"Estimated  [dx,dy,dz]: {est}")
    print(f"Error:                 {error}")
    print(f"Max error:             {error.max():.3f}")
    print(f"Estimation time:       {(t1-t0)*1000:.2f} ms")
    
    # Verify alignment
    aligned = prealign_for_flow(mov, est, order=1)
    alignment_error = np.mean(np.abs(aligned - ref))
    print(f"Alignment error:       {alignment_error:.6f}")
    
    assert np.allclose(est, true_dx_dy_dz, atol=0.5), f"Error too large: {error}"
    assert alignment_error < 0.5, f"Alignment error too large: {alignment_error}"
    print("✓ Multichannel test passed!")


def test_downsampling_accuracy():
    """Test that downsampling preserves alignment accuracy with proper scaling."""
    print("\n=== Test: Downsampling Accuracy ===")
    
    # Create larger reference volume
    ref = np.random.rand(50, 256, 256).astype(np.float32) * 0.5
    z, y, x = np.ogrid[:50, :256, :256]
    ref += 10 * np.exp(-((z-25)**2 + (y-128)**2 + (x-128)**2) / 500)
    
    # True displacement (backward warp convention)
    true_dx_dy_dz = np.array([5.0, -3.0, 4.0], dtype=np.float32)
    
    # Create moved volume
    mov = ndi_shift(ref, shift=(-true_dx_dy_dz[2], -true_dx_dy_dz[1], -true_dx_dy_dz[0]), 
                    order=1, mode='nearest')
    mov += 0.1 * np.random.randn(*ref.shape)
    
    # Test with different downsampling targets
    for target_size in [256, 128, 64]:
        t0 = time.time()
        est = estimate_rigid_xcorr_3d(ref, mov, target_hw=(target_size, target_size), up=10)
        t1 = time.time()
        error = np.abs(est - true_dx_dy_dz)
        
        print(f"Target size {target_size}: error = {error}, max = {error.max():.3f}, time = {(t1-t0)*1000:.1f}ms")
        
        # More lenient for smaller sizes due to downsampling
        tolerance = 0.5 if target_size >= 128 else 1.0
        assert np.allclose(est, true_dx_dy_dz, atol=tolerance), \
            f"Error too large for size {target_size}: {error}"
    
    print("✓ Downsampling test passed!")


def test_subpixel_accuracy():
    """Test subpixel accuracy with different upsampling factors."""
    print("\n=== Test: Subpixel Accuracy ===")
    
    # Create reference with smooth features for better subpixel accuracy
    ref = np.zeros((30, 96, 96), dtype=np.float32)
    z, y, x = np.ogrid[:30, :96, :96]
    ref += 10 * np.exp(-((z-15)**2 + (y-48)**2 + (x-48)**2) / 150)
    
    # Test fractional shifts
    fractional_shifts = [
        np.array([0.1, 0.2, 0.15], dtype=np.float32),
        np.array([0.5, -0.5, 0.25], dtype=np.float32),
        np.array([1.3, -0.7, 0.8], dtype=np.float32),
    ]
    
    for true_shift in fractional_shifts:
        # Create precisely shifted volume using scipy
        mov = ndi_shift(ref, shift=(-true_shift[2], -true_shift[1], -true_shift[0]), 
                       order=3, mode='nearest')  # Higher order for smooth shift
        
        # Test with high upsampling factor
        t0 = time.time()
        est = estimate_rigid_xcorr_3d(ref, mov, target_hw=None, up=100)
        t1 = time.time()
        error = np.abs(est - true_shift)
        
        print(f"True: {true_shift}, Est: {est}, Error: {error}, Max: {error.max():.3f}, Time: {(t1-t0)*1000:.1f}ms")
        
        # Should achieve subpixel accuracy
        assert error.max() < 0.2, f"Subpixel accuracy insufficient: {error}"
    
    print("✓ Subpixel accuracy test passed!")


def test_sign_convention():
    """Verify sign convention: estimate_rigid_xcorr_3d returns shift to apply to mov for backward warp."""
    print("\n=== Test: Sign Convention ===")
    
    # Create simple volumes for clear testing
    ref = np.zeros((20, 50, 50), dtype=np.float32)
    ref[10, 25, 25] = 100  # Single bright point at center
    
    # Move the point by known amount
    mov = np.zeros_like(ref)
    mov[10, 22, 28] = 100  # Point at different location
    
    print(f"Reference point at: Z=10, Y=25, X=25")
    print(f"Moved point at:     Z=10, Y=22, X=28")
    print(f"Movement: Y-3, X+3 (point moved up and right)")
    
    # For backward warp, we need to know where to fetch from
    # The point is at (22,28) in mov and should map to (25,25) in ref
    # So we need to fetch from (22,28) + displacement = (25,25)
    # Therefore displacement should be (+3, -3) in (Y,X)
    expected_shift = np.array([-3.0, 3.0, 0.0], dtype=np.float32)  # [dx, dy, dz]
    
    t0 = time.time()
    est = estimate_rigid_xcorr_3d(ref, mov, target_hw=None, up=1)
    t1 = time.time()
    
    print(f"Expected shift [dx,dy,dz]: {expected_shift}")
    print(f"Estimated:                 {est}")
    print(f"Time:                      {(t1-t0)*1000:.2f} ms")
    
    # Apply the shift and check
    aligned = prealign_for_flow(mov, est, order=0)  # order=0 for nearest neighbor
    aligned_point_pos = np.unravel_index(np.argmax(aligned), aligned.shape)
    print(f"After alignment, point at: Z={aligned_point_pos[0]}, Y={aligned_point_pos[1]}, X={aligned_point_pos[2]}")
    
    # Check sign is correct (within integer precision since up=1)
    assert np.allclose(est, expected_shift, atol=1.0), \
        f"Sign convention error: expected {expected_shift}, got {est}"
    
    # Check alignment worked
    assert aligned_point_pos == (10, 25, 25), \
        f"Alignment failed: point at {aligned_point_pos}, expected (10, 25, 25)"
    
    print("✓ Sign convention test passed!")


def test_z_axis_scaling():
    """Test that Z-axis downsampling works correctly with target_z parameter."""
    print("\n=== Test: Z-axis Scaling ===")
    
    # Create volume with different Z dimension
    ref = np.random.rand(80, 128, 128).astype(np.float32) * 0.5
    z, y, x = np.ogrid[:80, :128, :128]
    ref += 10 * np.exp(-((z-40)**2 + (y-64)**2 + (x-64)**2) / 300)
    
    # True displacement (backward warp convention)
    true_dx_dy_dz = np.array([2.5, -1.5, 3.5], dtype=np.float32)
    
    # Create moved volume
    mov = ndi_shift(ref, shift=(-true_dx_dy_dz[2], -true_dx_dy_dz[1], -true_dx_dy_dz[0]), 
                    order=1, mode='nearest')
    mov += 0.1 * np.random.randn(*ref.shape)
    
    # Test with Z downsampling
    t0 = time.time()
    est_no_z = estimate_rigid_xcorr_3d(ref, mov, target_hw=(128,128), target_z=None, up=10)
    t1 = time.time()
    time_no_z = (t1-t0)*1000
    
    t0 = time.time()
    est_with_z = estimate_rigid_xcorr_3d(ref, mov, target_hw=(128,128), target_z=40, up=10)
    t1 = time.time()
    time_with_z = (t1-t0)*1000
    
    error_no_z = np.abs(est_no_z - true_dx_dy_dz)
    error_with_z = np.abs(est_with_z - true_dx_dy_dz)
    
    print(f"No Z downsampling: error = {error_no_z}, max = {error_no_z.max():.3f}, time = {time_no_z:.1f}ms")
    print(f"Z downsampling to 40: error = {error_with_z}, max = {error_with_z.max():.3f}, time = {time_with_z:.1f}ms")
    
    # Both should work well
    assert error_no_z.max() < 0.5, f"Error without Z downsampling too large: {error_no_z}"
    assert error_with_z.max() < 0.8, f"Error with Z downsampling too large: {error_with_z}"
    
    print("✓ Z-axis scaling test passed!")


def test_pipeline_integration():
    """Test that prealignment integrates correctly with the full pipeline."""
    print("\n=== Test: Pipeline Integration ===")
    
    # Create test volume
    ref = np.random.rand(40, 100, 100).astype(np.float32) * 0.2
    z, y, x = np.ogrid[:40, :100, :100]
    ref += np.exp(-((z-20)**2 + (y-50)**2 + (x-50)**2) / 150)
    
    # True total displacement (what imregister_wrapper expects)
    true_total = np.array([4.5, -2.3, 1.8], dtype=np.float32)
    
    # Create moved volume
    mov = ndi_shift(ref, shift=(-true_total[2], -true_total[1], -true_total[0]), order=1, mode='nearest')
    mov += 0.05 * np.random.randn(*ref.shape)
    
    # Step 1: Estimate rigid displacement
    rigid = estimate_rigid_xcorr_3d(ref, mov, target_hw=(100,100), up=20)
    print(f"Rigid displacement estimated: dx={rigid[0]:.2f}, dy={rigid[1]:.2f}, dz={rigid[2]:.2f}")
    
    # Step 2: Pre-align
    mov_pre = prealign_for_flow(mov, rigid, order=1)
    
    # Step 3: Simulate optical flow on pre-aligned (should be near zero if perfect)
    residual = np.zeros(3, dtype=np.float32)  # Assume perfect prealignment for this test
    
    # Step 4: Compose displacements (this is what happens in the workers)
    total_flow = rigid + residual
    
    print(f"Total flow (rigid + residual): dx={total_flow[0]:.2f}, dy={total_flow[1]:.2f}, dz={total_flow[2]:.2f}")
    print(f"True total displacement:       dx={true_total[0]:.2f}, dy={true_total[1]:.2f}, dz={true_total[2]:.2f}")
    
    error = np.abs(total_flow - true_total)
    print(f"Error:                         {error}")
    
    # Check that composition is correct
    assert np.allclose(total_flow, true_total, atol=0.3), \
        f"Pipeline integration error: {error}"
    
    # Verify that pre-alignment actually aligns
    alignment_error = np.mean(np.abs(mov_pre - ref))
    print(f"Alignment error after prealign: {alignment_error:.6f}")
    assert alignment_error < 0.3, f"Pre-alignment didn't work: error={alignment_error}"
    
    print("✓ Pipeline integration test passed!")


if __name__ == "__main__":
    # Run all tests
    print("Running xcorr_prealignment tests...")
    test_pure_translation()
    test_translation_with_rotation()
    test_multichannel_alignment()
    test_downsampling_accuracy()
    test_subpixel_accuracy()
    test_sign_convention()
    test_z_axis_scaling()
    test_pipeline_integration()
    print("\n✅ All tests passed!")