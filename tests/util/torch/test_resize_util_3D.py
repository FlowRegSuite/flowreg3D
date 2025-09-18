"""
Tests for PyTorch implementation of resize_util_3D.
Verifies CPU-Torch parity for resize operations.
"""

import numpy as np
import torch

from flowreg3d.util.resize_util_3D import (
    imresize_fused_gauss_cubic3D as imresize_cpu,
    imresize2d_gauss_cubic as imresize2d_cpu
)
from flowreg3d.util.torch.resize_util_3D import (
    imresize_fused_gauss_cubic3D as imresize_torch,
    imresize2d_gauss_cubic as imresize2d_torch
)


class TestResizeUtil3D:
    """Test suite for 3D resize utilities with PyTorch backend."""

    def test_resize_3d_downsample(self):
        """Test 3D downsampling for CPU-Torch parity."""
        np.random.seed(42)
        data = np.random.randn(32, 48, 64).astype(np.float32)
        out_shape = (16, 24, 32)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)
        assert cpu_result.dtype == torch_result.dtype

    def test_resize_3d_upsample(self):
        """Test 3D upsampling for CPU-Torch parity."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32).astype(np.float32)
        out_shape = (32, 48, 64)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_3d_per_axis(self):
        """Test 3D resize with per-axis sigma computation."""
        np.random.seed(42)
        data = np.random.randn(32, 48, 64).astype(np.float32)
        out_shape = (16, 48, 32)  # Mixed: downsample Z and X, keep Y

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=True)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=True)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_3d_no_blur(self):
        """Test resize with sigma=0 (no Gaussian blur)."""
        np.random.seed(42)
        data = np.random.randn(32, 32, 32).astype(np.float32)
        out_shape = (64, 64, 64)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.0, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.0, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_3d_float64(self):
        """Test float64 dtype preservation."""
        np.random.seed(42)
        data = np.random.randn(16, 16, 16).astype(np.float64)
        out_shape = (32, 32, 32)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.8, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.8, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)
        assert cpu_result.dtype == np.float64
        assert torch_result.dtype == np.float64

    def test_resize_4d_with_channels(self):
        """Test 4D resize (with channel dimension)."""
        np.random.seed(42)
        data = np.random.randn(32, 48, 64, 3).astype(np.float32)
        out_shape = (16, 24, 32)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)
        assert cpu_result.shape == (16, 24, 32, 3)

    def test_resize_2d_wrapper(self):
        """Test 2D resize wrapper function."""
        np.random.seed(42)
        data = np.random.randn(64, 48).astype(np.float32)
        out_hw = (32, 24)

        cpu_result = imresize2d_cpu(data, out_hw, sigma_coeff=0.6)
        torch_result = imresize2d_torch(data, out_hw, sigma_coeff=0.6)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_2d_upsample(self):
        """Test 2D upsampling."""
        np.random.seed(42)
        data = np.random.randn(32, 24).astype(np.float32)
        out_hw = (64, 48)

        cpu_result = imresize2d_cpu(data, out_hw, sigma_coeff=0.6)
        torch_result = imresize2d_torch(data, out_hw, sigma_coeff=0.6)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_edge_cases(self):
        """Test edge cases: tiny arrays, single pixel dimensions."""
        # Tiny array
        np.random.seed(42)
        data = np.random.randn(2, 3, 4).astype(np.float32)
        out_shape = (4, 6, 8)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_identity(self):
        """Test identity resize (same input/output shape)."""
        np.random.seed(42)
        data = np.random.randn(32, 32, 32).astype(np.float32)
        out_shape = (32, 32, 32)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.6, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_resize_cuda(self):
        """Test resize on CUDA device if available."""
        if not torch.cuda.is_available():
            print("Skipping CUDA test (CUDA not available)")
            return
        np.random.seed(42)
        data = np.random.randn(16, 24, 32).astype(np.float32)
        data_tensor = torch.from_numpy(data).cuda()
        out_shape = (32, 48, 64)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data_tensor, out_shape, sigma_coeff=0.6, per_axis=False)

        assert torch_result.is_cuda
        np.testing.assert_allclose(cpu_result, torch_result.cpu().numpy(), rtol=1e-5, atol=1e-8)

    def test_resize_cpu_vectorized(self):
        """Test vectorized PyTorch operations on CPU."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32).astype(np.float32)
        data_tensor = torch.from_numpy(data)
        out_shape = (32, 48, 64)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.6, per_axis=False)
        torch_result = imresize_torch(data_tensor, out_shape, sigma_coeff=0.6, per_axis=False)

        assert not torch_result.is_cuda  # Ensure it's on CPU
        np.testing.assert_allclose(cpu_result, torch_result.cpu().numpy(), rtol=1e-6, atol=1e-9)

    def test_reflection_boundary(self):
        """Test reflection boundary condition handling."""
        # Create data with known pattern at boundaries
        data = np.zeros((8, 8, 8), dtype=np.float32)
        data[0, :, :] = 1.0  # First Z slice
        data[-1, :, :] = 2.0  # Last Z slice
        data[:, 0, :] = 3.0  # First Y slice
        data[:, -1, :] = 4.0  # Last Y slice

        out_shape = (16, 16, 16)

        cpu_result = imresize_cpu(data, out_shape, sigma_coeff=0.3, per_axis=False)
        torch_result = imresize_torch(data, out_shape, sigma_coeff=0.3, per_axis=False)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    # Run tests
    test = TestResizeUtil3D()

    print("Testing resize_util_3D PyTorch implementation...")
    test.test_resize_3d_downsample()
    print("✓ 3D downsample")

    test.test_resize_3d_upsample()
    print("✓ 3D upsample")

    test.test_resize_3d_per_axis()
    print("✓ Per-axis sigma")

    test.test_resize_3d_no_blur()
    print("✓ No blur (sigma=0)")

    test.test_resize_3d_float64()
    print("✓ Float64 dtype")

    test.test_resize_4d_with_channels()
    print("✓ 4D with channels")

    test.test_resize_2d_wrapper()
    print("✓ 2D wrapper")

    test.test_resize_2d_upsample()
    print("✓ 2D upsample")

    test.test_resize_edge_cases()
    print("✓ Edge cases")

    test.test_resize_identity()
    print("✓ Identity resize")

    test.test_reflection_boundary()
    print("✓ Reflection boundary")

    test.test_resize_cpu_vectorized()
    print("✓ CPU vectorized operations")

    if torch.cuda.is_available():
        test.test_resize_cuda()
        print("✓ CUDA operations")
    else:
        print("! CUDA not available, skipping GPU tests")

    print("\nAll resize_util_3D tests passed!")