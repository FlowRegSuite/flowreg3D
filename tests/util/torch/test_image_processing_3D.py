"""
Tests for PyTorch implementation of image_processing_3D.
Verifies CPU-Torch parity for normalization and filtering operations.
"""

import numpy as np
import torch
from collections import deque

import pytest

from flowreg3d.util.image_processing_3D import (
    normalize as normalize_cpu,
    apply_gaussian_filter as apply_gaussian_cpu,
    gaussian_filter_1d_half_kernel as gaussian_half_cpu,
)
from flowreg3d.util.torch.image_processing_3D import (
    normalize as normalize_torch,
    apply_gaussian_filter as apply_gaussian_torch,
    gaussian_filter_1d_half_kernel as gaussian_half_torch,
)

# Tolerances for CPU/Torch parity (allow small numerical drift)
NORM_RTOL = 1e-4
NORM_ATOL = 1e-4
GAUSS_RTOL = 5e-3
GAUSS_ATOL = 5e-4
HALF_RTOL = 1e-4
HALF_ATOL = 1e-6


class TestNormalize:
    """Test suite for normalization functions."""

    def test_normalize_global_4d(self):
        """Test global normalization on 4D arrays."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 3).astype(np.float32)

        cpu_result = normalize_cpu(data, channel_normalization="together")
        torch_result = normalize_torch(data, channel_normalization="together")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )
        assert cpu_result.dtype == torch_result.dtype

    def test_normalize_separate_4d(self):
        """Test per-channel normalization on 4D arrays."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 3).astype(np.float32)

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data, channel_normalization="separate")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )
        assert cpu_result.dtype == torch_result.dtype

    def test_normalize_with_reference(self):
        """Test normalization with reference array."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 3).astype(np.float32)
        ref = np.random.randn(16, 24, 32, 3).astype(np.float32)

        cpu_result = normalize_cpu(data, ref=ref, channel_normalization="separate")
        torch_result = normalize_torch(data, ref=ref, channel_normalization="separate")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )

    def test_normalize_5d(self):
        """Test normalization on 5D arrays (T,Z,Y,X,C)."""
        np.random.seed(42)
        data = np.random.randn(8, 16, 24, 32, 2).astype(np.float32)

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data, channel_normalization="separate")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )

    def test_normalize_uniform_data(self):
        """Test normalization on uniform data (all values same)."""
        data = np.ones((8, 8, 8, 2), dtype=np.float32) * 5.0

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data, channel_normalization="separate")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )
        # Should return zeros (arr - min_val) when all values are same
        np.testing.assert_allclose(cpu_result, np.zeros_like(data), atol=1e-10)

    def test_normalize_3d_fallback(self):
        """Test normalization fallback for 3D arrays."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32).astype(np.float32)

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data, channel_normalization="separate")

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )
        assert cpu_result.dtype == torch_result.dtype

    def test_normalize_eps_handling(self):
        """Test epsilon handling in global normalization."""
        data = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)

        cpu_result = normalize_cpu(data, channel_normalization="together", eps=1e-8)
        torch_result = normalize_torch(data, channel_normalization="together", eps=1e-8)

        np.testing.assert_allclose(
            cpu_result, torch_result, rtol=NORM_RTOL, atol=NORM_ATOL
        )

    def test_normalize_cpu_vectorized(self):
        """Test vectorized PyTorch normalization on CPU."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 3).astype(np.float32)
        data_tensor = torch.from_numpy(data)

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data_tensor, channel_normalization="separate")

        assert not torch_result.is_cuda  # Ensure it's on CPU
        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=NORM_RTOL, atol=NORM_ATOL
        )

    def test_normalize_cuda(self):
        """Test normalization on CUDA device if available."""
        if not torch.cuda.is_available():
            print("Skipping CUDA test (CUDA not available)")
            return
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 3).astype(np.float32)
        data_tensor = torch.from_numpy(data).cuda()

        cpu_result = normalize_cpu(data, channel_normalization="separate")
        torch_result = normalize_torch(data_tensor, channel_normalization="separate")

        assert torch_result.is_cuda
        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=NORM_RTOL, atol=NORM_ATOL
        )


class TestGaussianFilter:
    """Test suite for Gaussian filtering functions."""

    def test_gaussian_3d_single_sigma(self):
        """Test 3D Gaussian filtering with single sigma for all channels."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 2).astype(np.float32)
        sigma = np.array([1.5, 1.0, 2.0])  # sx, sy, sz

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect", truncate=4.0)
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect", truncate=4.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-8)

        # Verify float64 output
        assert cpu_result.dtype == np.float64
        assert torch_result.dtype == np.float64

    def test_gaussian_3d_per_channel_sigma(self):
        """Test 3D Gaussian filtering with per-channel sigmas."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 2).astype(np.float32)
        sigma = np.array(
            [
                [1.0, 1.5, 2.0],  # Channel 0
                [2.0, 1.0, 1.5],  # Channel 1
            ]
        )

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect", truncate=3.0)
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect", truncate=3.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-8)

    def test_gaussian_4d_spatiotemporal(self):
        """Test 4D spatiotemporal Gaussian filtering."""
        np.random.seed(42)
        data = np.random.randn(8, 16, 24, 32, 2).astype(np.float32)
        sigma = np.array([1.5, 1.0, 2.0, 0.8])  # sx, sy, sz, st

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect", truncate=4.0)
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect", truncate=4.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-8)

    def test_gaussian_zero_sigma(self):
        """Test Gaussian filtering with sigma=0 (no filtering)."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 2).astype(np.float32)
        sigma = np.array([0.0, 0.0, 0.0])

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect")
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect")

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-8)

    def test_gaussian_truncate_variation(self):
        """Test Gaussian filtering with different truncate values."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32, 1).astype(np.float32)
        sigma = np.array([1.0, 1.0, 1.0])

        for truncate in [2.0, 3.0, 4.0, 6.0]:
            cpu_result = apply_gaussian_cpu(
                data, sigma, mode="reflect", truncate=truncate
            )
            torch_result = apply_gaussian_torch(
                data, sigma, mode="reflect", truncate=truncate
            )

            np.testing.assert_allclose(
                cpu_result,
                torch_result,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Failed for truncate={truncate}",
            )

    def test_gaussian_mode_enforcement(self):
        """Test that non-reflect modes raise an error."""
        data = np.random.randn(8, 8, 8, 1).astype(np.float32)
        sigma = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="Only 'reflect' mode is supported"):
            _ = apply_gaussian_torch(data, sigma, mode="constant")

        with pytest.raises(ValueError, match="Only 'reflect' mode is supported"):
            _ = apply_gaussian_torch(data, sigma, mode="wrap")

    def test_gaussian_3d_direct(self):
        """Test direct 3D Gaussian filtering (unsupported dimensionality path)."""
        np.random.seed(42)
        data = np.random.randn(16, 24, 32).astype(np.float32)  # 3D without channels
        sigma = np.array([1.0, 1.5, 2.0])

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect")
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect")

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-8)

    def test_gaussian_float64_input(self):
        """Test Gaussian filtering with float64 input."""
        np.random.seed(42)
        data = np.random.randn(8, 12, 16, 1).astype(np.float64)
        sigma = np.array([1.0, 1.0, 1.0])

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect")
        torch_result = apply_gaussian_torch(data, sigma, mode="reflect")

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-10, atol=1e-12)
        assert cpu_result.dtype == np.float64
        assert torch_result.dtype == np.float64

    def test_gaussian_cpu_vectorized(self):
        """Test vectorized PyTorch Gaussian filtering on CPU."""
        np.random.seed(42)
        data = np.random.randn(8, 12, 16, 2).astype(np.float32)
        data_tensor = torch.from_numpy(data)
        sigma = np.array([1.0, 1.5, 0.5])

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect")
        torch_result = apply_gaussian_torch(data_tensor, sigma, mode="reflect")

        assert not torch_result.is_cuda  # Ensure it's on CPU
        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=1e-6, atol=1e-8
        )

    def test_gaussian_cuda(self):
        """Test Gaussian filtering on CUDA device if available."""
        if not torch.cuda.is_available():
            print("Skipping CUDA test (CUDA not available)")
            return
        np.random.seed(42)
        data = np.random.randn(8, 12, 16, 2).astype(np.float32)
        data_tensor = torch.from_numpy(data).cuda()
        sigma = np.array([1.0, 1.5, 0.5])

        cpu_result = apply_gaussian_cpu(data, sigma, mode="reflect")
        torch_result = apply_gaussian_torch(data_tensor, sigma, mode="reflect")

        assert torch_result.is_cuda
        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=1e-6, atol=1e-8
        )


class TestGaussianHalfKernel:
    """Test suite for half-kernel Gaussian filtering."""

    def test_half_kernel_basic(self):
        """Test basic half-kernel Gaussian filtering."""
        np.random.seed(42)
        buffer_size = 5
        frame_shape = (32, 48)

        # Create buffer with NumPy arrays
        buffer_np = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            buffer_np.append(np.random.randn(*frame_shape).astype(np.float32))

        sigma_t = 1.0
        cpu_result = gaussian_half_cpu(buffer_np, sigma_t, truncate=4.0)
        torch_result = gaussian_half_torch(buffer_np, sigma_t, truncate=4.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-6, atol=1e-9)

    def test_half_kernel_various_sigmas(self):
        """Test half-kernel with various sigma values."""
        np.random.seed(42)
        buffer_size = 5
        frame_shape = (32, 48)

        buffer_np = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            buffer_np.append(np.random.randn(*frame_shape).astype(np.float32))

        for sigma_t in [0.0, 0.5, 1.0, 2.0, 3.0]:
            cpu_result = gaussian_half_cpu(buffer_np, sigma_t, truncate=4.0)
            torch_result = gaussian_half_torch(buffer_np, sigma_t, truncate=4.0)

            np.testing.assert_allclose(
                cpu_result,
                torch_result,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"Failed for sigma_t={sigma_t}",
            )

    def test_half_kernel_torch_tensors(self):
        """Test half-kernel with PyTorch tensor inputs."""
        np.random.seed(42)
        buffer_size = 5
        frame_shape = (32, 48)

        # Create NumPy buffer
        buffer_np = deque(maxlen=buffer_size)
        frames = []
        for _ in range(buffer_size):
            frame = np.random.randn(*frame_shape).astype(np.float32)
            frames.append(frame)
            buffer_np.append(frame)

        # Create PyTorch buffer from same data
        buffer_torch = deque(maxlen=buffer_size)
        for frame in frames:
            buffer_torch.append(torch.from_numpy(frame))

        sigma_t = 1.5
        cpu_result = gaussian_half_cpu(buffer_np, sigma_t, truncate=4.0)
        torch_result = gaussian_half_torch(buffer_torch, sigma_t, truncate=4.0)

        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=1e-6, atol=1e-9
        )

    def test_half_kernel_single_frame(self):
        """Test half-kernel with single frame in buffer."""
        frame = np.random.randn(16, 16).astype(np.float32)
        buffer = deque([frame])

        cpu_result = gaussian_half_cpu(buffer, 1.0)
        torch_result = gaussian_half_torch(buffer, 1.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-10, atol=1e-10)
        # Should return copy of the single frame
        np.testing.assert_array_equal(cpu_result, frame)

    def test_half_kernel_empty_buffer(self):
        """Test half-kernel with empty buffer."""
        buffer = deque()

        cpu_result = gaussian_half_cpu(buffer, 1.0)
        torch_result = gaussian_half_torch(buffer, 1.0)

        assert cpu_result is None
        assert torch_result is None

    def test_half_kernel_truncate_variation(self):
        """Test half-kernel with different truncate values."""
        np.random.seed(42)
        buffer_size = 8
        frame_shape = (16, 16)

        buffer = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            buffer.append(np.random.randn(*frame_shape).astype(np.float32))

        sigma_t = 2.0
        for truncate in [2.0, 3.0, 4.0, 6.0]:
            cpu_result = gaussian_half_cpu(buffer, sigma_t, truncate=truncate)
            torch_result = gaussian_half_torch(buffer, sigma_t, truncate=truncate)

            np.testing.assert_allclose(
                cpu_result,
                torch_result,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"Failed for truncate={truncate}",
            )

    def test_half_kernel_float64(self):
        """Test half-kernel with float64 data."""
        np.random.seed(42)
        buffer_size = 3
        frame_shape = (16, 24)

        buffer = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            buffer.append(np.random.randn(*frame_shape).astype(np.float64))

        sigma_t = 1.0
        cpu_result = gaussian_half_cpu(buffer, sigma_t, truncate=4.0)
        torch_result = gaussian_half_torch(buffer, sigma_t, truncate=4.0)

        np.testing.assert_allclose(cpu_result, torch_result, rtol=1e-10, atol=1e-12)
        assert cpu_result.dtype == np.float64

    def test_half_kernel_cpu_vectorized(self):
        """Test vectorized PyTorch half-kernel on CPU."""
        np.random.seed(42)
        buffer_size = 4
        frame_shape = (32, 32)

        buffer = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            frame = torch.randn(*frame_shape, dtype=torch.float32)
            buffer.append(frame)

        sigma_t = 1.0
        torch_result = gaussian_half_torch(buffer, sigma_t, truncate=4.0)

        assert not torch_result.is_cuda  # Ensure it's on CPU

        # Compare against NumPy version
        buffer_np = deque(maxlen=buffer_size)
        for frame in buffer:
            buffer_np.append(frame.numpy())
        cpu_result = gaussian_half_cpu(buffer_np, sigma_t, truncate=4.0)

        np.testing.assert_allclose(
            cpu_result, torch_result.numpy(), rtol=1e-6, atol=1e-9
        )

    def test_half_kernel_cuda(self):
        """Test half-kernel on CUDA device if available."""
        if not torch.cuda.is_available():
            print("Skipping CUDA test (CUDA not available)")
            return
        np.random.seed(42)
        buffer_size = 4
        frame_shape = (32, 32)

        buffer = deque(maxlen=buffer_size)
        for _ in range(buffer_size):
            frame = torch.randn(*frame_shape, dtype=torch.float32, device="cuda")
            buffer.append(frame)

        sigma_t = 1.0
        torch_result = gaussian_half_torch(buffer, sigma_t, truncate=4.0)

        assert torch_result.is_cuda
        # Compare against CPU version
        buffer_cpu = deque(maxlen=buffer_size)
        for frame in buffer:
            buffer_cpu.append(frame.cpu().numpy())
        cpu_result = gaussian_half_cpu(buffer_cpu, sigma_t, truncate=4.0)

        np.testing.assert_allclose(
            cpu_result, torch_result.cpu().numpy(), rtol=1e-5, atol=1e-8
        )


if __name__ == "__main__":
    # Run tests
    print("Testing image_processing_3D PyTorch implementation...")

    # Test normalize
    print("\n=== Normalization Tests ===")
    test_norm = TestNormalize()
    test_norm.test_normalize_global_4d()
    print("✓ Global normalization 4D")
    test_norm.test_normalize_separate_4d()
    print("✓ Per-channel normalization 4D")
    test_norm.test_normalize_with_reference()
    print("✓ Normalization with reference")
    test_norm.test_normalize_5d()
    print("✓ Normalization 5D")
    test_norm.test_normalize_uniform_data()
    print("✓ Uniform data normalization")
    test_norm.test_normalize_3d_fallback()
    print("✓ 3D fallback")
    test_norm.test_normalize_eps_handling()
    print("✓ Epsilon handling")
    test_norm.test_normalize_cpu_vectorized()
    print("✓ CPU vectorized normalization")
    if torch.cuda.is_available():
        test_norm.test_normalize_cuda()
        print("✓ CUDA normalization")
    else:
        print("! CUDA not available, skipping GPU tests")

    # Test Gaussian filter
    print("\n=== Gaussian Filter Tests ===")
    test_gauss = TestGaussianFilter()
    test_gauss.test_gaussian_3d_single_sigma()
    print("✓ 3D single sigma")
    test_gauss.test_gaussian_3d_per_channel_sigma()
    print("✓ 3D per-channel sigma")
    test_gauss.test_gaussian_4d_spatiotemporal()
    print("✓ 4D spatiotemporal")
    test_gauss.test_gaussian_zero_sigma()
    print("✓ Zero sigma")
    test_gauss.test_gaussian_truncate_variation()
    print("✓ Truncate variations")
    test_gauss.test_gaussian_mode_enforcement()
    print("✓ Mode enforcement")
    test_gauss.test_gaussian_3d_direct()
    print("✓ 3D direct")
    test_gauss.test_gaussian_float64_input()
    print("✓ Float64 input")
    test_gauss.test_gaussian_cpu_vectorized()
    print("✓ CPU vectorized filtering")
    if torch.cuda.is_available():
        test_gauss.test_gaussian_cuda()
        print("✓ CUDA filtering")
    else:
        print("! CUDA not available, skipping GPU tests")

    # Test half kernel
    print("\n=== Half Kernel Tests ===")
    test_half = TestGaussianHalfKernel()
    test_half.test_half_kernel_basic()
    print("✓ Basic half kernel")
    test_half.test_half_kernel_various_sigmas()
    print("✓ Various sigmas")
    test_half.test_half_kernel_torch_tensors()
    print("✓ Torch tensor input")
    test_half.test_half_kernel_single_frame()
    print("✓ Single frame")
    test_half.test_half_kernel_empty_buffer()
    print("✓ Empty buffer")
    test_half.test_half_kernel_truncate_variation()
    print("✓ Truncate variations")
    test_half.test_half_kernel_float64()
    print("✓ Float64 data")
    test_half.test_half_kernel_cpu_vectorized()
    print("✓ CPU vectorized half kernel")
    if torch.cuda.is_available():
        test_half.test_half_kernel_cuda()
        print("✓ CUDA half kernel")
    else:
        print("! CUDA not available, skipping GPU tests")

    print("\nAll image_processing_3D tests passed!")
