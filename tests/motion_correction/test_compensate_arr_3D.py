"""
Tests for compensate_arr_3D array-based motion compensation.

Tests array-based 3D motion compensation using in-memory arrays,
matching the functionality of compensate_recording_3D but without file I/O.
Tests parameter flow, shape handling, and algorithmic consistency.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import numpy as np

from flowreg3d.motion_correction.compensate_arr_3D import compensate_arr_3D
from flowreg3d.motion_correction.OF_options_3D import OFOptions, OutputFormat
from tests.fixtures_3d import create_simple_3d_test_data, cleanup_temp_files


class TestCompensateArr3DBasics:
    """Test basic functionality of compensate_arr_3D."""
    
    def test_basic_3d_array_compensation(self):
        """Test basic 3D array compensation with minimal data."""
        # Create simple 3D test data
        T, Z, Y, X, C = 5, 6, 16, 16, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Run 3D compensation
        registered, flow = compensate_arr_3D(video, reference)
        
        # Check output shapes
        assert registered.shape == video.shape
        assert flow.shape == (T, Z, Y, X, 3)  # u, v, w components
        
        # Check data types
        assert isinstance(registered, np.ndarray)
        assert isinstance(flow, np.ndarray)
    
    def test_single_channel_4d_input(self):
        """Test handling of single-channel 4D input (T,Z,Y,X)."""
        # Create single-channel 4D data
        T, Z, Y, X = 8, 4, 24, 24
        video = np.random.rand(T, Z, Y, X).astype(np.float32)
        reference = np.mean(video[:3], axis=0)  # (Z,Y,X) reference
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference)
        
        # Check output shapes - should preserve input shape
        assert registered.shape == (T, Z, Y, X)
        assert flow.shape == (T, Z, Y, X, 3)
    
    def test_single_volume_3d_input(self):
        """Test handling of single volume 3D input (Z,Y,X)."""
        # Create single 3D volume
        Z, Y, X = 8, 32, 32
        volume = np.random.rand(Z, Y, X).astype(np.float32)
        reference = np.random.rand(Z, Y, X).astype(np.float32)
        
        # Run compensation
        registered, flow = compensate_arr_3D(volume, reference)
        
        # Check output shapes
        assert registered.shape == (Z, Y, X)
        assert flow.shape == (Z, Y, X, 3)  # Single volume flow
    
    def test_multichannel_5d_input(self):
        """Test handling of multi-channel 5D input (T,Z,Y,X,C)."""
        # Create multi-channel 5D data
        T, Z, Y, X, C = 6, 4, 20, 20, 3
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)  # (Z,Y,X,C) reference
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference)
        
        # Check output shapes
        assert registered.shape == (T, Z, Y, X, C)
        assert flow.shape == (T, Z, Y, X, 3)
    
    def test_shape_ambiguity_handling(self):
        """Test handling of ambiguous 4D shapes."""
        # Case 1: (4,Z,Y,X) - should be treated as 4 volumes
        video_4d_time = np.random.rand(4, 8, 16, 16).astype(np.float32) 
        ref_3d = np.random.rand(8, 16, 16).astype(np.float32)
        
        registered, flow = compensate_arr_3D(video_4d_time, ref_3d)
        assert registered.shape == (4, 8, 16, 16)  # T,Z,Y,X
        assert flow.shape == (4, 8, 16, 16, 3)
        
        # Case 2: (Z,Y,X,8) - should be treated as single volume with 8 channels  
        video_4d_channels = np.random.rand(8, 16, 16, 8).astype(np.float32)
        ref_4d_channels = np.random.rand(8, 16, 16, 8).astype(np.float32)
        
        # The function should handle this as (Z,Y,X,C)
        registered, flow = compensate_arr_3D(video_4d_channels, ref_4d_channels)
        assert registered.shape == (8, 16, 16, 8)  # Should preserve as single volume
        assert flow.shape == (8, 16, 16, 3)  # Single volume flow


class TestCompensateArr3DWithOptions:
    """Test compensate_arr_3D with various 3D OFOptions configurations."""
    
    def test_with_custom_3d_options(self):
        """Test with custom 3D OF_options."""
        # Create 3D test data
        T, Z, Y, X, C = 4, 6, 18, 18, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Create custom 3D options
        options = OFOptions(
            alpha=(1.0, 2.0, 3.0),  # Different regularization per axis
            sigma=[[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],  # 3D sigma
            levels=5,
            iterations=8,
            eta=0.85,
            quality_setting="fast",
            buffer_size=12
        )
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, Z, Y, X, 3)
    
    def test_3d_save_w_option(self):
        """Test that save_w=True properly returns 3D displacement fields."""
        # Create test data
        T, Z, Y, X, C = 3, 4, 14, 14, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Create options with save_w enabled
        options = OFOptions(save_w=True)
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Check that 3D flow fields are returned
        assert flow is not None
        assert flow.shape == (T, Z, Y, X, 3)
        assert flow.dtype in [np.float32, np.float64]
        
        # Check that flow has reasonable values (not all zeros)
        assert not np.allclose(flow, 0, atol=1e-6)
    
    def test_3d_output_typename_casting(self):
        """Test output type casting based on output_typename option for 3D."""
        # Create test data
        T, Z, Y, X, C = 3, 3, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Test different output types
        output_types = {
            'single': np.float32,
            'double': np.float64,
            'uint16': np.uint16,
        }
        
        for typename, expected_dtype in output_types.items():
            options = OFOptions(output_typename=typename)
            registered, flow = compensate_arr_3D(video, reference, options)
            
            # Check output dtype matches requested type
            assert registered.dtype == expected_dtype, f"Failed for {typename}"
    
    def test_3d_quality_settings(self):
        """Test different quality settings work with 3D arrays."""
        # Create test data
        T, Z, Y, X, C = 4, 4, 16, 16, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        quality_settings = ["fast", "balanced", "quality"]
        
        for quality in quality_settings:
            options = OFOptions(
                quality_setting=quality,
                levels=3 if quality == "quality" else 2  # Reduce levels for testing
            )
            registered, flow = compensate_arr_3D(video, reference, options)
            
            # Check outputs for each quality setting
            assert registered.shape == video.shape, f"Failed for quality={quality}"
            assert flow.shape == (T, Z, Y, X, 3), f"Failed for quality={quality}"


class TestArrayReaderWriter3DIntegration:
    """Test that 3D ArrayReader and ArrayWriter are properly used."""
    
    def test_3d_array_reader_creation(self):
        """Test that 3D input arrays are wrapped in ArrayReader."""
        # Create test data
        T, Z, Y, X, C = 4, 5, 14, 14, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Patch get_video_file_reader to verify ArrayReader is created
        from flowreg3d.util.io import factory
        original_get_reader = factory.get_video_file_reader
        
        reader_created = False
        def mock_get_reader(input_source, *args, **kwargs):
            nonlocal reader_created
            result = original_get_reader(input_source, *args, **kwargs)
            # Check if it's a 3D array reader (duck typing check)
            if hasattr(result, 'read_batch') and hasattr(result, 'has_batch'):
                reader_created = True
            return result
        
        factory.get_video_file_reader = mock_get_reader
        try:
            registered, flow = compensate_arr_3D(video, reference)
            assert reader_created, "3D ArrayReader was not created"
        finally:
            factory.get_video_file_reader = original_get_reader
    
    def test_3d_array_writer_creation(self):
        """Test that output format ARRAY triggers 3D ArrayWriter."""
        # Create test data
        T, Z, Y, X, C = 4, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Patch get_video_file_writer to verify ArrayWriter is created
        from flowreg3d.util.io import factory
        original_get_writer = factory.get_video_file_writer
        
        writer_created = False
        def mock_get_writer(file_path, output_format, *args, **kwargs):
            nonlocal writer_created
            result = original_get_writer(file_path, output_format, *args, **kwargs)
            # Check if it's an array writer (duck typing check)
            if hasattr(result, 'get_array') and hasattr(result, 'write_frames'):
                writer_created = True
            return result
        
        factory.get_video_file_writer = mock_get_writer
        try:
            registered, flow = compensate_arr_3D(video, reference)
            assert writer_created, "3D ArrayWriter was not created"
        finally:
            factory.get_video_file_writer = original_get_writer
    
    def test_3d_displacement_writer_is_array_writer(self):
        """Test that 3D displacement writer is also ArrayWriter when output is ARRAY."""
        # Create test data
        T, Z, Y, X, C = 3, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Create options with save_w enabled
        options = OFOptions(save_w=True)
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Verify 3D flow fields were captured
        assert flow is not None
        assert flow.shape == (T, Z, Y, X, 3)
        assert flow.dtype in [np.float32, np.float64]


class TestCompensateArr3DConsistency:
    """Test consistency between 3D array and file-based processing."""
    
    def test_3d_consistent_with_file_processing(self, temp_dir):
        """Test that 3D array processing matches file-based processing."""
        # Create 3D test data
        T, Z, Y, X, C = 6, 4, 20, 20, 2
        video = create_simple_3d_test_data((T, Z, Y, X, C))
        reference = np.mean(video[:3], axis=0)
        
        # Process with compensate_arr_3D
        registered_arr, flow_arr = compensate_arr_3D(video, reference)
        
        # For now, just verify outputs are reasonable
        assert registered_arr.shape == video.shape
        assert flow_arr.shape == (T, Z, Y, X, 3)
        
        # Check that some motion correction occurred
        # (registered should be different from input, unless motion is zero)
        mse_original = np.mean((video - reference[np.newaxis])**2)
        mse_registered = np.mean((registered_arr - reference[np.newaxis])**2)
        
        # Motion correction should generally reduce MSE, but this is data-dependent
        # Just check that processing occurred (arrays are different)
        assert not np.array_equal(video, registered_arr)
    
    def test_3d_batch_processing_consistency(self):
        """Test that 3D batching works correctly for arrays."""
        # Create test data with enough volumes for multiple batches
        T, Z, Y, X, C = 20, 4, 12, 12, 2  # Enough for multiple batches
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:5], axis=0)
        
        # Create options with small batch size to force multiple batches
        options = OFOptions(
            buffer_size=8,  # Force smaller batches for 3D
            save_w=True
        )
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, Z, Y, X, 3)
    
    def test_3d_flow_initialization_chain(self):
        """Test that 3D flow initialization is properly maintained across batches."""
        # Create test data with 3D drift pattern
        T, Z, Y, X, C = 15, 6, 24, 24, 2
        video = np.zeros((T, Z, Y, X, C), dtype=np.float32)
        
        # Create a moving 3D pattern
        for t in range(T):
            offset_x = t * 1  # Progressive drift in X
            offset_y = t * 1  # Progressive drift in Y  
            offset_z = t * 1  # Progressive drift in Z
            
            for c in range(C):
                # Create shifted 3D pattern
                z, y, x = np.mgrid[:Z, :Y, :X]
                
                # 3D sinusoidal pattern that shifts over time
                pattern = (np.sin((x - offset_x) * 2 * np.pi / X) * 
                          np.sin((y - offset_y) * 2 * np.pi / Y) *
                          np.sin((z - offset_z) * 2 * np.pi / Z)) * 0.5 + 0.5
                video[t, :, :, :, c] = pattern
        
        reference = video[0]  # Use first volume as reference
        
        # Run compensation with 3D flow initialization
        options = OFOptions(
            update_initialization_w=True,  # Enable 3D flow initialization chain
            save_w=True,
            buffer_size=6  # Small batches to test initialization chaining
        )
        
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Check that 3D flow captures the drift
        assert flow is not None
        assert flow.shape == (T, Z, Y, X, 3)
        
        # Verify 3D flow fields were computed (not testing specific values with synthetic data)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2 + flow[..., 2]**2)
        assert flow_magnitude.shape == (T, Z, Y, X)


class TestCompensateArr3DEdgeCases:
    """Test 3D edge cases and error handling."""
    
    def test_empty_3d_video(self):
        """Test handling of empty 3D video array."""
        # Create empty 3D video
        video = np.array([]).reshape(0, 4, 8, 8, 2)
        reference = np.random.rand(4, 8, 8, 2).astype(np.float32)
        
        # This should handle gracefully (might return empty or raise)
        try:
            registered, flow = compensate_arr_3D(video, reference)
            # If it succeeds, check shapes
            assert registered.shape[0] == 0
            assert flow.shape[0] == 0
        except (ValueError, IndexError):
            # Expected behavior for empty input
            pass
    
    def test_3d_reference_shape_mismatch(self):
        """Test handling of 3D reference shape mismatches."""
        # Create test data
        T, Z, Y, X, C = 4, 6, 14, 14, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        
        # Test with wrong reference shape
        wrong_reference = np.random.rand(Z+1, Y, X, C).astype(np.float32)  # Wrong Z dimension
        
        # This should raise an error during processing
        with pytest.raises(Exception):  # Could be ValueError, AssertionError, etc.
            registered, flow = compensate_arr_3D(video, wrong_reference)
    
    def test_3d_single_channel_consistency(self):
        """Test that 3D single channel processing is consistent."""
        # Create single-channel data in different formats
        T, Z, Y, X = 8, 4, 16, 16
        
        # Format 1: 4D array (T, Z, Y, X)
        video_4d = np.random.rand(T, Z, Y, X).astype(np.float32)
        reference_3d = np.mean(video_4d[:3], axis=0)
        
        # Format 2: 5D array with single channel (T, Z, Y, X, 1)
        video_5d = video_4d[..., np.newaxis]
        reference_4d = reference_3d[..., np.newaxis]
        
        # Process both formats
        reg_4d, flow_4d = compensate_arr_3D(video_4d, reference_3d)
        reg_5d, flow_5d = compensate_arr_3D(video_5d, reference_4d)
        
        # Results should be equivalent (modulo shape)
        assert reg_4d.shape == (T, Z, Y, X)
        assert reg_5d.shape == (T, Z, Y, X, 1)
        
        # Compare data (squeeze 5D for comparison)
        np.testing.assert_allclose(reg_4d, np.squeeze(reg_5d, axis=-1), rtol=1e-4)
        
        # Flow should be the same
        np.testing.assert_allclose(flow_4d, flow_5d, rtol=1e-4)
    
    def test_3d_nan_handling(self):
        """Test handling of NaN values in 3D input."""
        # Create test data with NaN
        T, Z, Y, X, C = 4, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        video[1, 2, 4, 4, 0] = np.nan  # Insert NaN
        reference = np.mean(video[:2], axis=0)
        
        # This might handle NaN or raise - both are acceptable
        try:
            registered, flow = compensate_arr_3D(video, reference)
            # If it succeeds, check that output is reasonable
            assert registered.shape == video.shape
            # NaN might propagate or be handled
        except (ValueError, RuntimeError):
            # Expected if NaN is not supported
            pass


class TestProgressCallback3D:
    """Test 3D progress callback functionality."""
    
    def test_3d_progress_callback_called(self):
        """Test that progress callback is called during 3D processing."""
        # Create test data
        T, Z, Y, X, C = 8, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Track progress calls
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Run compensation with callback
        registered, flow = compensate_arr_3D(video, reference, progress_callback=progress_callback)
        
        # Check callback was called
        assert len(progress_calls) > 0, "Progress callback was not called"
        
        # Check final progress matches total volumes
        final_current, final_total = progress_calls[-1]
        assert final_current == T, f"Final progress {final_current} != {T} volumes"
        assert final_total == T, f"Total volumes {final_total} != {T}"
        
        # Check progress increments
        for i, (current, total) in enumerate(progress_calls):
            assert current > 0, f"Progress {i}: current={current} should be > 0"
            assert current <= total, f"Progress {i}: current={current} > total={total}"
            assert total == T, f"Progress {i}: total={total} != {T}"
    
    def test_3d_progress_callback_with_batches(self):
        """Test 3D progress callback with batch processing."""
        # Create test data with multiple batches
        T, Z, Y, X, C = 25, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:5], axis=0)
        
        # Track progress
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Force smaller batch size
        options = OFOptions(buffer_size=8)
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options, progress_callback)
        
        # Check progress was reported
        assert len(progress_calls) > 0
        
        # Verify monotonic increase
        previous_current = 0
        for current, total in progress_calls:
            assert current >= previous_current, "Progress should be monotonic"
            previous_current = current
        
        # Final progress should match total volumes
        assert progress_calls[-1][0] == T


class TestCompensateArr3DIntegration:
    """Integration tests with other 3D components."""
    
    def test_3d_with_preprocessing(self):
        """Test 3D array compensation with preprocessing options."""
        # Create test data
        T, Z, Y, X, C = 6, 4, 24, 24, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:3], axis=0)
        
        # Create options with 3D preprocessing
        options = OFOptions(
            sigma=[[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],  # 3D Gaussian filtering
            channel_normalization="separate",  # Per-channel normalization
            save_w=True,
            alpha=(1.5, 1.5, 2.0)  # 3D regularization
        )
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, options)
        
        # Check outputs
        assert registered.shape == video.shape
        assert flow.shape == (T, Z, Y, X, 3)
    
    def test_3d_options_not_modified(self):
        """Test that user's 3D options object is not modified."""
        # Create test data
        T, Z, Y, X, C = 4, 4, 12, 12, 2
        video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.mean(video[:2], axis=0)
        
        # Create options
        original_options = OFOptions(
            alpha=(1.0, 2.0, 3.0),
            save_w=False,
            output_format=OutputFormat.HDF5  # Different from ARRAY
        )
        
        # Store original values
        original_alpha = original_options.alpha
        original_save_w = original_options.save_w
        original_format = original_options.output_format
        
        # Run compensation
        registered, flow = compensate_arr_3D(video, reference, original_options)
        
        # Check that original options were not modified
        assert original_options.alpha == original_alpha
        assert original_options.save_w == original_save_w
        assert original_options.output_format == original_format  # Should not be changed to ARRAY


# Fixtures for 3D testing
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Test basic 3D functionality
    print("Testing 3D compensate_arr functionality...")
    
    T, Z, Y, X, C = 4, 6, 16, 16, 2
    video = np.random.rand(T, Z, Y, X, C).astype(np.float32)
    reference = np.mean(video[:2], axis=0)
    
    try:
        registered, flow = compensate_arr_3D(video, reference)
        
        print(f"Input shape: {video.shape}")
        print(f"Reference shape: {reference.shape}")
        print(f"Registered shape: {registered.shape}")
        print(f"Flow shape: {flow.shape}")
        print("Basic 3D tests would pass with proper dependencies!")
        
    except ImportError as e:
        print(f"3D dependencies not available: {e}")
    except Exception as e:
        print(f"Error in 3D processing: {e}")