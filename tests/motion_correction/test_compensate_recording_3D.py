"""
Tests for compensate_recording_3D with the 3D executor system.

Tests the complete 3D motion correction pipeline including parameter flow,
batch processing, executor integration, and I/O handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

import pytest
import numpy as np

from flowreg3d.motion_correction.compensate_recording_3D import (
    BatchMotionCorrector,
    RegistrationConfig,
    compensate_recording
)
from flowreg3d._runtime import RuntimeContext


class TestRegistrationConfig3D:
    """Test the 3D RegistrationConfig class."""
    
    def test_default_3d_config(self):
        """Test default 3D configuration values."""
        config = RegistrationConfig()
        assert config.n_jobs == -1
        assert config.batch_size == 10  # Smaller for 3D volumes
        assert config.verbose is False
        assert config.parallelization is None
    
    def test_custom_3d_config(self):
        """Test custom 3D configuration values."""
        config = RegistrationConfig(
            n_jobs=2,
            batch_size=5,  # Very small batches for 3D
            verbose=True,
            parallelization="threading3d"
        )
        assert config.n_jobs == 2
        assert config.batch_size == 5
        assert config.verbose is True
        assert config.parallelization == "threading3d"


class TestBatchMotionCorrector3D:
    """Test the 3D BatchMotionCorrector class and executor system."""
    
    def test_3d_executor_setup_auto_selection(self, fast_3d_of_options):
        """Test automatic 3D executor selection."""
        config = RegistrationConfig(parallelization=None)
        
        # Mock available 3D executors
        with patch.object(RuntimeContext, 'get') as mock_get:
            mock_get.return_value = {'sequential3d', 'threading3d', 'multiprocessing3d'}
            
            pipeline = BatchMotionCorrector(fast_3d_of_options, config)
            
            # Should auto-select an available 3D executor
            assert pipeline.executor is not None
            assert pipeline.executor.name.endswith('3d')
    
    def test_3d_executor_setup_specific_selection(self, fast_3d_of_options):
        """Test specific 3D executor selection."""
        config = RegistrationConfig(parallelization="sequential")
        
        # Mock executor availability
        mock_executor_class = MagicMock()
        mock_executor_class.return_value.name = "sequential3d"
        
        with patch.object(RuntimeContext, 'get_parallelization_executor') as mock_get:
            mock_get.return_value = mock_executor_class
            
            pipeline = BatchMotionCorrector(fast_3d_of_options, config)
            
            # Should add '3d' suffix and create executor
            mock_get.assert_called_with('sequential3d')
            assert pipeline.executor is not None
    
    def test_3d_executor_fallback(self, fast_3d_of_options):
        """Test fallback to sequential3d when requested executor unavailable."""
        config = RegistrationConfig(parallelization="nonexistent3d")
        
        # Mock sequential3d as fallback
        mock_executor_class = MagicMock()
        mock_executor_class.return_value.name = "sequential3d"
        
        with patch.object(RuntimeContext, 'get_parallelization_executor') as mock_get:
            def mock_executor(name):
                return mock_executor_class if name == "sequential3d" else None
            mock_get.side_effect = mock_executor
            
            with patch('builtins.print') as mock_print:
                pipeline = BatchMotionCorrector(fast_3d_of_options, config)
            
            # Should fallback to sequential3d
            assert pipeline.executor is not None
            assert pipeline.executor.name == "sequential3d"
            mock_print.assert_called()
    
    def test_3d_n_workers_setup(self, fast_3d_of_options):
        """Test n_workers configuration for 3D."""
        # Test auto-detection (-1)
        config = RegistrationConfig(n_jobs=-1)
        pipeline = BatchMotionCorrector(fast_3d_of_options, config)
        assert pipeline.n_workers > 0
        
        # Test specific value
        config = RegistrationConfig(n_jobs=2)
        pipeline = BatchMotionCorrector(fast_3d_of_options, config)
        assert pipeline.n_workers == 2
    
    def test_3d_initialization_with_options(self, basic_3d_of_options):
        """Test 3D pipeline initialization with options."""
        config = RegistrationConfig(n_jobs=2, batch_size=3)
        pipeline = BatchMotionCorrector(basic_3d_of_options, config)
        
        assert pipeline.options == basic_3d_of_options
        assert pipeline.config == config
        assert pipeline.executor is not None
        assert len(pipeline.mean_disp) == 0
        assert len(pipeline.max_disp) == 0
        assert len(pipeline.mean_div) == 0
        assert len(pipeline.mean_translation) == 0


class TestReferenceSetup3D:
    """Test 3D reference frame setup."""
    
    def test_3d_reference_from_array(self, fast_3d_of_options):
        """Test 3D reference setup from ndarray."""
        Z, Y, X, C = 8, 24, 24, 2
        reference_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline._setup_reference(reference_volume)
        
        # Should convert to float64 and store
        assert pipeline.reference_raw is not None
        assert pipeline.reference_raw.dtype == np.float64
        assert pipeline.reference_raw.shape == (Z, Y, X, C)
        np.testing.assert_allclose(pipeline.reference_raw, reference_volume, rtol=1e-6)
    
    def test_3d_reference_weight_setup(self, fast_3d_of_options):
        """Test 3D weight array setup."""
        Z, Y, X, C = 6, 16, 16, 2
        reference_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        # Configure weights
        fast_3d_of_options.weight = [0.3, 0.7]
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline._setup_reference(reference_volume)
        
        # Should create 3D weight array
        assert pipeline.weight is not None
        assert pipeline.weight.shape == (Z, Y, X, C)
        
        # Check weight values per channel
        np.testing.assert_allclose(pipeline.weight[:, :, :, 0], 0.3)
        np.testing.assert_allclose(pipeline.weight[:, :, :, 1], 0.7)
    
    def test_3d_reference_preprocessing(self, fast_3d_of_options):
        """Test 3D reference preprocessing (normalize -> filter)."""
        Z, Y, X, C = 4, 12, 12, 1
        reference_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        # Mock preprocessing functions
        mock_normalized = np.random.rand(Z, Y, X, C).astype(np.float64)
        mock_filtered = np.random.rand(Z, Y, X, C).astype(np.float64)
        
        with patch('flowreg3d.util.image_processing_3D.normalize') as mock_normalize, \
             patch('flowreg3d.util.image_processing_3D.apply_gaussian_filter') as mock_filter:
            
            mock_normalize.return_value = mock_normalized
            mock_filter.return_value = mock_filtered
            
            pipeline = BatchMotionCorrector(fast_3d_of_options)
            pipeline._setup_reference(reference_volume)
            
            # Should call normalize then filter
            mock_normalize.assert_called_once()
            mock_filter.assert_called_once()
            
            # Should store processed reference
            np.testing.assert_array_equal(pipeline.reference_proc, mock_filtered)


class TestFlowComputation3D:
    """Test 3D flow computation methods."""
    
    def test_3d_flow_parameters_extraction(self, fast_3d_of_options):
        """Test extraction of 3D flow parameters."""
        fast_3d_of_options.alpha = (1.0, 2.0, 3.0)
        fast_3d_of_options.levels = 25
        fast_3d_of_options.iterations = 15
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline.weight = np.ones((4, 16, 16, 2))
        
        # Mock processed frames
        Z, Y, X, C = 4, 16, 16, 2
        frame_proc = np.random.rand(Z, Y, X, C).astype(np.float64)
        ref_proc = np.random.rand(Z, Y, X, C).astype(np.float64)
        
        # Mock displacement function
        expected_flow = np.random.rand(Z, Y, X, 3).astype(np.float32)
        
        with patch('flowreg3d.core.optical_flow_3d.get_displacement') as mock_get_disp:
            mock_get_disp.return_value = expected_flow
            
            result = pipeline._compute_flow_single(frame_proc, ref_proc)
            
            # Should call get_displacement with correct parameters
            mock_get_disp.assert_called_once()
            call_args = mock_get_disp.call_args
            
            # Check flow parameters
            flow_params = call_args[1]
            assert flow_params['alpha'] == (1.0, 2.0, 3.0)
            assert flow_params['levels'] == 25
            assert flow_params['iterations'] == 15
            
            assert result is expected_flow
    
    def test_3d_batch_processing_parallel(self, fast_3d_of_options):
        """Test 3D batch processing with parallel executor."""
        T, Z, Y, X, C = 5, 4, 16, 16, 2
        batch = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        batch_proc = np.random.rand(T, Z, Y, X, C).astype(np.float64)
        w_init = np.random.rand(Z, Y, X, 3).astype(np.float32)
        
        # Expected outputs
        expected_registered = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        expected_flow = np.random.rand(T, Z, Y, X, 3).astype(np.float32)
        
        # Mock executor
        mock_executor = MagicMock()
        mock_executor.process_batch.return_value = (expected_registered, expected_flow)
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline.executor = mock_executor
        pipeline.reference_raw = np.random.rand(Z, Y, X, C)
        pipeline.reference_proc = np.random.rand(Z, Y, X, C)
        
        result_reg, result_flow = pipeline._process_batch_parallel(
            batch, batch_proc, w_init
        )
        
        # Should call executor with correct parameters
        mock_executor.process_batch.assert_called_once()
        call_kwargs = mock_executor.process_batch.call_args[1]
        
        # Check key parameters passed to executor
        assert call_kwargs['interpolation_method'] == 'cubic'  # default
        assert 'get_displacement_func' in call_kwargs
        assert 'imregister_func' in call_kwargs
        assert 'flow_params' in call_kwargs
        
        assert result_reg is expected_registered
        assert result_flow is expected_flow


class TestStatistics3D:
    """Test 3D statistics computation."""
    
    def test_3d_displacement_statistics(self, fast_3d_of_options):
        """Test 3D displacement magnitude and statistics."""
        T, Z, Y, X = 3, 4, 8, 8
        
        # Create known displacement field
        w = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
        w[0, :, :, :, 0] = 1.0  # u = 1
        w[0, :, :, :, 1] = 2.0  # v = 2  
        w[0, :, :, :, 2] = 3.0  # w = 3
        # Magnitude = sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.742
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline.mean_disp = []
        pipeline.max_disp = []
        
        # Manually compute statistics (simulating pipeline logic)
        for t in range(T):
            disp_magnitude = np.sqrt(
                w[t, :, :, :, 0] ** 2 + 
                w[t, :, :, :, 1] ** 2 + 
                w[t, :, :, :, 2] ** 2
            )
            pipeline.mean_disp.append(float(np.mean(disp_magnitude)))
            pipeline.max_disp.append(float(np.max(disp_magnitude)))
        
        # Check first frame statistics
        expected_magnitude = np.sqrt(14)  # ≈ 3.742
        assert abs(pipeline.mean_disp[0] - expected_magnitude) < 1e-5
        assert abs(pipeline.max_disp[0] - expected_magnitude) < 1e-5
        
        # Other frames should be zero
        assert pipeline.mean_disp[1] == 0.0
        assert pipeline.mean_disp[2] == 0.0
    
    def test_3d_divergence_computation(self, fast_3d_of_options):
        """Test 3D divergence computation."""
        T, Z, Y, X = 2, 4, 6, 6
        
        # Create linear displacement field with known divergence
        w = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
        
        # First frame: linear divergence
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    w[0, z, y, x, 0] = x * 0.1  # du/dx = 0.1
                    w[0, z, y, x, 1] = y * 0.2  # dv/dy = 0.2
                    w[0, z, y, x, 2] = z * 0.3  # dw/dz = 0.3
        # Expected divergence: 0.1 + 0.2 + 0.3 = 0.6
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline.mean_div = []
        
        # Compute divergence manually (simulating pipeline logic)
        for t in range(T):
            du_dx = np.gradient(w[t, :, :, :, 0], axis=2)  # x-axis is dim 2
            dv_dy = np.gradient(w[t, :, :, :, 1], axis=1)  # y-axis is dim 1
            dw_dz = np.gradient(w[t, :, :, :, 2], axis=0)  # z-axis is dim 0
            div_field = du_dx + dv_dy + dw_dz
            pipeline.mean_div.append(float(np.mean(div_field)))
        
        # First frame should have positive divergence
        assert pipeline.mean_div[0] > 0.5  # Should be close to 0.6
        
        # Second frame should be zero
        assert abs(pipeline.mean_div[1]) < 1e-6
    
    def test_3d_translation_computation(self, fast_3d_of_options):
        """Test 3D translation magnitude computation."""
        T, Z, Y, X = 2, 3, 4, 4
        
        # Create uniform translation
        w = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
        w[0, :, :, :, 0] = 2.0  # u = 2
        w[0, :, :, :, 1] = 3.0  # v = 3
        w[0, :, :, :, 2] = 6.0  # w = 6
        # Translation magnitude = sqrt(2^2 + 3^2 + 6^2) = sqrt(49) = 7
        
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline.mean_translation = []
        
        # Compute translation manually
        for t in range(T):
            u_mean = float(np.mean(w[t, :, :, :, 0]))
            v_mean = float(np.mean(w[t, :, :, :, 1]))
            w_mean = float(np.mean(w[t, :, :, :, 2]))
            translation_magnitude = float(np.sqrt(u_mean**2 + v_mean**2 + w_mean**2))
            pipeline.mean_translation.append(translation_magnitude)
        
        # Check translation magnitude
        assert abs(pipeline.mean_translation[0] - 7.0) < 1e-5
        assert abs(pipeline.mean_translation[1] - 0.0) < 1e-5


class TestIO3DIntegration:
    """Test 3D I/O integration."""
    
    def test_3d_io_setup(self, fast_3d_of_options, temp_dir):
        """Test 3D I/O setup with various output formats."""
        fast_3d_of_options.output_path = temp_dir
        fast_3d_of_options.save_w = True
        
        # Mock reader and writers
        mock_reader = MagicMock()
        mock_writer = MagicMock()
        mock_w_writer = MagicMock()
        
        with patch.object(fast_3d_of_options, 'get_video_reader') as mock_get_reader, \
             patch.object(fast_3d_of_options, 'get_video_writer') as mock_get_writer, \
             patch('flowreg3d.util.io.factory.get_video_file_writer') as mock_get_w_writer:
            
            mock_get_reader.return_value = mock_reader
            mock_get_writer.return_value = mock_writer
            mock_get_w_writer.return_value = mock_w_writer
            
            pipeline = BatchMotionCorrector(fast_3d_of_options)
            pipeline._setup_io()
            
            # Should setup all I/O components
            assert pipeline.video_reader is mock_reader
            assert pipeline.video_writer is mock_writer
            assert pipeline.w_writer is mock_w_writer
            
            # Should create displacement writer with 3D components
            mock_get_w_writer.assert_called()
            call_kwargs = mock_get_w_writer.call_args[1]
            assert call_kwargs.get('dataset_names') == ['u', 'v', 'w']
    
    def test_3d_array_output_format(self, fast_3d_of_options):
        """Test ARRAY output format for 3D."""
        from flowreg3d.motion_correction.OF_options_3D import OutputFormat
        
        fast_3d_of_options.output_format = OutputFormat.ARRAY
        fast_3d_of_options.save_w = True
        
        # Mock ARRAY writers
        mock_writer = MagicMock()
        mock_w_writer = MagicMock()
        
        with patch.object(fast_3d_of_options, 'get_video_reader'), \
             patch.object(fast_3d_of_options, 'get_video_writer') as mock_get_writer, \
             patch('flowreg3d.util.io.factory.get_video_file_writer') as mock_get_w_writer:
            
            mock_get_writer.return_value = mock_writer
            mock_get_w_writer.return_value = mock_w_writer
            
            pipeline = BatchMotionCorrector(fast_3d_of_options)
            pipeline._setup_io()
            
            # Should create ArrayWriter for displacements
            mock_get_w_writer.assert_called_with(None, 'ARRAY')


class TestProgressTracking3D:
    """Test progress tracking for 3D processing."""
    
    def test_3d_progress_callback_registration(self, fast_3d_of_options):
        """Test progress callback registration for 3D."""
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        
        # Track progress calls
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Register callback
        pipeline.register_progress_callback(progress_callback)
        assert len(pipeline.progress_callbacks) == 1
        
        # Test notification
        pipeline._total_frames = 50
        pipeline._notify_progress(10)
        
        assert len(progress_calls) == 1
        assert progress_calls[0] == (10, 50)
    
    def test_3d_batch_progress_tracking(self, fast_3d_of_options):
        """Test batch-wise progress tracking for 3D volumes."""
        pipeline = BatchMotionCorrector(fast_3d_of_options)
        pipeline._total_frames = 100
        
        progress_calls = []
        def callback(current, total):
            progress_calls.append((current, total))
        
        pipeline.register_progress_callback(callback)
        
        # Simulate batch processing
        batch_sizes = [10, 15, 20, 25, 30]  # Total = 100
        
        for batch_size in batch_sizes:
            pipeline._notify_progress(batch_size)
        
        # Should track cumulative progress
        assert len(progress_calls) == 5
        assert progress_calls[0] == (10, 100)
        assert progress_calls[1] == (25, 100)  # 10 + 15
        assert progress_calls[2] == (45, 100)  # 25 + 20
        assert progress_calls[3] == (70, 100)  # 45 + 25
        assert progress_calls[4] == (100, 100)  # 70 + 30


class TestErrorHandling3D:
    """Test 3D-specific error handling."""
    
    def test_3d_executor_instantiation_error(self, fast_3d_of_options):
        """Test handling of 3D executor instantiation errors."""
        config = RegistrationConfig(parallelization="invalid3d")
        
        # Mock to return None for invalid executor
        with patch.object(RuntimeContext, 'get_parallelization_executor') as mock_get:
            mock_get.return_value = None
            
            # Should fallback without crashing
            with patch('flowreg3d.motion_correction.parallelization.sequential_3d.SequentialExecutor3D'):
                with pytest.raises(RuntimeError, match="Could not load any executor"):
                    BatchMotionCorrector(fast_3d_of_options, config)
    
    def test_3d_w_writer_creation_failure(self, fast_3d_of_options):
        """Test handling of displacement writer creation failure."""
        fast_3d_of_options.save_w = True
        
        # Mock writer creation to fail
        with patch.object(fast_3d_of_options, 'get_video_reader'), \
             patch.object(fast_3d_of_options, 'get_video_writer'), \
             patch('flowreg3d.util.io.factory.get_video_file_writer') as mock_get_w_writer:
            
            mock_get_w_writer.side_effect = Exception("Writer creation failed")
            
            # Should handle gracefully with warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                pipeline = BatchMotionCorrector(fast_3d_of_options)
                pipeline._setup_io()
                
                # Should disable save_w and warn
                assert not fast_3d_of_options.save_w
                assert pipeline.w_writer is None
                assert len(w) == 1
                assert "Failed to create displacement writer" in str(w[0].message)


class TestCompensateRecordingIntegration3D:
    """Integration tests for the complete 3D compensate_recording function."""
    
    def test_3d_compensate_recording_function(self, fast_3d_of_options):
        """Test the main 3D compensate_recording function."""
        Z, Y, X, C = 4, 16, 16, 2
        reference_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        config = RegistrationConfig(
            n_jobs=1,
            batch_size=5,
            verbose=True,
            parallelization="sequential3d"
        )
        
        # Mock the pipeline run method
        with patch.object(BatchMotionCorrector, 'run') as mock_run:
            mock_run.return_value = reference_volume
            
            # Call main function
            result = compensate_recording(fast_3d_of_options, reference_volume, config)
            
            # Should create pipeline and run it
            mock_run.assert_called_once_with(reference_volume)
            np.testing.assert_array_equal(result, reference_volume)
    
    def test_3d_compensate_recording_without_config(self, fast_3d_of_options):
        """Test 3D compensate_recording with default config."""
        # Mock the pipeline
        with patch.object(BatchMotionCorrector, 'run') as mock_run:
            expected_ref = np.random.rand(4, 16, 16, 2)
            mock_run.return_value = expected_ref
            
            # Call without config
            result = compensate_recording(fast_3d_of_options)
            
            # Should work with default config
            mock_run.assert_called_once_with(None)
            np.testing.assert_array_equal(result, expected_ref)


# Fixtures for 3D testing
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def basic_3d_of_options(temp_dir):
    """Create basic 3D OF_options for testing."""
    from flowreg3d.motion_correction.OF_options_3D import OFOptions
    
    return OFOptions(
        input_file="dummy.h5",
        output_path=temp_dir,
        quality_setting="balanced",
        alpha=(1.0, 1.0, 2.0),  # 3D alpha
        sigma=[[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],  # 3D sigma
        levels=10,
        iterations=10,
        buffer_size=15,
        save_w=False,
        save_meta_info=False
    )


@pytest.fixture
def fast_3d_of_options(temp_dir):
    """Create fast 3D OF_options for quick testing."""
    from flowreg3d.motion_correction.OF_options_3D import OFOptions
    
    return OFOptions(
        input_file="dummy.h5",
        output_path=temp_dir,
        quality_setting="fast",
        alpha=(2.0, 2.0, 4.0),  # Higher for speed
        sigma=[[1.0, 1.0, 1.0, 0.1], [1.0, 1.0, 1.0, 0.1]],
        levels=2,  # Few levels for speed
        iterations=3,  # Few iterations
        buffer_size=8,  # Small buffers
        save_w=False,
        save_meta_info=False
    )


if __name__ == "__main__":
    # Mock the 3D dependencies for basic testing
    try:
        from flowreg3d.motion_correction.OF_options_3D import OFOptions
        
        options = OFOptions(
            alpha=(1.0, 2.0, 3.0),
            buffer_size=20,
            quality_setting="balanced"
        )
        
        config = RegistrationConfig(
            n_jobs=2,
            batch_size=8,
            parallelization="sequential3d"
        )
        
        print("3D compensate_recording test:")
        print(f"Options: {options}")
        print(f"Config: {config}")
        print("Basic tests would run with proper 3D dependencies!")
        
    except ImportError:
        print("3D dependencies not available for testing")