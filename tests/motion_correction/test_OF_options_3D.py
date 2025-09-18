"""
Tests for OF_options_3D parameter validation and flow integration.

Tests the 3D optical flow options configuration, parameter normalization,
and integration with the motion correction pipeline.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import tifffile

from flowreg3d.motion_correction.OF_options_3D import (
    OFOptions, 
    OutputFormat, 
    QualitySetting, 
    ChannelNormalization, 
    InterpolationMethod,
    ConstancyAssumption,
    NamingConvention
)


class TestOFOptions3DBasics:
    """Test basic functionality of OFOptions for 3D."""
    
    def test_default_3d_options(self):
        """Test default 3D configuration values."""
        options = OFOptions()
        
        # Check 3D-specific defaults
        assert options.alpha == (0.25, 0.25, 0.25)  # 3-tuple for z,y,x
        assert options.buffer_size == 10  # Volume buffer size
        assert options.min_level == 5  # Default for 3D
        
        # Check 3D sigma format: [[sx, sy, sz, st], [sx, sy, sz, st]]
        expected_sigma = [[1.0, 1.0, 1.0, 0.1], [1.0, 1.0, 1.0, 0.1]]
        assert options.sigma == expected_sigma
        
        # Check reference frames for 3D volumes
        assert options.reference_frames == list(range(50, 500))
        
    def test_custom_3d_options(self):
        """Test custom 3D configuration values."""
        alpha_3d = (1.0, 2.0, 3.0)  # Different values for z, y, x
        sigma_3d = [[2.0, 2.0, 1.5, 0.2], [1.5, 1.5, 1.0, 0.15]]

        options = OFOptions(
            alpha=alpha_3d,
            sigma=sigma_3d,
            buffer_size=20,
            min_level=3,
            quality_setting=QualitySetting.BALANCED
        )

        assert options.alpha == alpha_3d
        assert options.sigma == sigma_3d
        assert options.buffer_size == 20
        assert options.min_level == 3
        # Setting min_level >= 0 forces CUSTOM quality_setting
        assert options.quality_setting == QualitySetting.CUSTOM
    
    def test_3d_quality_settings(self):
        """Test quality setting effects for 3D."""
        quality_levels = [
            (QualitySetting.FAST, 6),
            (QualitySetting.BALANCED, 4),
            (QualitySetting.QUALITY, 0),
        ]
        
        for quality, expected_min_level in quality_levels:
            options = OFOptions(quality_setting=quality, min_level=-1)
            assert options.effective_min_level == expected_min_level


class TestAlpha3DValidation:
    """Test alpha parameter validation for 3D."""
    
    def test_alpha_scalar_to_3tuple(self):
        """Test scalar alpha conversion to 3-tuple."""
        options = OFOptions(alpha=2.5)
        assert options.alpha == (2.5, 2.5, 2.5)
    
    def test_alpha_2tuple_to_3tuple(self):
        """Test 2-tuple alpha extension to 3-tuple."""
        options = OFOptions(alpha=(1.0, 2.0))
        # Should extend as (z, y, x) = (y, y, x) for 2D->3D compatibility
        assert options.alpha == (1.0, 1.0, 2.0)
    
    def test_alpha_3tuple_preserved(self):
        """Test 3-tuple alpha preservation."""
        alpha_3d = (1.0, 2.0, 3.0)
        options = OFOptions(alpha=alpha_3d)
        assert options.alpha == alpha_3d
    
    def test_alpha_validation_positive(self):
        """Test alpha validation requires positive values."""
        with pytest.raises(ValueError, match="Alpha must be positive"):
            OFOptions(alpha=-1.0)
        
        with pytest.raises(ValueError, match="All alpha values must be positive"):
            OFOptions(alpha=(1.0, -2.0, 3.0))
    
    def test_alpha_validation_wrong_length(self):
        """Test alpha validation rejects wrong lengths."""
        with pytest.raises(ValueError, match="Alpha must be scalar, 2-element, or 3-element tuple"):
            OFOptions(alpha=(1.0, 2.0, 3.0, 4.0))


class TestSigma3DValidation:
    """Test sigma parameter validation for 3D."""
    
    def test_sigma_1d_3element_to_4element(self):
        """Test 1D 3-element sigma conversion to 4-element."""
        options = OFOptions(sigma=[1.0, 2.0, 0.5])  # [sx, sy, st] -> [sx, sy, sz=1.0, st]
        expected = [[1.0, 2.0, 1.0, 0.5]]
        assert options.sigma == expected
    
    def test_sigma_1d_4element_preserved(self):
        """Test 1D 4-element sigma preservation."""
        sigma_4d = [2.0, 2.5, 1.5, 0.3]
        options = OFOptions(sigma=sigma_4d)
        expected = [sigma_4d]  # Wrapped in list for per-channel format
        assert options.sigma == expected
    
    def test_sigma_2d_3to4_conversion(self):
        """Test 2D sigma conversion from 3 to 4 elements per channel."""
        sigma_2d_3elem = [[1.0, 1.5, 0.2], [2.0, 2.5, 0.3]]  # [sx, sy, st] per channel
        options = OFOptions(sigma=sigma_2d_3elem)
        expected = [[1.0, 1.5, 1.0, 0.2], [2.0, 2.5, 1.0, 0.3]]  # Insert sz=1.0
        assert options.sigma == expected
    
    def test_sigma_2d_4element_preserved(self):
        """Test 2D 4-element sigma preservation."""
        sigma_4d = [[1.0, 1.5, 2.0, 0.2], [2.0, 2.5, 1.8, 0.3]]
        options = OFOptions(sigma=sigma_4d)
        assert options.sigma == sigma_4d
    
    def test_sigma_validation_wrong_size(self):
        """Test sigma validation rejects wrong sizes."""
        with pytest.raises(ValueError, match="1D sigma must be.*for 3D"):
            OFOptions(sigma=[1.0, 2.0])  # Only 2 elements
        
        with pytest.raises(ValueError, match="2D sigma must be.*for 3D"):
            OFOptions(sigma=[[1.0, 2.0], [1.5, 2.5]])  # Only 2 elements per channel


class TestWeight3DHandling:
    """Test weight parameter handling for 3D."""
    
    def test_weight_normalization(self):
        """Test weight normalization to sum to 1."""
        options = OFOptions(weight=[2.0, 3.0])
        expected = [0.4, 0.6]  # 2/5, 3/5
        assert options.weight == expected
    
    def test_weight_numpy_array_handling(self):
        """Test weight handling with numpy arrays."""
        weight_array = np.array([1.0, 3.0, 2.0])
        options = OFOptions(weight=weight_array)
        expected = [1/6, 3/6, 2/6]  # Normalized to sum to 1
        np.testing.assert_allclose(options.weight, expected)
    
    def test_get_weight_at_3d(self):
        """Test get_weight_at method for 3D."""
        options = OFOptions(weight=[0.3, 0.7])
        
        # Test existing channels
        assert options.get_weight_at(0, 2) == 0.3
        assert options.get_weight_at(1, 2) == 0.7
        
        # Test missing channel (should default to 1/n_channels)
        assert options.get_weight_at(2, 3) == 1.0/3.0


class TestSigmaAccess3D:
    """Test sigma access methods for 3D."""
    
    def test_get_sigma_at_single_channel(self):
        """Test get_sigma_at with single sigma for all channels."""
        sigma_1d = [2.0, 2.5, 1.5, 0.3]
        options = OFOptions(sigma=sigma_1d)
        
        # Should return same sigma for all channels
        result = options.get_sigma_at(0)
        np.testing.assert_array_equal(result, sigma_1d)
        
        result = options.get_sigma_at(1)
        np.testing.assert_array_equal(result, sigma_1d)
    
    def test_get_sigma_at_per_channel(self):
        """Test get_sigma_at with per-channel sigma."""
        sigma_2d = [[1.0, 1.5, 2.0, 0.2], [2.0, 2.5, 1.8, 0.3]]
        options = OFOptions(sigma=sigma_2d)
        
        # Should return specific sigma for each channel
        np.testing.assert_array_equal(options.get_sigma_at(0), sigma_2d[0])
        np.testing.assert_array_equal(options.get_sigma_at(1), sigma_2d[1])
    
    def test_get_sigma_at_missing_channel(self):
        """Test get_sigma_at with missing channel (should use channel 0)."""
        sigma_2d = [[1.0, 1.5, 2.0, 0.2], [2.0, 2.5, 1.8, 0.3]]
        options = OFOptions(sigma=sigma_2d, verbose=True)
        
        # Missing channel should use channel 0
        result = options.get_sigma_at(5)
        np.testing.assert_array_equal(result, sigma_2d[0])


class TestReference3DHandling:
    """Test reference frame handling for 3D."""
    
    def test_reference_ndarray_3d(self):
        """Test reference frame as 3D ndarray."""
        Z, Y, X, C = 10, 32, 32, 2
        ref_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        options = OFOptions(reference_frames=ref_volume)
        
        # Should return the same array
        result = options.get_reference_frame()
        np.testing.assert_array_equal(result, ref_volume)
    
    def test_reference_tiff_file_3d(self, temp_dir):
        """Test reference frame from 3D TIFF file."""
        Z, Y, X, C = 8, 24, 24, 2
        ref_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        # Save as TIFF
        tiff_path = Path(temp_dir) / "reference_3d.tif"
        tifffile.imwrite(str(tiff_path), ref_volume)
        
        options = OFOptions(reference_frames=str(tiff_path))
        
        # Should load the same volume
        result = options.get_reference_frame()
        np.testing.assert_allclose(result, ref_volume, rtol=1e-6)
    
    def test_reference_frame_indices_3d(self):
        """Test reference frame computation from frame indices."""
        # Create mock video reader
        Z, Y, X, C = 6, 16, 16, 2
        T = 20
        mock_frames = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        
        mock_reader = MagicMock()
        mock_reader.__getitem__.return_value = mock_frames[:10]  # First 10 frames
        
        options = OFOptions(
            reference_frames=list(range(10)),
            verbose=False
        )
        
        # Mock the 3D compensation function
        with patch('flowreg3d.motion_correction.compensate_arr_3D.compensate_arr_3D') as mock_compensate:
            expected_ref = np.mean(mock_frames[:10], axis=0)
            mock_compensate.return_value = (mock_frames[:10], None)
            
            result = options.get_reference_frame(mock_reader)
            
            # Should call compensate_arr_3D for preregistration
            mock_compensate.assert_called_once()
            
            # Result should be mean of compensated frames
            assert result.shape == (Z, Y, X, C)


class TestParameterFlow3D:
    """Test parameter flow to 3D pipeline components."""
    
    def test_to_dict_3d_parameters(self):
        """Test to_dict method produces correct 3D parameters."""
        alpha_3d = (1.0, 2.0, 3.0)
        options = OFOptions(
            alpha=alpha_3d,
            weight=[0.4, 0.6],
            levels=50,
            min_level=3,
            eta=0.85,
            iterations=20,
            update_lag=8,
            a_data=0.5,
            a_smooth=1.2,
            constancy_assumption=ConstancyAssumption.GRADIENT
        )
        
        params = options.to_dict()
        
        # Check all parameters are correctly passed
        assert params['alpha'] == alpha_3d
        assert params['weight'] == [0.4, 0.6]
        assert params['levels'] == 50
        assert params['min_level'] == 3
        assert params['eta'] == 0.85
        assert params['iterations'] == 20
        assert params['update_lag'] == 8
        assert params['a_data'] == 0.5
        assert params['a_smooth'] == 1.2
        assert params['const_assumption'] == 'gc'
    
    def test_effective_min_level_quality_mapping(self):
        """Test effective_min_level mapping for different quality settings."""
        test_cases = [
            (QualitySetting.FAST, -1, 6),
            (QualitySetting.BALANCED, -1, 4), 
            (QualitySetting.QUALITY, -1, 0),
            (QualitySetting.CUSTOM, 5, 5),  # Custom uses explicit min_level
        ]
        
        for quality, min_level, expected in test_cases:
            options = OFOptions(quality_setting=quality, min_level=min_level)
            assert options.effective_min_level == expected


class TestIO3DIntegration:
    """Test I/O integration for 3D data."""
    
    def test_get_video_reader_3d(self):
        """Test video reader creation for 3D."""
        # Create test 3D array
        T, Z, Y, X, C = 5, 8, 32, 32, 2
        test_volume = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        
        options = OFOptions(
            input_file=test_volume,
            buffer_size=15,
            bin_size=1
        )
        
        # Mock the factory function
        mock_reader = MagicMock()
        with patch('flowreg3d.util.io.factory.get_video_file_reader') as mock_factory:
            mock_factory.return_value = mock_reader
            
            reader = options.get_video_reader()
            
            # Should call factory with correct parameters
            mock_factory.assert_called_once_with(
                test_volume,
                buffer_size=15,
                bin_size=1,
                dim_order='TZYX'
            )
            assert reader == mock_reader
            
            # Should cache the reader
            assert options.input_file == mock_reader
    
    def test_get_video_writer_3d_formats(self, temp_dir):
        """Test video writer creation for different 3D formats."""
        options = OFOptions(output_path=temp_dir)
        
        # Mock the factory function
        mock_writer = MagicMock()
        with patch('flowreg3d.util.io.factory.get_video_file_writer') as mock_factory:
            mock_factory.return_value = mock_writer
            
            # Test different output formats
            formats = [OutputFormat.TIFF, OutputFormat.HDF5, OutputFormat.ARRAY]
            
            for fmt in formats:
                options.output_format = fmt
                options._video_writer = None  # Reset cache
                
                writer = options.get_video_writer()
                
                mock_factory.assert_called()
                assert writer == mock_writer


class TestSaveLoad3D:
    """Test save/load functionality for 3D options."""
    
    def test_save_options_3d(self, temp_dir):
        """Test saving 3D options to JSON."""
        alpha_3d = (1.5, 2.0, 2.5)
        sigma_3d = [[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]]
        
        options = OFOptions(
            alpha=alpha_3d,
            sigma=sigma_3d,
            buffer_size=25,
            output_path=temp_dir,
            verbose=True
        )
        
        json_path = Path(temp_dir) / "options_3d.json"
        options.save_options(json_path)
        
        # Check file was created
        assert json_path.exists()
        
        # Check contents
        with json_path.open() as f:
            lines = f.readlines()
            
        # Should have header
        assert lines[0].startswith("Compensation options")
        
        # Parse JSON content
        json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
        json_data = json.loads("".join(lines[json_start:]))
        
        assert json_data['alpha'] == list(alpha_3d)
        assert json_data['sigma'] == sigma_3d
        assert json_data['buffer_size'] == 25
    
    def test_load_options_3d(self, temp_dir):
        """Test loading 3D options from JSON."""
        # Create JSON file
        json_path = Path(temp_dir) / "test_options.json"
        
        options_data = {
            "alpha": [1.5, 2.0, 2.5],
            "sigma": [[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],
            "buffer_size": 25,
            "quality_setting": "balanced",
            "output_path": str(temp_dir)
        }
        
        with json_path.open("w") as f:
            f.write("Compensation options 2024-01-01\n\n")
            json.dump(options_data, f, indent=2)
        
        # Load options
        loaded = OFOptions.load_options(json_path)
        
        assert loaded.alpha == (1.5, 2.0, 2.5)
        assert loaded.sigma == [[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]]
        assert loaded.buffer_size == 25
        assert loaded.quality_setting == QualitySetting.BALANCED
    
    def test_save_load_with_3d_reference(self, temp_dir):
        """Test save/load with 3D reference volume."""
        Z, Y, X, C = 6, 24, 24, 2
        ref_volume = np.random.rand(Z, Y, X, C).astype(np.float32)
        
        options = OFOptions(
            reference_frames=ref_volume,
            output_path=temp_dir
        )
        
        json_path = Path(temp_dir) / "options_with_ref.json"
        options.save_options(json_path)
        
        # Should save reference as separate TIFF file
        ref_tiff = Path(temp_dir) / "reference_frames.tif"
        assert ref_tiff.exists()
        
        # Load and check
        loaded = OFOptions.load_options(json_path)
        loaded_ref = loaded.get_reference_frame()
        
        np.testing.assert_allclose(loaded_ref, ref_volume, rtol=1e-6)


class TestConvenienceFunction3D:
    """Test convenience functions for 3D."""
    
    def test_compensate_inplace_3d(self):
        """Test compensate_inplace convenience function for 3D."""
        T, Z, Y, X, C = 5, 6, 16, 16, 2
        frames = np.random.rand(T, Z, Y, X, C).astype(np.float32)
        reference = np.random.rand(Z, Y, X, C).astype(np.float32)

        # Mock the 3D functions
        mock_displacement = np.random.rand(Z, Y, X, 3).astype(np.float32)
        mock_compensated = np.random.rand(T, Z, Y, X, C).astype(np.float32)

        with patch('flowreg3d.get_displacement') as mock_get_disp, \
             patch('flowreg3d.motion_correction.compensate_arr_3D.compensate_arr_3D') as mock_compensate:

            mock_get_disp.return_value = mock_displacement
            mock_compensate.return_value = mock_compensated

            from flowreg3d.motion_correction.OF_options_3D import compensate_inplace

            options = OFOptions(alpha=(1.0, 2.0, 3.0))
            result_comp, result_disp = compensate_inplace(frames, reference, options)

            # Should call get_displacement for each timepoint
            assert mock_get_disp.call_count == T
            mock_compensate.assert_called_once()
    
    def test_get_mcp_schema(self):
        """Test JSON schema generation."""
        from flowreg3d.motion_correction.OF_options_3D import get_mcp_schema
        
        schema = get_mcp_schema()
        
        # Should be a valid JSON schema dict
        assert isinstance(schema, dict)
        assert 'properties' in schema
        assert 'title' in schema
        
        # Should include 3D-specific properties
        properties = schema['properties']
        assert 'alpha' in properties
        assert 'sigma' in properties
        assert 'buffer_size' in properties


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Test basic functionality
    opts = OFOptions(
        alpha=(1.0, 2.0, 3.0),
        sigma=[[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],
        quality_setting=QualitySetting.BALANCED,
        buffer_size=20
    )
    
    print("3D OF_options test:")
    print(f"Alpha: {opts.alpha}")
    print(f"Sigma: {opts.sigma}")
    print(f"Effective min_level: {opts.effective_min_level}")
    print(f"Parameters dict: {opts.to_dict()}")
    print("Basic tests passed!")