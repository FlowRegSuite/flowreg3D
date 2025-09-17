"""
Pytest configuration and fixtures for PyFlowReg tests.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Tuple

import pytest
import numpy as np

from tests.fixtures_3d import (
    create_test_video_hdf5,
    create_simple_test_data,
    get_minimal_of_options,
    cleanup_temp_files,
    create_test_3d_video_hdf5,
    create_simple_3d_test_data,
    create_3d_reference_volume,
    get_minimal_3d_of_options
)
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig
from pyflowreg._runtime import RuntimeContext

# Try to import 3D-specific components
try:
    from flowreg3d.motion_correction.compensate_recording_3D import RegistrationConfig as RegistrationConfig3D
    from flowreg3d._runtime import RuntimeContext as RuntimeContext3D
    HAS_3D_SUPPORT = True
except ImportError:
    RegistrationConfig3D = RegistrationConfig
    RuntimeContext3D = RuntimeContext
    HAS_3D_SUPPORT = False


@pytest.fixture(scope="session", autouse=True)
def initialize_runtime_context():
    """Initialize RuntimeContext for all tests."""
    RuntimeContext.init(force=True)
    
    # Import parallelization module to trigger executor registration
    import pyflowreg.motion_correction.parallelization
    
    # Initialize 3D runtime context if available
    if HAS_3D_SUPPORT:
        try:
            RuntimeContext3D.init(force=True)
            import flowreg3d.motion_correction.parallelization
        except ImportError:
            pass


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def small_test_video(temp_dir):
    """Create a small test video file for quick testing."""
    shape = (10, 16, 32, 2)  # Small size for fast tests
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "small_test.h5"),
        pattern="motion",
        noise_level=0.05
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def medium_test_video(temp_dir):
    """Create a medium-sized test video for more comprehensive testing."""
    shape = (50, 32, 64, 2)  # Original requested size
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "medium_test.h5"),
        pattern="motion",
        noise_level=0.1
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def static_test_video(temp_dir):
    """Create a static test video for baseline testing."""
    shape = (20, 16, 32, 2)
    video_path = create_test_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "static_test.h5"),
        pattern="static",
        noise_level=0.02
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def test_data_array():
    """Create test data as numpy array without file I/O."""
    shape = (20, 16, 32, 2)
    data = create_simple_test_data(shape)
    return data, shape


@pytest.fixture(scope="function")
def basic_of_options(temp_dir):
    """Create basic OF_options for testing."""
    options = get_minimal_of_options()
    options.output_path = temp_dir
    return options


@pytest.fixture(scope="function")
def fast_of_options(temp_dir):
    """Create very fast OF_options for quick testing."""
    options = get_minimal_of_options()
    options.output_path = temp_dir
    options.levels = 1  # Single level for speed
    options.iterations = 2  # Minimal iterations
    options.alpha = 50.0  # Lower regularization for speed
    return options


@pytest.fixture(scope="function")
def sequential_config():
    """Create configuration for sequential executor."""
    return RegistrationConfig(
        n_jobs=1,
        batch_size=10,
        verbose=True,
        parallelization="sequential"
    )


@pytest.fixture(scope="function")
def threading_config():
    """Create configuration for threading executor."""
    return RegistrationConfig(
        n_jobs=2,
        batch_size=10,
        verbose=True,
        parallelization="threading"
    )


@pytest.fixture(scope="function")
def multiprocessing_config():
    """Create configuration for multiprocessing executor."""
    return RegistrationConfig(
        n_jobs=2,
        batch_size=10,
        verbose=True,
        parallelization="multiprocessing"
    )


@pytest.fixture(scope="function")
def auto_config():
    """Create configuration with auto-selection of executor."""
    return RegistrationConfig(
        n_jobs=2,
        batch_size=10,
        verbose=True,
        parallelization=None  # Auto-select
    )


@pytest.fixture(params=["sequential", "threading", "multiprocessing"])
def executor_config(request):
    """Parametrized fixture to test all executor types."""
    return RegistrationConfig(
        n_jobs=2,
        batch_size=5,
        verbose=True,
        parallelization=request.param
    )


@pytest.fixture(scope="function")
def reference_frame():
    """Create a simple reference frame for testing."""
    H, W, C = 16, 32, 2
    ref = np.zeros((H, W, C), dtype=np.float32)
    
    # Add some structure
    center_y, center_x = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    
    for c in range(C):
        # Create circular pattern
        radius = min(H, W) // 4
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        ref[:, :, c] = mask.astype(np.float32) * 0.8 + 0.2
    
    return ref


@pytest.fixture(scope="session")
def available_executors():
    """Get list of available executors for testing."""
    return list(RuntimeContext.get_available_parallelization())


# ========================
# 3D-specific fixtures
# ========================

@pytest.fixture(scope="function")
def small_3d_test_video(temp_dir):
    """Create a small 3D test video file for quick testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    shape = (8, 4, 16, 16, 2)  # Small 3D size for fast tests
    video_path = create_test_3d_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "small_3d_test.h5"),
        pattern="motion",
        noise_level=0.05
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def medium_3d_test_video(temp_dir):
    """Create a medium-sized 3D test video for more comprehensive testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    shape = (20, 8, 32, 32, 2)  # Medium 3D size
    video_path = create_test_3d_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "medium_3d_test.h5"),
        pattern="drift",
        noise_level=0.1
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def static_3d_test_video(temp_dir):
    """Create a static 3D test video for baseline testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    shape = (12, 6, 20, 20, 2)
    video_path = create_test_3d_video_hdf5(
        shape=shape,
        output_path=str(Path(temp_dir) / "static_3d_test.h5"),
        pattern="static",
        noise_level=0.02
    )
    yield video_path, shape
    cleanup_temp_files(video_path)


@pytest.fixture(scope="function")
def test_3d_data_array():
    """Create 3D test data as numpy array without file I/O."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    shape = (15, 6, 24, 24, 2)
    data = create_simple_3d_test_data(shape)
    return data, shape


@pytest.fixture(scope="function")
def basic_3d_of_options(temp_dir):
    """Create basic 3D OF_options for testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    options = get_minimal_3d_of_options()
    options.output_path = temp_dir
    return options


@pytest.fixture(scope="function")
def fast_3d_of_options(temp_dir):
    """Create very fast 3D OF_options for quick testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    options = get_minimal_3d_of_options()
    options.output_path = temp_dir
    options.levels = 2  # Very few levels for speed
    options.iterations = 3  # Minimal iterations
    options.alpha = (2.0, 2.0, 4.0)  # Higher regularization for speed
    options.buffer_size = 5  # Smaller buffers for testing
    return options


@pytest.fixture(scope="function")
def sequential_3d_config():
    """Create configuration for sequential 3D executor."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    return RegistrationConfig3D(
        n_jobs=1,
        batch_size=5,  # Small batches for 3D
        verbose=True,
        parallelization="sequential3d"
    )


@pytest.fixture(scope="function")
def threading_3d_config():
    """Create configuration for threading 3D executor."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    return RegistrationConfig3D(
        n_jobs=2,
        batch_size=4,
        verbose=True,
        parallelization="threading3d"
    )


@pytest.fixture(scope="function")
def multiprocessing_3d_config():
    """Create configuration for multiprocessing 3D executor."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    return RegistrationConfig3D(
        n_jobs=2,
        batch_size=4,
        verbose=True,
        parallelization="multiprocessing3d"
    )


@pytest.fixture(params=["sequential3d", "threading3d", "multiprocessing3d"])
def executor_3d_config(request):
    """Parametrized fixture to test all 3D executor types."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    return RegistrationConfig3D(
        n_jobs=2,
        batch_size=3,
        verbose=True,
        parallelization=request.param
    )


@pytest.fixture(scope="function")
def reference_3d_volume():
    """Create a simple 3D reference volume for testing."""
    if not HAS_3D_SUPPORT:
        pytest.skip("3D support not available")
    
    Z, Y, X, C = 6, 20, 20, 2
    ref = create_3d_reference_volume((Z, Y, X, C))
    return ref


@pytest.fixture(scope="session")
def available_3d_executors():
    """Get list of available 3D executors for testing."""
    if not HAS_3D_SUPPORT:
        return []
    
    try:
        return list(RuntimeContext3D.get_available_parallelization())
    except:
        return ['sequential3d']  # Fallback


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "executor: marks tests that test specific executors"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark slow tests
        if "large" in item.name or "comprehensive" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark executor tests
        if "executor" in item.name or item.name.startswith("test_compensate"):
            item.add_marker(pytest.mark.executor)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default for most tests)
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)