"""
Test fixtures and utilities for 3D FlowReg tests.

Provides 3D-specific test data generation, fixtures, and utilities
for testing the 3D motion correction pipeline.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np


def create_test_3d_video_hdf5(
    shape: Tuple[int, int, int, int, int] = (20, 8, 32, 32, 2),
    output_path: str = None,
    pattern: str = "motion",
    noise_level: float = 0.1
) -> str:
    """
    Create a test HDF5 3D video file with simulated motion.
    
    Args:
        shape: Video shape in TxZxYxXxC format (T=time, Z=depth, Y=height, X=width, C=channels)
        output_path: Output file path. If None, creates temporary file.
        pattern: Motion pattern ('motion', 'static', 'random', 'drift')
        noise_level: Amount of noise to add (0.0 to 1.0)
        
    Returns:
        Path to created HDF5 file
    """
    T, Z, Y, X, C = shape
    
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = str(Path(temp_dir) / "test_3d_video.h5")
    
    # Generate 3D test frames based on pattern
    if pattern == "motion":
        # Create frames with simulated 3D translational motion
        frames = np.zeros(shape, dtype=np.float32)
        
        # Create a 3D object to move
        center_z, center_y, center_x = Z // 2, Y // 2, X // 2
        radius = min(Z, Y, X) // 6
        
        for t in range(T):
            for c in range(C):
                # Simulate 3D motion - helical pattern
                offset_x = int(radius * 0.3 * np.cos(2 * np.pi * t / T))
                offset_y = int(radius * 0.3 * np.sin(2 * np.pi * t / T))
                offset_z = int(radius * 0.2 * np.sin(4 * np.pi * t / T))  # Faster Z oscillation
                
                # Create 3D ellipsoidal object
                z, y, x = np.ogrid[:Z, :Y, :X]
                obj_z = center_z + offset_z
                obj_y = center_y + offset_y
                obj_x = center_x + offset_x
                
                # 3D ellipsoid mask
                mask = ((x - obj_x)**2 / radius**2 + 
                       (y - obj_y)**2 / radius**2 + 
                       (z - obj_z)**2 / (radius*0.7)**2) <= 1.0
                frames[t, :, :, :, c] = mask.astype(np.float32) * 0.8
                
                # Add 3D background texture
                frames[t, :, :, :, c] += 0.2 * np.random.random((Z, Y, X))
                
                # Add noise
                if noise_level > 0:
                    frames[t, :, :, :, c] += noise_level * np.random.random((Z, Y, X))
    
    elif pattern == "drift":
        # Create frames with consistent 3D drift
        frames = np.zeros(shape, dtype=np.float32)
        
        # Create multiple 3D objects that drift together
        for t in range(T):
            for c in range(C):
                # Linear drift in all directions
                drift_x = t * 1.5  # 1.5 pixels per frame
                drift_y = t * 1.0  # 1.0 pixels per frame  
                drift_z = t * 0.5  # 0.5 pixels per frame
                
                # Create pattern with multiple objects
                z, y, x = np.mgrid[:Z, :Y, :X]
                
                # Object 1 - sphere
                center1_z, center1_y, center1_x = Z//4, Y//4, X//4
                sphere1 = ((x - center1_x - drift_x)**2 + 
                          (y - center1_y - drift_y)**2 + 
                          (z - center1_z - drift_z)**2) <= (min(Z,Y,X)//8)**2
                
                # Object 2 - sphere
                center2_z, center2_y, center2_x = 3*Z//4, 3*Y//4, 3*X//4
                sphere2 = ((x - center2_x - drift_x)**2 + 
                          (y - center2_y - drift_y)**2 + 
                          (z - center2_z - drift_z)**2) <= (min(Z,Y,X)//10)**2
                
                frames[t, :, :, :, c] = (sphere1 * 0.8 + sphere2 * 0.6).astype(np.float32)
                
                # Add noise
                if noise_level > 0:
                    frames[t, :, :, :, c] += noise_level * np.random.random((Z, Y, X))
                    
    elif pattern == "static":
        # Create static 3D frames with just noise
        frames = np.ones(shape, dtype=np.float32) * 0.5
        if noise_level > 0:
            frames += noise_level * np.random.random(shape)
            
    elif pattern == "random":
        # Create random 3D frames
        frames = np.random.random(shape).astype(np.float32)
        
    else:
        # Default zeros
        frames = np.zeros(shape, dtype=np.float32)
    
    # Ensure values are in [0, 1] range
    frames = np.clip(frames, 0.0, 1.0)
    
    # Convert to uint16 for more realistic microscopy data
    frames = (frames * 65535).astype(np.uint16)
    
    # Write using 3D video writer
    try:
        from flowreg3d.util.io.factory import get_video_file_writer
        writer = get_video_file_writer(output_path, 'HDF5')
    except ImportError:
        # Fallback to standard HDF5 writer if 3D not available
        from pyflowreg.util.io.factory import get_video_file_writer
        writer = get_video_file_writer(output_path, 'HDF5')
    
    try:
        # Write frames in batches to simulate real usage
        batch_size = 8  # Smaller batches for 3D
        for start_idx in range(0, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            batch = frames[start_idx:end_idx]
            writer.write_frames(batch)
    finally:
        writer.close()
    
    return output_path


def create_simple_3d_test_data(shape: Tuple[int, int, int, int, int] = (20, 8, 32, 32, 2)) -> np.ndarray:
    """
    Create simple 3D test data as numpy array without file I/O.
    
    Args:
        shape: Data shape in TxZxYxXxC format
        
    Returns:
        3D test data array
    """
    T, Z, Y, X, C = shape
    
    # Create frames with a simple moving 3D pattern
    frames = np.zeros(shape, dtype=np.float32)
    
    for t in range(T):
        for c in range(C):
            # Simple 3D gradient that shifts over time
            z, y, x = np.mgrid[:Z, :Y, :X]
            
            shift_x = t * 1.5  # Pixels shift per frame in X
            shift_y = t * 1.0  # Pixels shift per frame in Y
            shift_z = t * 0.5  # Pixels shift per frame in Z
            
            # 3D sinusoidal pattern
            pattern_x = np.sin((x + shift_x) * 2 * np.pi / X)
            pattern_y = np.sin((y + shift_y) * 2 * np.pi / Y) 
            pattern_z = np.sin((z + shift_z) * 2 * np.pi / Z)
            
            # Combine patterns
            pattern_3d = (pattern_x * pattern_y * pattern_z) * 0.5 + 0.5
            frames[t, :, :, :, c] = pattern_3d
    
    return frames


def create_3d_reference_volume(shape: Tuple[int, int, int, int] = (8, 32, 32, 2)) -> np.ndarray:
    """
    Create a 3D reference volume for testing.
    
    Args:
        shape: Reference shape in ZxYxXxC format
        
    Returns:
        3D reference volume
    """
    Z, Y, X, C = shape
    ref = np.zeros(shape, dtype=np.float32)
    
    # Create structured 3D pattern
    center_z, center_y, center_x = Z // 2, Y // 2, X // 2
    z, y, x = np.ogrid[:Z, :Y, :X]
    
    for c in range(C):
        # Create 3D Gaussian-like pattern
        radius_z = Z // 3
        radius_y = Y // 4  
        radius_x = X // 4
        
        # 3D ellipsoidal pattern
        pattern = np.exp(-((x - center_x)**2 / radius_x**2 + 
                          (y - center_y)**2 / radius_y**2 + 
                          (z - center_z)**2 / radius_z**2))
        
        ref[:, :, :, c] = pattern * (0.5 + c * 0.2) + 0.2
    
    return ref


def get_minimal_3d_of_options():
    """
    Create minimal 3D OF_options for testing.
    
    Returns:
        Basic 3D OFOptions configuration suitable for testing
    """
    try:
        from flowreg3d.motion_correction.OF_options_3D import OFOptions
    except ImportError:
        # Fallback if 3D version not available
        from pyflowreg.motion_correction.OF_options import OFOptions
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    
    return OFOptions(
        input_file="dummy_3d.h5",  # Will be overridden in tests
        output_path=temp_dir,
        quality_setting="fast",  # Use fast for testing
        alpha=(1.0, 1.0, 2.0),  # 3D regularization (z, y, x)
        levels=3,  # Fewer levels for faster testing
        iterations=5,  # Fewer iterations for faster testing
        eta=0.9,
        update_lag=5,
        a_smooth=1.0,
        a_data=0.5,
        sigma=[[2.0, 2.0, 1.5, 0.3], [1.8, 1.8, 1.2, 0.25]],  # 3D sigma [sx, sy, sz, st] per channel
        weight=[1.0, 1.0],  # Equal weight for 2 channels
        min_level=0,
        interpolation_method="cubic",
        update_reference=False,
        update_initialization_w=True,
        save_w=False,  # Don't save flow fields by default for testing
        save_meta_info=False,  # Don't save metadata by default for testing
        channel_normalization="joint",
        buffer_size=8,  # Smaller buffers for 3D testing
    )


def create_3d_motion_field(shape: Tuple[int, int, int, int] = (5, 8, 32, 32), 
                          motion_type: str = "translation") -> np.ndarray:
    """
    Create synthetic 3D motion field for testing.
    
    Args:
        shape: Motion field shape in TxZxYxX format
        motion_type: Type of motion ('translation', 'rotation', 'deformation')
        
    Returns:
        3D motion field with shape (T, Z, Y, X, 3) for [u, v, w] components
    """
    T, Z, Y, X = shape
    motion_field = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
    
    if motion_type == "translation":
        # Simple translational motion
        for t in range(T):
            motion_field[t, :, :, :, 0] = t * 1.5  # u component (X direction)
            motion_field[t, :, :, :, 1] = t * 1.0  # v component (Y direction)
            motion_field[t, :, :, :, 2] = t * 0.5  # w component (Z direction)
    
    elif motion_type == "rotation":
        # Rotational motion around Z axis
        center_y, center_x = Y // 2, X // 2
        z, y, x = np.mgrid[:Z, :Y, :X]
        
        for t in range(T):
            angle = t * 0.1  # Rotation angle
            
            # Distance from center
            dx = x - center_x
            dy = y - center_y
            
            # Rotational velocity field
            motion_field[t, :, :, :, 0] = -dy * np.sin(angle) * 0.5  # u
            motion_field[t, :, :, :, 1] = dx * np.sin(angle) * 0.5   # v
            motion_field[t, :, :, :, 2] = 0  # No Z motion for XY rotation
    
    elif motion_type == "deformation":
        # Deformation/expansion motion
        center_z, center_y, center_x = Z // 2, Y // 2, X // 2
        z, y, x = np.mgrid[:Z, :Y, :X]
        
        for t in range(T):
            expansion = t * 0.05  # Expansion factor
            
            motion_field[t, :, :, :, 0] = (x - center_x) * expansion  # u
            motion_field[t, :, :, :, 1] = (y - center_y) * expansion  # v  
            motion_field[t, :, :, :, 2] = (z - center_z) * expansion  # w
    
    return motion_field


def apply_3d_motion_to_volume(volume: np.ndarray, motion_field: np.ndarray) -> np.ndarray:
    """
    Apply 3D motion field to a volume to create moved version.
    
    Args:
        volume: Input volume with shape (Z, Y, X, C)
        motion_field: Motion field with shape (Z, Y, X, 3) for [u, v, w]
        
    Returns:
        Moved volume with same shape as input
    """
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        # Fallback - return original volume if scipy not available
        return volume.copy()
    
    Z, Y, X = volume.shape[:3]
    C = volume.shape[3] if volume.ndim == 4 else 1
    
    if volume.ndim == 3:
        volume = volume[:, :, :, np.newaxis]
    
    moved_volume = np.zeros_like(volume)
    
    # Create coordinate grids
    z_coords, y_coords, x_coords = np.mgrid[:Z, :Y, :X]
    
    for c in range(C):
        # Apply motion field to coordinates
        z_new = z_coords - motion_field[:, :, :, 2]  # w component
        y_new = y_coords - motion_field[:, :, :, 1]  # v component
        x_new = x_coords - motion_field[:, :, :, 0]  # u component
        
        # Interpolate
        coordinates = np.array([z_new.ravel(), y_new.ravel(), x_new.ravel()])
        moved_flat = map_coordinates(
            volume[:, :, :, c], 
            coordinates, 
            order=1, 
            mode='reflect'
        )
        moved_volume[:, :, :, c] = moved_flat.reshape((Z, Y, X))
    
    # Return in original shape
    if moved_volume.shape[3] == 1:
        return moved_volume[:, :, :, 0]
    else:
        return moved_volume


def cleanup_temp_files(*file_paths):
    """
    Clean up temporary files and directories.
    
    Args:
        *file_paths: Paths to files or directories to clean up
    """
    import shutil
    
    for path in file_paths:
        if path and Path(path).exists():
            try:
                path_obj = Path(path)
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
            except Exception as e:
                print(f"Warning: Could not clean up {path}: {e}")


if __name__ == "__main__":
    # Test 3D fixtures
    print("Testing 3D fixtures...")
    
    # Test 3D data creation
    shape_3d = (10, 6, 24, 24, 2)
    data = create_simple_3d_test_data(shape_3d)
    print(f"Created 3D test data with shape: {data.shape}")
    
    # Test 3D reference volume
    ref_shape = (6, 24, 24, 2)
    ref = create_3d_reference_volume(ref_shape)
    print(f"Created 3D reference with shape: {ref.shape}")
    
    # Test 3D motion field
    motion = create_3d_motion_field((5, 6, 24, 24), "translation")
    print(f"Created 3D motion field with shape: {motion.shape}")
    
    # Test minimal 3D options
    try:
        options = get_minimal_3d_of_options()
        print(f"Created minimal 3D options: alpha={options.alpha}, sigma shape={np.array(options.sigma).shape}")
    except ImportError as e:
        print(f"Could not create 3D options: {e}")
    
    print("3D fixtures test completed!")