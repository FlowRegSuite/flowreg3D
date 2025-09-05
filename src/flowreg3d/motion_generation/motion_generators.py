"""
3D Motion generators for creating synthetic displacement fields.
Extended from 2D versions to support volumetric data (Z, Y, X).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


def warp_volume_3d(volume, flow):
    """
    Warp a 3D volume using a 3D displacement field.
    
    Args:
        volume: Input volume of shape (Z, Y, X) or (Z, Y, X, C)
        flow: Displacement field of shape (Z, Y, X, 3) where last dim is (dx, dy, dz)
    
    Returns:
        Warped volume with same shape as input
    """
    depth, height, width = volume.shape[:3]
    has_channels = volume.ndim == 4
    
    # Create coordinate grids
    zi, yi, xi = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32), 
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    
    # Apply displacement (flow is [dx, dy, dz])
    xi_warped = xi + flow[:, :, :, 0]  # dx
    yi_warped = yi + flow[:, :, :, 1]  # dy
    zi_warped = zi + flow[:, :, :, 2]  # dz
    
    # Prepare output
    warped = np.zeros_like(volume)
    
    if has_channels:
        for c in range(volume.shape[3]):
            # Use griddata for interpolation
            points = np.column_stack([zi_warped.ravel(), yi_warped.ravel(), xi_warped.ravel()])
            values = volume[:, :, :, c].ravel()
            grid_points = np.column_stack([zi.ravel(), yi.ravel(), xi.ravel()])
            
            warped_channel = griddata(points, values, grid_points, 
                                     method='linear', fill_value=0)
            warped[:, :, :, c] = warped_channel.reshape(depth, height, width)
    else:
        points = np.column_stack([zi_warped.ravel(), yi_warped.ravel(), xi_warped.ravel()])
        values = volume.ravel()
        grid_points = np.column_stack([zi.ravel(), yi.ravel(), xi.ravel()])
        
        warped_flat = griddata(points, values, grid_points, 
                              method='linear', fill_value=0)
        warped = warped_flat.reshape(depth, height, width)
    
    return warped.astype(volume.dtype)


class Rotational3DFlowAugmentor:
    """Generate 3D rotational flow fields around specified axes."""
    
    def __init__(self, max_rot_deg=10, center=None, p=0.2, center_jitter=5, 
                 axes=('xy', 'xz', 'yz')):
        """
        Args:
            max_rot_deg: Maximum rotation in degrees per axis
            center: Center of rotation (z, y, x). If None, uses volume center
            p: Probability of applying augmentation
            center_jitter: Random jitter to apply to center
            axes: Which rotation axes to use
        """
        self.max_rot_deg = max_rot_deg
        self.center = center
        self.p = p
        self.center_jitter = center_jitter
        self.axes = axes
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
            
        depth, height, width = flow.shape[:3]
        
        # Set center if not provided
        if self.center is None:
            center = np.array([depth/2, height/2, width/2])
        else:
            center = np.array(self.center)
        
        # Add jitter to center
        center += np.random.uniform(-self.center_jitter, self.center_jitter, 3)
        
        # Create coordinate grids centered at rotation center
        Z, Y, X = np.meshgrid(
            np.arange(depth, dtype=np.float32) - center[0],
            np.arange(height, dtype=np.float32) - center[1],
            np.arange(width, dtype=np.float32) - center[2],
            indexing='ij'
        )
        
        # Random rotation angles for each axis
        angles = {}
        if 'xy' in self.axes:
            angles['xy'] = np.random.uniform(-self.max_rot_deg, self.max_rot_deg)
        if 'xz' in self.axes:
            angles['xz'] = np.random.uniform(-self.max_rot_deg, self.max_rot_deg)
        if 'yz' in self.axes:
            angles['yz'] = np.random.uniform(-self.max_rot_deg, self.max_rot_deg)
        
        # Apply rotations
        X_rot, Y_rot, Z_rot = X.copy(), Y.copy(), Z.copy()
        
        for axis, angle_deg in angles.items():
            angle_rad = np.radians(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            if axis == 'xy':  # Rotation around Z axis
                X_new = cos_a * X_rot - sin_a * Y_rot
                Y_new = sin_a * X_rot + cos_a * Y_rot
                X_rot, Y_rot = X_new, Y_new
            elif axis == 'xz':  # Rotation around Y axis
                X_new = cos_a * X_rot - sin_a * Z_rot
                Z_new = sin_a * X_rot + cos_a * Z_rot
                X_rot, Z_rot = X_new, Z_new
            elif axis == 'yz':  # Rotation around X axis
                Y_new = cos_a * Y_rot - sin_a * Z_rot
                Z_new = sin_a * Y_rot + cos_a * Z_rot
                Y_rot, Z_rot = Y_new, Z_new
        
        # Add displacement to flow (output as [dx, dy, dz])
        flow[:, :, :, 0] += X_rot - X  # dx
        flow[:, :, :, 1] += Y_rot - Y  # dy
        flow[:, :, :, 2] += Z_rot - Z  # dz
        
        return flow


class Translational3DFlowAugmentor:
    """Generate 3D translational flow fields."""
    
    def __init__(self, max_disp=10, p=0.3):
        """
        Args:
            max_disp: Maximum displacement in pixels for each axis
            p: Probability of applying augmentation
        """
        self.max_disp = max_disp
        self.p = p
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
        
        # Random displacement for each axis
        d_z = np.random.uniform(-self.max_disp, self.max_disp)
        d_y = np.random.uniform(-self.max_disp, self.max_disp)
        d_x = np.random.uniform(-self.max_disp, self.max_disp)
        
        flow[:, :, :, 0] += d_x  # dx
        flow[:, :, :, 1] += d_y  # dy
        flow[:, :, :, 2] += d_z  # dz
        
        return flow


class Jitter3DFlowAugmentor:
    """Generate 3D jitter/wave-like flow fields."""
    
    def __init__(self, max_magnitude=2, max_periods=5, min_periods=2, p=0.9,
                 axes=('x', 'y', 'z')):
        """
        Args:
            max_magnitude: Maximum jitter magnitude in pixels
            max_periods: Maximum number of wave periods
            min_periods: Minimum number of wave periods
            p: Probability of applying augmentation
            axes: Which axes to apply jitter along
        """
        self.max_magnitude = max_magnitude
        self.max_periods = max_periods
        self.min_periods = min_periods
        self.p = p
        self.axes = axes
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
        
        depth, height, width = flow.shape[:3]
        
        for axis in self.axes:
            if np.random.rand() < 0.5:  # 50% chance per axis
                periods = np.random.uniform(self.min_periods, self.max_periods)
                phase = np.random.uniform(0, 2*np.pi)
                magnitude = np.random.uniform(1, self.max_magnitude)
                
                if axis == 'x':
                    # Jitter along X axis
                    x_wave = np.linspace(phase, periods * 2*np.pi + phase, width)
                    jitter = magnitude * np.sin(x_wave)
                    flow[:, :, :, 2] += jitter[np.newaxis, np.newaxis, :]
                elif axis == 'y':
                    # Jitter along Y axis
                    y_wave = np.linspace(phase, periods * 2*np.pi + phase, height)
                    jitter = magnitude * np.sin(y_wave)
                    flow[:, :, :, 1] += jitter[np.newaxis, :, np.newaxis]
                elif axis == 'z':
                    # Jitter along Z axis
                    z_wave = np.linspace(phase, periods * 2*np.pi + phase, depth)
                    jitter = magnitude * np.sin(z_wave)
                    flow[:, :, :, 2] += jitter[:, np.newaxis, np.newaxis]  # dz (z-jitter)
        
        return flow


class Expansion3DFlowAugmentor:
    """Generate 3D expansion/contraction flow fields."""
    
    def __init__(self, max_magnitude=0.05, min_magnitude=-0.05, center=None, 
                 center_jitter=5, p=0.4, anisotropic=True):
        """
        Args:
            max_magnitude: Maximum expansion factor
            min_magnitude: Minimum expansion factor (negative for contraction)
            center: Center of expansion (z, y, x). If None, uses volume center
            center_jitter: Random jitter to apply to center
            p: Probability of applying augmentation
            anisotropic: If True, use different magnitudes for each axis
        """
        self.max_magnitude = max_magnitude
        self.min_magnitude = min_magnitude
        self.center = center
        self.center_jitter = center_jitter
        self.p = p
        self.anisotropic = anisotropic
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
        
        depth, height, width = flow.shape[:3]
        
        # Set center if not provided
        if self.center is None:
            center = np.array([depth/2, height/2, width/2])
        else:
            center = np.array(self.center)
        
        # Add jitter to center
        center += np.random.uniform(-self.center_jitter, self.center_jitter, 3)
        
        # Get magnitude for each axis
        if self.anisotropic:
            magnitude_z = np.random.uniform(self.min_magnitude, self.max_magnitude)
            magnitude_y = np.random.uniform(self.min_magnitude, self.max_magnitude)
            magnitude_x = np.random.uniform(self.min_magnitude, self.max_magnitude)
        else:
            magnitude = np.random.uniform(self.min_magnitude, self.max_magnitude)
            magnitude_z = magnitude_y = magnitude_x = magnitude
        
        # Create coordinate grids centered at expansion center
        Z, Y, X = np.meshgrid(
            np.arange(depth, dtype=np.float32) - center[0],
            np.arange(height, dtype=np.float32) - center[1],
            np.arange(width, dtype=np.float32) - center[2],
            indexing='ij'
        )
        
        # Add expansion displacement (output as [dx, dy, dz])
        flow[:, :, :, 0] += X * magnitude_x  # dx
        flow[:, :, :, 1] += Y * magnitude_y  # dy
        flow[:, :, :, 2] += Z * magnitude_z  # dz
        
        return flow


class Random3DFlowAugmentor:
    """Generate random smooth 3D flow fields using Gaussian filtering."""
    
    def __init__(self, p=0.3, min_sigma=2, max_sigma=10, max_magnitude=3):
        """
        Args:
            p: Probability of applying augmentation
            min_sigma: Minimum Gaussian filter sigma
            max_sigma: Maximum Gaussian filter sigma  
            max_magnitude: Maximum flow magnitude in pixels
        """
        self.p = p
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_magnitude = max_magnitude
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
        
        depth, height, width = flow.shape[:3]
        
        # Generate random noise for each flow component
        noise = np.random.randn(depth, height, width, 3)
        
        # Apply Gaussian smoothing with random sigma
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        for i in range(3):
            noise[:, :, :, i] = gaussian_filter(noise[:, :, :, i], sigma=sigma)
        
        # Normalize and scale
        noise -= noise.mean(axis=(0, 1, 2), keepdims=True)
        noise_std = noise.std(axis=(0, 1, 2), keepdims=True)
        noise_std[noise_std == 0] = 1  # Avoid division by zero
        noise = noise / noise_std
        
        # Scale to desired magnitude
        magnitude = np.random.uniform(0, self.max_magnitude)
        noise *= magnitude
        
        flow += noise
        
        return flow


class Shear3DFlowAugmentor:
    """Generate 3D shear flow fields."""
    
    def __init__(self, max_shear=0.1, p=0.3, planes=('xy', 'xz', 'yz')):
        """
        Args:
            max_shear: Maximum shear coefficient
            p: Probability of applying augmentation
            planes: Which shear planes to use
        """
        self.max_shear = max_shear
        self.p = p
        self.planes = planes
    
    def __call__(self, flow):
        if np.random.rand() > self.p:
            return flow
        
        depth, height, width = flow.shape[:3]
        
        # Create coordinate grids
        Z, Y, X = np.meshgrid(
            np.arange(depth, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )
        
        # Apply random shear for selected planes
        for plane in self.planes:
            if np.random.rand() < 0.5:  # 50% chance per plane
                shear = np.random.uniform(-self.max_shear, self.max_shear)
                
                if plane == 'xy':
                    # Shear in XY plane
                    flow[:, :, :, 0] += shear * Y  # dx displacement based on Y
                elif plane == 'xz':
                    # Shear in XZ plane
                    flow[:, :, :, 0] += shear * Z  # dx displacement based on Z
                elif plane == 'yz':
                    # Shear in YZ plane
                    flow[:, :, :, 1] += shear * Z  # dy displacement based on Z
        
        return flow


class FlowGenerator3D:
    """Generator for creating 3D flow fields with multiple augmentations."""
    
    def __init__(self):
        self.augmentors = []
    
    def add_augmentor(self, augmentor):
        """Add an augmentor to the pipeline."""
        self.augmentors.append(augmentor)
        return self
    
    def __call__(self, depth=64, height=128, width=128):
        """
        Generate a 3D flow field.
        
        Args:
            depth: Z dimension size
            height: Y dimension size  
            width: X dimension size
            
        Returns:
            flow: Flow field of shape (depth, height, width, 3)
            invalid_map: Boolean mask of invalid regions
        """
        # Initialize zero flow
        flow = np.zeros((depth, height, width, 3), dtype=np.float32)
        
        # Apply all augmentors
        for augmentor in self.augmentors:
            flow = augmentor(flow)
        
        # Compute forward mapping coordinates
        Z, Y, X = np.meshgrid(
            np.arange(depth, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )
        
        # Map coordinates using [dx, dy, dz] convention
        X_mapped = X + flow[:, :, :, 0]  # dx
        Y_mapped = Y + flow[:, :, :, 1]  # dy
        Z_mapped = Z + flow[:, :, :, 2]  # dz
        
        # Mark invalid regions (out of bounds)
        invalid_map = (
            (Z_mapped < 0) | (Z_mapped >= depth) |
            (Y_mapped < 0) | (Y_mapped >= height) |
            (X_mapped < 0) | (X_mapped >= width)
        )
        
        return flow, invalid_map


def get_default_3d_generator():
    """Create a default 3D flow generator with common augmentations."""
    generator = FlowGenerator3D()
    generator.add_augmentor(Rotational3DFlowAugmentor(max_rot_deg=5))
    generator.add_augmentor(Translational3DFlowAugmentor(max_disp=10))
    generator.add_augmentor(Random3DFlowAugmentor())
    generator.add_augmentor(Expansion3DFlowAugmentor())
    generator.add_augmentor(Jitter3DFlowAugmentor())
    generator.add_augmentor(Shear3DFlowAugmentor())
    return generator


def get_low_disp_3d_generator():
    """Create a 3D flow generator with low displacement augmentations."""
    generator = FlowGenerator3D()
    generator.add_augmentor(Translational3DFlowAugmentor(max_disp=5))
    generator.add_augmentor(Rotational3DFlowAugmentor(max_rot_deg=2))
    generator.add_augmentor(Random3DFlowAugmentor(max_magnitude=1.5))
    generator.add_augmentor(Expansion3DFlowAugmentor(max_magnitude=0.02))
    # Add small guaranteed translation and rotation
    generator.add_augmentor(Translational3DFlowAugmentor(max_disp=1, p=1.0))
    generator.add_augmentor(Rotational3DFlowAugmentor(max_rot_deg=0.5, p=1.0))
    generator.add_augmentor(Jitter3DFlowAugmentor(max_magnitude=1))
    return generator


def get_test_3d_generator():
    """Create a simple test generator for debugging."""
    generator = FlowGenerator3D()
    generator.add_augmentor(Translational3DFlowAugmentor(max_disp=5, p=1.0))
    generator.add_augmentor(Rotational3DFlowAugmentor(max_rot_deg=3, p=1.0))
    return generator


def get_high_disp_3d_generator():
    """Create a 3D flow generator with high displacement emphasizing expansion and jitter."""
    generator = FlowGenerator3D()
    generator.add_augmentor(Expansion3DFlowAugmentor(max_magnitude=0.15, p=1.0))
    generator.add_augmentor(Expansion3DFlowAugmentor(max_magnitude=0.1, p=1.0))
    generator.add_augmentor(Jitter3DFlowAugmentor(max_magnitude=3, p=1.0))
    generator.add_augmentor(Translational3DFlowAugmentor(max_disp=8, p=1.0))
    generator.add_augmentor(Rotational3DFlowAugmentor(max_rot_deg=3, p=1.0))
    generator.add_augmentor(Random3DFlowAugmentor(max_magnitude=2.5, p=1.0))
    return generator
