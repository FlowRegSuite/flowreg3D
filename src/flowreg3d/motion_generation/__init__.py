"""3D motion generation module for creating synthetic displacement fields."""

# Use relative imports (best practice)
from .motion_generators import (
    FlowGenerator3D,
    Rotational3DFlowAugmentor,
    Translational3DFlowAugmentor,
    Jitter3DFlowAugmentor,
    Expansion3DFlowAugmentor,
    Random3DFlowAugmentor,
    Shear3DFlowAugmentor,
    warp_volume_3d,
    get_default_3d_generator,
    get_low_disp_3d_generator,
    get_test_3d_generator,
    get_high_disp_3d_generator,
)

# Define public API explicitly
__all__ = [
    "FlowGenerator3D",
    "Rotational3DFlowAugmentor",
    "Translational3DFlowAugmentor",
    "Jitter3DFlowAugmentor",
    "Expansion3DFlowAugmentor",
    "Random3DFlowAugmentor",
    "Shear3DFlowAugmentor",
    "warp_volume_3d",
    "get_default_3d_generator",
    "get_low_disp_3d_generator",
    "get_test_3d_generator",
    "get_high_disp_3d_generator",
]
