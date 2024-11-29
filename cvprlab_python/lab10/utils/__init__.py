"""
utils/__init__.py
Initialize the utilities package and provide convenient imports
"""

from .camera import (
    load_camera_params,
    get_light_ray,
    intersect_rays,
    load_2d_points
)

from .geometry import (
    calculate_fundamental_matrix,
    compute_homography,
    to_homogeneous,
    from_homogeneous,
    homogeneous_line_endpoints
)

from .visualization import (
    plot_epipolar_lines,
    plot_camera_setup,
    select_corresponding_points,
    InteractiveEpipolarVisualizer,
    HomographyVisualizer
)

__all__ = [
    # Camera utilities
    'load_camera_params',
    'get_light_ray',
    'intersect_rays',
    'load_2d_points',
    
    # Geometry utilities
    'calculate_fundamental_matrix',
    'compute_homography',
    'to_homogeneous',
    'from_homogeneous',
    'homogeneous_line_endpoints',
    
    # Visualization utilities
    'plot_epipolar_lines',
    'plot_camera_setup',
    'select_corresponding_points',
    'InteractiveEpipolarVisualizer',
    'HomographyVisualizer'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Based on MATLAB code by Miroslaw Bober'
__description__ = 'CVPR Lab utilities for multiview geometry exercises'