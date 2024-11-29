from .fundamental import (
    exercise1_fundamental_matrix,
    exercise2_epipolar_geometry,
    exercise3_visual_verification,
    exercise4_mathematical_verification
)

from .triangulation import (
    exercise5_triangulation,
    exercise6_multiple_points
)

from .homography import (
    exercise7_homography_estimation,
    exercise8_homography_interactive
)

__all__ = [
    # Fundamental matrix exercises
    'exercise1_fundamental_matrix',
    'exercise2_epipolar_geometry',
    'exercise3_visual_verification',
    'exercise4_mathematical_verification',
    
    # Triangulation exercises
    'exercise5_triangulation',
    'exercise6_multiple_points',
    
    # Homography exercises
    'exercise7_homography_estimation',
    'exercise8_homography_interactive'
]