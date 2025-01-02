"""
Visual Hull Lab Package
EEE3032 - Computer Vision and Pattern Recognition

This package implements the visual hull reconstruction lab exercises.
"""

from .camera import Camera, load_camera
from .visualization import plot_cameras, display_masks
from .reconstruction import generate_masks, reconstruct_visual_hull

__version__ = '1.0.0'
__author__ = 'Original MATLAB by Miroslaw Bober (m.bober@surrey.ac.uk), Python lab by Armin Mustafa (armin.mustafa@surrey.ac.uk)'
