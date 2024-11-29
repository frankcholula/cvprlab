"""
Shape Recognition Lab Utilities
=============================

This package contains utility functions for the Shape Recognition Lab exercises.

Available modules:
- draw_shape: Interactive shape drawing
- chaincode: Chain code generation and processing 
- fourier_descriptors: Shape description using Fourier analysis
- eigenmodel: Pattern classification using eigenmodels

For the Computer Vision and Pattern Recognition course.
"""

# Import main functions to make them directly accessible
from .draw_shape import draw_shape
from .chaincode import chaincode
from .chaincode_rasterize import chaincode_rasterize
from .sample_polygon_perimeter import sample_polygon_perimeter
from .fourier_descriptors import compute_fd_angular
from .eigenmodel import (
    EigenModel,
    eigen_build,
    eigen_deflate,
    eigen_mahalanobis
)

# Define what should be available when using "from utils import *"
__all__ = [
    'draw_shape',
    'chaincode',
    'chaincode_rasterize',
    'sample_polygon_perimeter',
    'compute_fd_angular',
    'EigenModel',
    'eigen_build',
    'eigen_deflate',
    'eigen_mahalanobis',
]

# Version information
__version__ = '1.0.0'
__author__ = 'Original MATLAB code by John Collomosse (J.Collomosse@surrey.ac.uk)'
__credits__ = ['John Collomosse', 'Python implementation by Your Name']