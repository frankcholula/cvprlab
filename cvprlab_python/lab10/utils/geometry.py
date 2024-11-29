"""
This module provides functions for:
1. Fundamental matrix estimation
2. Homography computation
3. Geometric transformations
4. Line computations
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2

def calculate_fundamental_matrix(points1: np.ndarray,
                               points2: np.ndarray,
                               method: str = 'normalized_8_point') -> np.ndarray:
    """
    Calculate fundamental matrix from point correspondences.
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        method: Algorithm to use ('normalized_8_point' or 'ransac')
        
    Returns:
        3x3 fundamental matrix
    """
    if len(points1) < 8 or len(points2) < 8:
        raise ValueError("At least 8 point correspondences required")
    
    if method == 'normalized_8_point':
        # Normalize points
        points1_norm, T1 = normalize_points(points1)
        points2_norm, T2 = normalize_points(points2)
        
        # Build constraint matrix
        A = []
        for (x1, y1), (x2, y2) in zip(points1_norm, points2_norm):
            A.append([
                x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1
            ])
        A = np.array(A)
        
        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        
        # Enforce rank-2 constraint
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ V
        
        # Denormalize
        F = T2.T @ F @ T1
        
    elif method == 'ransac':
        F, _ = cv2.findFundamentalMat(
            points1, points2, cv2.FM_RANSAC, 1.0, 0.99
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return F

def compute_homography(points1: np.ndarray,
                      points2: np.ndarray,
                      method: str = 'direct') -> np.ndarray:
    """
    Compute homography matrix mapping points1 to points2.
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        method: Algorithm to use ('direct' or 'ransac')
        
    Returns:
        3x3 homography matrix
    """
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError("At least 4 point correspondences required")
    
    if method == 'direct':
        # Normalize points
        points1_norm, T1 = normalize_points(points1)
        points2_norm, T2 = normalize_points(points2)
        
        # Build constraint matrix
        A = []
        for (x1, y1), (x2, y2) in zip(points1_norm, points2_norm):
            A.extend([
                [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2],
                [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]
            ])
        A = np.array(A)
        
        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        
        # Denormalize
        H = np.linalg.inv(T2) @ H @ T1
        
    elif method == 'ransac':
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize scale
    H = H / H[2, 2]
    
    return H

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply isotropic normalization to points.
    
    Args:
        points: Nx2 array of points
        
    Returns:
        Tuple of:
        - normalized_points: Normalized points
        - T: 3x3 normalization matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Shift to origin
    points_centered = points - centroid
    
    # Calculate average distance from origin
    avg_dist = np.mean(np.linalg.norm(points_centered, axis=1))
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist if avg_dist > 0 else 1.0
    
    # Create transformation matrix
    T = np.array([
        [scale, 0, -scale*centroid[0]],
        [0, scale, -scale*centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation
    points_h = np.column_stack([points, np.ones(len(points))])
    normalized_points = (T @ points_h.T).T[:, :2]
    
    return normalized_points, T

def homogeneous_line_endpoints(line: np.ndarray,
                             img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find endpoints of a line in homogeneous coordinates.
    
    Args:
        line: Line coefficients [a, b, c] where ax + by + c = 0
        img_shape: (height, width) of image
        
    Returns:
        Two endpoints of line segment within image bounds
    """
    height, width = img_shape
    
    # Normalize line coefficients
    line = line / np.sqrt(line[0]**2 + line[1]**2)
    a, b, c = line
    
    # Find intersections with image boundaries
    points = []
    
    # Check left and right edges
    for x in [0, width-1]:
        y = -(a*x + c) / b if abs(b) > 1e-10 else None
        if y is not None and 0 <= y < height:
            points.append(np.array([x, y]))
    
    # Check top and bottom edges
    for y in [0, height-1]:
        x = -(b*y + c) / a if abs(a) > 1e-10 else None
        if x is not None and 0 <= x < width:
            points.append(np.array([x, y]))
    
    # Take first two unique intersection points
    points = np.unique(points, axis=0)
    if len(points) < 2:
        raise ValueError("Line does not intersect image")
    
    return points[0], points[1]

def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert points to homogeneous coordinates"""
    if points.ndim == 1:
        return np.append(points, 1)
    return np.column_stack([points, np.ones(len(points))])

def from_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert points from homogeneous coordinates"""
    if points.ndim == 1:
        return points[:-1] / points[-1]
    return points[:, :-1] / points[:, -1:]