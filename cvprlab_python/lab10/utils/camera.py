"""
This module provides functions for:
1. Camera parameter loading
2. Light ray calculations
3. Point loading and processing
4. Camera geometry computations
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path

def load_camera_params(filename: str, num_cameras: Optional[int] = None) -> List[Dict]:
    """
    Load camera calibration parameters from file.
    
    Args:
        filename: Path to calibration file
        num_cameras: Optional number of cameras to load
        
    Returns:
        List of dictionaries containing camera parameters:
        - K: 3x3 intrinsic matrix
        - R: 3x3 rotation matrix
        - t: 3x1 translation vector
        - image_size: (height, width)
    """
    cameras = []
    try:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        num_cams = num_cameras or int(lines[0].split()[0])
        i = 1

        for _ in range(num_cams):
            # Each camera takes exactly 7 lines
            dims = [int(x) for x in lines[i].split()]
            imgsize = (dims[1], dims[3])
            intrinsics = [float(x) for x in lines[i + 1].split()]
            fx, fy, cx, cy = intrinsics
            R = np.zeros((3, 3))
            for j in range(3):
                R[j] = [float(x) for x in lines[i + 3 + j].split()]

            T = np.array([float(x) for x in lines[i + 6].split()])

            K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            cameras.append({
                    'K': K,
                    'R': R,
                    't': T,
                    'image_size': (dims[1] , dims[3])
                })

            i += 7
        return cameras
    except Exception as e:
        print(f"Error loading calibration file: {str(e)}")
        return None


def get_light_ray(R: np.ndarray, 
                  t: np.ndarray, 
                  K: np.ndarray, 
                  point_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate light ray passing through a 2D image point.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera matrix
        point_2d: 2D point in image coordinates
        
    Returns:
        Tuple of:
        - ray_origin: 3D camera center
        - ray_direction: Normalized 3D ray direction
    """
    # Calculate camera center in world coordinates
    ray_origin = -R.T @ t
    
    # Convert 2D point to normalized coordinates
    point_2d = point_2d - K[:2, 2]  # Remove principal point offset
    normalized_point = np.array([
        point_2d[0] / K[0, 0],  # x / fx
        point_2d[1] / K[1, 1],  # y / fy
        1.0
    ])
    
    # Calculate ray direction in world coordinates
    ray_direction = R.T @ normalized_point
    
    # Normalize direction vector
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    return ray_origin, ray_direction

def intersect_rays(ray1_origin: np.ndarray,
                  ray1_dir: np.ndarray,
                  ray2_origin: np.ndarray,
                  ray2_dir: np.ndarray) -> np.ndarray:
    """
    Find the closest point to two skew rays in 3D space.
    
    Args:
        ray1_origin: Origin of first ray
        ray1_dir: Direction of first ray (normalized)
        ray2_origin: Origin of second ray
        ray2_dir: Direction of second ray (normalized)
        
    Returns:
        3D point closest to both rays
    """
    # Build system of equations
    A = np.array([
        [np.dot(ray1_dir, ray1_dir), -np.dot(ray1_dir, ray2_dir)],
        [np.dot(ray1_dir, ray2_dir), -np.dot(ray2_dir, ray2_dir)]
    ])
    
    b = np.array([
        np.dot(ray1_dir, ray2_origin - ray1_origin),
        np.dot(ray2_dir, ray2_origin - ray1_origin)
    ])
    
    # Solve for parameters
    s, t = np.linalg.solve(A, b)
    
    # Calculate points on each ray
    point1 = ray1_origin + s * ray1_dir
    point2 = ray2_origin + t * ray2_dir
    
    # Return midpoint
    return (point1 + point2) / 2

def load_2d_points(filename: str) -> np.ndarray:
    """
    Load 2D point observations from file.
    
    Args:
        filename: Path to points file
        
    Returns:
        2xN array of 2D points
    """
    try:
        with open(filename, 'r') as f:
            # Read number of points
            n_points = int(f.readline().strip())
            
            # Initialize arrays
            points = np.zeros((2, n_points))
            
            # Read points
            for i in range(n_points):
                x, y = map(float, f.readline().split())
                points[:, i] = [x, y]
            
            return points
            
    except Exception as e:
        raise IOError(f"Error loading 2D points: {str(e)}")

def project_point(point_3d: np.ndarray,
                 camera: Dict) -> np.ndarray:
    """
    Project 3D point into camera image plane.
    
    Args:
        point_3d: 3D point coordinates
        camera: Camera parameters dictionary
        
    Returns:
        2D point in image coordinates
    """
    # Convert to homogeneous coordinates
    point_h = np.append(point_3d, 1)
    
    # Project into camera coordinates
    point_cam = camera['R'] @ point_3d + camera['t']
    
    # Project into image plane
    point_img = camera['K'] @ point_cam
    
    # Convert to inhomogeneous coordinates
    point_2d = point_img[:2] / point_img[2]
    
    return point_2d