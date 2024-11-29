"""
Exercises related to 3D reconstruction and camera geometry
Ex 5,6

This module implements:
1. Single point triangulation with visualization
2. Multiple point triangulation with trajectory analysis
3. Camera setup visualization
4. Error analysis and validation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

from utils.camera import (
    load_camera_params,
    get_light_ray,
    intersect_rays,
    load_2d_points
)
from utils.visualization import plot_camera_setup

def exercise5_triangulation():
    """
    Exercise 5: Single Point Triangulation
    
    Students will:
    1. Load and visualize camera setup
    2. Understand light ray calculation
    3. Implement ray intersection
    4. Reconstruct 3D points
    
    Returns:
        point_3d: Reconstructed 3D point coordinates
    """
    print("\nExercise 5: 3D Point Triangulation")
    print("\nThis exercise demonstrates:")
    print("1. Camera setup visualization")
    print("2. Light ray calculation")
    print("3. Ray intersection and 3D reconstruction")
    
    try:
        # Load camera parameters
        cameras = load_camera_params('data/calibration.txt')
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot studio floor (x-z plane) and origin
        ax.scatter([0], [0], [0], c='g', marker='*', s=100, label='Origin (Floor)')
        
        # Plot camera setup
        plot_camera_setup(ax, cameras)
        
        # Select two cameras for triangulation
        cam1_idx, cam2_idx = 0, 4  # Using cameras 1 and 5 as in MATLAB code
        print(f"\nUsing cameras {cam1_idx+1} and {cam2_idx+1} for triangulation")
        
        # Load 2D points
        points_cam1 = load_2d_points(f'data/points2D_cam{cam1_idx+1}.txt')
        points_cam2 = load_2d_points(f'data/points2D_cam{cam2_idx+1}.txt')
        
        # Find first valid point pair
        valid_idx = None
        for idx in range(points_cam1.shape[1]):
            if (points_cam1[0, idx] >= 0 and points_cam2[0, idx] >= 0):
                valid_idx = idx
                break
                
        if valid_idx is None:
            raise ValueError("No valid point pairs found")
            
        pt1 = points_cam1[:, valid_idx]
        pt2 = points_cam2[:, valid_idx]
        
        # Calculate light rays
        ray1_origin, ray1_dir = get_light_ray(
            cameras[cam1_idx]['R'],
            cameras[cam1_idx]['t'],
            cameras[cam1_idx]['K'],
            pt1
        )
        
        ray2_origin, ray2_dir = get_light_ray(
            cameras[cam2_idx]['R'],
            cameras[cam2_idx]['t'],
            cameras[cam2_idx]['K'],
            pt2
        )
        
        # Plot rays
        scale = 3.0  # Scale factor for ray visualization
        for origin, direction, color in [
            (ray1_origin, ray1_dir, 'r'),
            (ray2_origin, ray2_dir, 'b')
        ]:
            endpoint = origin + direction * scale
            ax.plot([origin[0], endpoint[0]],
                   [origin[1], endpoint[1]],
                   [origin[2], endpoint[2]],
                   f'{color}--', linewidth=2,
                   label=f'Camera {color} ray')
        
        # Find and plot intersection
        point_3d = intersect_rays(ray1_origin, ray1_dir,
                                ray2_origin, ray2_dir)
        ax.scatter(*point_3d, c='m', s=100, label='Reconstructed Point')
        
        # Set axis labels and properties
        ax.set_xlabel('X (metres)')
        ax.set_ylabel('Y (metres)')
        ax.set_zlabel('Z (metres)')
        ax.legend()
        
        # Set reasonable axis limits
        ax.set_xlim([-3, 8])
        ax.set_ylim([-3, 8])
        ax.set_zlim([0, 5])
        
        print("\nReconstructed 3D point coordinates (metres):")
        print(f"X: {point_3d[0]:.3f}")
        print(f"Y: {point_3d[1]:.3f}")
        print(f"Z: {point_3d[2]:.3f}")
        
        plt.show()
        
        return point_3d
        
    except Exception as e:
        print(f"Error in triangulation: {str(e)}")
        return None

def exercise6_multiple_points():
    """
    Exercise 6: Multiple Point Triangulation
    
    Students will:
    1. Process multiple point observations
    2. Reconstruct 3D trajectory
    3. Analyze reconstruction accuracy
    4. Visualize results
    
    Returns:
        points_3d: List of reconstructed 3D points
    """
    print("\nExercise 6: Multiple Point Triangulation")
    print("\nThis exercise demonstrates:")
    print("1. Batch 3D reconstruction")
    print("2. Trajectory visualization")
    print("3. Error analysis")
    
    try:
        # Load camera parameters
        cameras = load_camera_params('data/calibration.txt')
        
        # Create 3D visualization
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot studio setup
        ax.scatter([0], [0], [0], c='g', marker='*', s=100, label='Origin (Floor)')
        plot_camera_setup(ax, cameras)
        
        # Select cameras for triangulation
        cam1_idx, cam2_idx = 0, 4  # Using cameras 1 and 5
        
        # Load all point observations
        points_cam1 = load_2d_points(f'data/points2D_cam{cam1_idx+1}.txt')
        points_cam2 = load_2d_points(f'data/points2D_cam{cam2_idx+1}.txt')
        
        points_3d = []
        times = []
        
        print("\nProcessing points...")
        start_time = time.time()
        
        # Process all points
        for frame in range(points_cam1.shape[1]):
            # Check if point is visible in both views
            if points_cam1[0, frame] < 0 or points_cam2[0, frame] < 0:
                continue
                
            # Get point pair
            pt1 = points_cam1[:, frame]
            pt2 = points_cam2[:, frame]
            
            # Calculate light rays
            ray1_origin, ray1_dir = get_light_ray(
                cameras[cam1_idx]['R'],
                cameras[cam1_idx]['t'],
                cameras[cam1_idx]['K'],
                pt1
            )
            
            ray2_origin, ray2_dir = get_light_ray(
                cameras[cam2_idx]['R'],
                cameras[cam2_idx]['t'],
                cameras[cam2_idx]['K'],
                pt2
            )
            
            # Triangulate point
            point_3d = intersect_rays(ray1_origin, ray1_dir,
                                    ray2_origin, ray2_dir)
            
            points_3d.append(point_3d)
            times.append(frame)
            
        points_3d = np.array(points_3d)
        
        # Plot reconstructed trajectory
        if len(points_3d) > 0:
            ax.scatter(points_3d[:, 0],
                      points_3d[:, 1],
                      points_3d[:, 2],
                      c=times, cmap='viridis',
                      s=20, label='Trajectory')
            
            # Plot trajectory line
            ax.plot(points_3d[:, 0],
                   points_3d[:, 1],
                   points_3d[:, 2],
                   'k-', alpha=0.3, linewidth=1)
        
        # Set axis labels and properties
        ax.set_xlabel('X (metres)')
        ax.set_ylabel('Y (metres)')
        ax.set_zlabel('Z (metres)')
        ax.legend()
        
        # Set reasonable axis limits
        ax.set_xlim([-3, 8])
        ax.set_ylim([-3, 8])
        ax.set_zlim([0, 5])
        
        # Add colorbar for time
        if len(points_3d) > 0:
            scatter = ax.scatter(points_3d[:, 0],
                               points_3d[:, 1],
                               points_3d[:, 2],
                               c=times, cmap='viridis')
            plt.colorbar(scatter, label='Frame number')
        
        # Print statistics
        processing_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"Total frames processed: {len(times)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if len(points_3d) > 0:
            print("\nTrajectory statistics:")
            print(f"Height range: {np.min(points_3d[:, 1]):.2f} to {np.max(points_3d[:, 1]):.2f} metres")
            print(f"Total path length: {np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1)):.2f} metres")
        
        plt.show()
        
        return points_3d
        
    except Exception as e:
        print(f"Error in multiple point triangulation: {str(e)}")
        return None

def analyze_reconstruction_accuracy(points_3d: np.ndarray) -> None:
    """
    Analyze the accuracy of 3D reconstruction
    
    Args:
        points_3d: Array of reconstructed 3D points
    """
    if len(points_3d) == 0:
        print("No points to analyze")
        return
        
    # Calculate velocities
    velocities = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
    
    # Plot velocity histogram
    plt.figure(figsize=(10, 5))
    plt.hist(velocities, bins=30)
    plt.xlabel('Velocity (metres/frame)')
    plt.ylabel('Count')
    plt.title('Distribution of Point Velocities')
    plt.show()
    
    # Print statistics
    print("\nReconstruction Statistics:")
    print(f"Average velocity: {np.mean(velocities):.3f} metres/frame")
    print(f"Maximum velocity: {np.max(velocities):.3f} metres/frame")
    print(f"Minimum velocity: {np.min(velocities):.3f} metres/frame")
    print(f"Velocity std dev: {np.std(velocities):.3f} metres/frame")

if __name__ == "__main__":
    # Test exercises independently
    print("Testing triangulation exercises...")
    
    # Test single point triangulation
    point = exercise5_triangulation()
    if point is not None:
        print("\nSingle point triangulation successful!")
    
    # Test multiple point triangulation
    points = exercise6_multiple_points()
    if points is not None and len(points) > 0:
        print("\nMultiple point triangulation successful!")
        analyze_reconstruction_accuracy(points)