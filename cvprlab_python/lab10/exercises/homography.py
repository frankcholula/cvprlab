"""
Exercises related to homography estimation and planar transformations
Ex 7,8
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

from utils.visualization import (
    select_corresponding_points,
    HomographyVisualizer
)
from utils.geometry import compute_homography

def load_homography_images() -> Tuple[np.ndarray, np.ndarray]:
    """Load the image pair for homography exercises"""
    img1 = cv2.imread('data/images/parade1.bmp')
    img2 = cv2.imread('data/images/parade2.bmp')
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load parade images")
        
    return img1, img2

def exercise7_homography_estimation() -> np.ndarray:
    """
    Exercise 7: Homography Estimation
    
    Students will:
    1. Select corresponding points on a plane
    2. Compute homography matrix
    3. Verify the mapping mathematically
    
    Returns:
        H: 3x3 homography matrix
    """
    print("\nExercise 7: Homography Estimation")
    print("\nInstructions:")
    print("1. Select 4 corresponding points in both images")
    print("2. Points should be on the same plane (e.g., building facade)")
    print("3. Choose points that are well-distributed")
    print("4. Be as accurate as possible in point selection")
    
    try:
        # Load images
        img1, img2 = load_homography_images()
        
        # Get point correspondences
        points1, points2 = select_corresponding_points(
            img1, img2, n_points=4,
            title="Select 4 coplanar point pairs"
        )
        
        # Compute homography
        H = compute_homography(points1, points2)
        
        print("\nEstimated Homography Matrix:")
        print(H)
        
        # Verify mapping
        print("\nVerifying point mappings:")
        points1_h = np.column_stack([points1, np.ones(len(points1))])
        for i, (pt1, pt2) in enumerate(zip(points1_h, points2)):
            # Apply homography
            pt2_mapped = H @ pt1
            pt2_mapped = pt2_mapped / pt2_mapped[2]
            
            # Compare with actual point
            error = np.linalg.norm(pt2_mapped[:2] - pt2)
            print(f"Point {i+1} mapping error: {error:.3f} pixels")
        
        return H
        
    except Exception as e:
        print(f"Error in homography estimation: {str(e)}")
        return None

def exercise8_homography_interactive(H: Optional[np.ndarray] = None):
    """
    Exercise 8: Interactive Homography Visualization
    
    Students will:
    1. Visualize point mappings under homography
    2. Test homography with different point selections
    3. Understand planar transformation properties
    
    Args:
        H: Pre-computed homography matrix (if None, will compute new one)
    """
    print("\nExercise 8: Interactive Homography Visualization")
    
    try:
        img1, img2 = load_homography_images()
        
        if H is None:
            print("No homography matrix provided.")
            print("Computing new homography...")
            H = exercise7_homography_estimation()
            
        if H is None:
            print("Failed to compute homography.")
            return
            
        print("\nInteractive visualization:")
        print("- Click points in left image")
        print("- See corresponding points in right image")
        print("- Press 'q' to quit")
        print("\nTry points:")
        print("1. On the same plane as original points")
        print("2. Off the plane")
        print("3. In a straight line")
        print("Observe how well the mapping works in each case")
        
        # Create interactive visualization
        visualizer = HomographyVisualizer(img1, img2, H)
        visualizer.show()
        
        print("\nAdditional experiments:")
        print("1. Select points all in a line and compute new homography")
        print("2. Select points not all on same plane")
        print("3. Observe how these affect the mapping")
        
        while True:
            choice = input("\nTry new points? (y/n): ").lower()
            if choice != 'y':
                break
                
            new_H = exercise7_homography_estimation()
            if new_H is not None:
                visualizer = HomographyVisualizer(img1, img2, new_H)
                visualizer.show()
        
    except Exception as e:
        print(f"Error in interactive visualization: {str(e)}")

def verify_homography_properties(H: np.ndarray):
    """
    Utility function to verify mathematical properties of homography matrix
    
    Args:
        H: 3x3 homography matrix
    """
    print("\nVerifying homography properties:")
    
    # Check determinant (should be non-zero)
    det = np.linalg.det(H)
    print(f"Determinant: {det:.2e} (should be non-zero)")
    
    # Check rank (should be 3)
    rank = np.linalg.matrix_rank(H)
    print(f"Rank: {rank} (should be 3)")
    
    # Scale invariance
    H_normalized = H / H[2, 2]
    print("\nScale-normalized homography:")
    print(H_normalized)