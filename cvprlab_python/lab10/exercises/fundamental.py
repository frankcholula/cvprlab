"""
Exercises related to fundamental matrix estimation and epipolar geometry
Ex 1-4
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

from utils.visualization import (
    select_corresponding_points,
    plot_epipolar_lines,
    InteractiveEpipolarVisualizer
)
from utils.geometry import calculate_fundamental_matrix

def load_image_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Load the standard image pair used in exercises"""
    img1 = cv2.imread('data/images/view1.jpg')
    img2 = cv2.imread('data/images/view2.jpg')
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load test images")
        
    return img1, img2

def exercise1_fundamental_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exercise 1: Fundamental Matrix Estimation
    
    Students will:
    1. Select 8 corresponding points in two images
    2. Compute fundamental matrix using 8-point algorithm
    3. Understand the constraint x2.T * F * x1 = 0
    
    Returns:
        F: Fundamental matrix
        points_left, points_right: Selected point correspondences
    """
    print("\nExercise 1: Fundamental Matrix Estimation")
    print("Select 8 corresponding points in both images")
    print("Tips:")
    print("- Choose points from different depths")
    print("- Avoid choosing points all on same plane")
    print("- Be as accurate as possible in point selection")
    
    img1, img2 = load_image_pair()
    
    # Get point correspondences
    points_left, points_right = select_corresponding_points(
        img1, img2, n_points=8,
        title="Select 8 point pairs for fundamental matrix estimation"
    )
    
    # Calculate fundamental matrix
    F = calculate_fundamental_matrix(points_left, points_right)
    
    print("\nEstimated Fundamental Matrix:")
    print(F)
    
    return F, points_left, points_right

def exercise2_epipolar_geometry():
    """
    Exercise 2: Understanding Epipolar Geometry
    
    Paper exercise explanations and code verification
    Students will understand:
    1. How F maps points to lines
    2. The epipolar constraint
    3. Properties of the fundamental matrix
    """
    print("\nExercise 2: Understanding Epipolar Geometry")
    print("\nKey concepts to understand:")
    print("1. The fundamental matrix F maps a point in one image")
    print("   to a line in the other image (epipolar line)")
    print("2. For corresponding points x and x':")
    print("   x'.T * F * x = 0")
    print("3. F has rank 2 and 7 degrees of freedom")
    print("\nVerify these properties in the code:")
    print("- View the calculate_fundamental_matrix() implementation")
    print("- Notice how rank-2 constraint is enforced")
    print("- Understand the normalization step")

def exercise3_visual_verification(
    F: Optional[np.ndarray] = None,
    points_left: Optional[np.ndarray] = None,
    points_right: Optional[np.ndarray] = None
):
    """
    Exercise 3: Visual Verification of Epipolar Lines
    
    Students will:
    1. Visualize epipolar lines for selected points
    2. Verify geometrical relationships
    3. Understand correspondence geometry
    """
    print("\nExercise 3: Visual Verification")
    
    if F is None or points_left is None or points_right is None:
        print("No fundamental matrix available.")
        print("Please run Exercise 1 first.")
        return
        
    img1, img2 = load_image_pair()
    
    # Show epipolar lines for selected points
    plot_epipolar_lines(img1, img2, F, points_left, points_right)
    
    # Interactive visualization
    visualizer = InteractiveEpipolarVisualizer(img1, img2, F)
    print("\nClick points in either image to see epipolar lines")
    print("Press 'q' to quit")
    visualizer.show()

def exercise4_mathematical_verification(
    F: Optional[np.ndarray] = None,
    points_left: Optional[np.ndarray] = None,
    points_right: Optional[np.ndarray] = None
):
    """
    Exercise 4: Mathematical Verification
    
    Students will:
    1. Verify the epipolar constraint numerically
    2. Check properties of F
    3. Understand numerical aspects
    """
    if F is None or points_left is None or points_right is None:
        print("No fundamental matrix available.")
        print("Please run Exercise 1 first.")
        return
        
    print("\nExercise 4: Mathematical Verification")
    
    # Add homogeneous coordinates
    points_left_h = np.column_stack([points_left, np.ones(len(points_left))])
    points_right_h = np.column_stack([points_right, np.ones(len(points_right))])
    
    # Verify epipolar constraint for each point pair
    print("\nVerifying epipolar constraint x'.T * F * x = 0")
    print("Values should be close to zero:")
    for i, (left, right) in enumerate(zip(points_left_h, points_right_h)):
        constraint = right @ F @ left
        print(f"Point pair {i+1}: {constraint:.2e}")
    
    # Verify F properties
    print("\nVerifying F properties:")
    
    # Rank
    rank = np.linalg.matrix_rank(F)
    print(f"Rank of F (should be 2): {rank}")
    
    # Determinant
    det = np.linalg.det(F)
    print(f"Determinant of F (should be 0): {det:.2e}")
    
    # Singular values
    U, s, Vt = np.linalg.svd(F)
    print("\nSingular values of F:")
    print(f"σ1: {s[0]:.6f}")
    print(f"σ2: {s[1]:.6f}")
    print(f"σ3: {s[2]:.6f} (should be ≈ 0)")