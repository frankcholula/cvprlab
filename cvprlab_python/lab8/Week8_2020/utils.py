"""
Helper functions for the contour lab exercises.
These functions are used across different exercises.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_image(img, title="Image", cmap=None):
    """
    Display an image with matplotlib.
    
    Args:
        img: Image to display
        title: Title for the plot
        cmap: Color map (e.g., 'gray' for grayscale images)
    """
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('on')
    plt.show()

def normalize_image(img):
    """
    Normalize image values to range [0,1].
    
    Args:
        img: Input image
    Returns:
        Normalized image
    """
    vmin = np.min(img)
    vrange = np.max(img) - vmin
    return (img - vmin) / vrange

def compute_edge_map(img):
    """
    Compute edge map using Sobel operators.
    
    Args:
        img: Input grayscale image
    Returns:
        Edge magnitude map
    """
    # Compute x and y gradients
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(dx**2 + dy**2)
    
    return normalize_image(magnitude)

def create_curve_matrices(curve_type='hermite'):
    """
    Get basis matrix for different curve types.
    
    Args:
        curve_type: Type of curve ('hermite', 'bezier', 'catmull-rom', 'bspline')
    Returns:
        Basis matrix for specified curve type
    """
    if curve_type == 'hermite':
        return np.array([
            [2, -3, 0, 1],
            [-2, 3, 0, 0],
            [1, -2, 1, 0],
            [1, -1, 0, 0]
        ])
    elif curve_type == 'bezier':
        return np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ])
    elif curve_type == 'catmull-rom':
        return 0.5 * np.array([
            [-1, 2, -1, 0],
            [3, -5, 0, 2],
            [-3, 4, 1, 0],
            [1, -1, 0, 0]
        ])
    elif curve_type == 'bspline':
        return (1/6) * np.array([
            [-1, 3, -3, 1],
            [3, -6, 0, 4],
            [-3, 3, 3, 1],
            [1, 0, 0, 0]
        ])
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

def create_parameter_space(num_points=100):
    """
    Create parameter space for curve generation.
    
    Args:
        num_points: Number of points to generate
    Returns:
        Q matrix [t^3; t^2; t; 1] for curve computation
    """
    t = np.linspace(0, 1, num_points)
    return np.vstack([t**3, t**2, t, np.ones_like(t)])

def optimize_snake_points(field, points, alpha=1.0, beta=1.1, gamma=1.2, iterations=100):
    """
    Optimize snake points to fit image edges.
    
    Args:
        field: Energy field (edge map)
        points: Initial snake points (2xN array)
        alpha: Continuity weight
        beta: Smoothness weight
        gamma: Image force weight
        iterations: Number of iterations
    Returns:
        Optimized snake points
    """
    current = points.copy()
    
    for _ in range(iterations):
        for i in range(current.shape[1]):
            x, y = current[:, i].astype(int)
            if 0 <= x < field.shape[1] and 0 <= y < field.shape[0]:
                # Sample neighborhood
                neighborhood = field[
                    max(0, y-1):min(field.shape[0], y+2),
                    max(0, x-1):min(field.shape[1], x+2)
                ]
                if neighborhood.size > 0:
                    min_y, min_x = np.unravel_index(
                        np.argmin(neighborhood),
                        neighborhood.shape
                    )
                    current[:, i] = [
                        x + (min_x - 1),
                        y + (min_y - 1)
                    ]
    
    return current