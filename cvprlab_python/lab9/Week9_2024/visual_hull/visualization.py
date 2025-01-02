"""
Visualization functions for the visual hull lab.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cameras(cameras):
    """
    Exercise 1: Visualize camera positions in 3D.
    
    Args:
        cameras (list): List of Camera objects
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for cam in cameras:
        # Calculate camera center
        origin = -np.dot(cam.R.T, cam.T)
        
        # Draw camera as pyramid
        pyramid_points = np.array([
            [0.2, 0.2, 0.3],
            [-0.2, 0.2, 0.3],
            [-0.2, -0.2, 0.3],
            [0.2, -0.2, 0.3],
            [0, 0, 0]
        ])
        
        # Transform points to camera position
        transformed_points = np.dot(cam.R.T, pyramid_points.T).T + origin
        
        # Draw camera frame
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([transformed_points[i,0], transformed_points[j,0]],
                   [transformed_points[i,1], transformed_points[j,1]],
                   [transformed_points[i,2], transformed_points[j,2]], 'b-')
            ax.plot([transformed_points[i,0], transformed_points[4,0]],
                   [transformed_points[i,1], transformed_points[4,1]],
                   [transformed_points[i,2], transformed_points[4,2]], 'b-')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_ylim(-2, 6)
    ax.set_xlim(-2, 6)
    ax.set_title('Visual Hull - Camera Positions')
    plt.show()

def display_masks(image, mask, cam_idx):
    """
    Display original image and generated mask side by side.
    
    Args:
        image (ndarray): Original RGB image
        mask (ndarray): Binary mask
        cam_idx (int): Camera index for title
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(image)
    plt.title(f'Original Image {cam_idx}')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Generated Mask {cam_idx}')
    plt.show()
