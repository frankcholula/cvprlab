"""
Visual hull reconstruction functions.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from visualization import display_masks
import os

def generate_masks(num_cameras=8, threshold=0.38):
    """
    Exercise 2: Generate binary masks from multi-view images.
    
    Args:
        num_cameras (int): Number of cameras
        threshold (float): Blue dominance threshold
        
    Returns:
        list: Binary masks for each view
    """
    masks = []
    
    for cam in range(num_cameras):
        # Read image
        img_path = f'mvdata/cam{cam}.png'
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Calculate blue dominance
        blue_dominance = img[:,:,2] / (img[:,:,0] + img[:,:,1] + img[:,:,2] + 1e-10)
        mask = (blue_dominance < threshold).astype(np.uint8)
        masks.append(mask)
        
        # Display results using visualization module
        display_masks(img, mask, cam)
    
    return masks

def reconstruct_visual_hull(cameras, masks, bounds, steps, threshold=5):
    """
    Exercise 3: Reconstruct 3D visual hull from multiple views.
    
    Args:
        cameras (list): Camera objects
        masks (list): Binary masks
        bounds (dict): Space bounds {x:(min,max), y:(min,max), z:(min,max)}
        steps (tuple): Step sizes (dx,dy,dz)
        threshold (int): Minimum camera agreements
        
    Returns:
        ndarray: 3D voxel volume
    """
    x_range = np.arange(bounds['x'][0], bounds['x'][1], steps[0])
    y_range = np.arange(bounds['y'][0], bounds['y'][1], steps[1])
    z_range = np.arange(bounds['z'][0], bounds['z'][1], steps[2])
    
    voxel_volume = np.zeros((len(x_range), len(y_range), len(z_range)))
    
    # Setup 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process each voxel
    total_steps = len(x_range)
    for i, x in enumerate(x_range):
        print(f'Reconstructing... {round(100*i/total_steps)}% done')
        
        for j, y in enumerate(y_range):
            for k, z in enumerate(z_range):
                hits = 0
                
                for cam, mask in zip(cameras, masks):
                    # Project 3D point to 2D
                    p = cam.T + np.dot(cam.R, np.array([x, y, z]))
                    
                    if p[2] <= 0:  # Behind camera
                        continue
                    
                    # Calculate pixel coordinates
                    u = int(round(p[0] * cam.fx / p[2] + cam.cx))
                    v = int(round(p[1] * cam.fy / p[2] + cam.cy))
                    
                    # Check bounds and mask
                    if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
                        if mask[v, u] > 0:
                            hits += 1
                            if hits >= threshold:
                                voxel_volume[i, j, k] = 1
                                ax.scatter(x, y, z, c='b', marker='.', alpha=0.1)
                                break
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_ylim(-2, 6)
    ax.set_xlim(-2, 6)
    ax.set_title('Visual Hull Reconstruction')
    plt.show()
    
    occupancy = np.sum(voxel_volume) / voxel_volume.size * 100
    print(f'Voxel occupancy: {occupancy:.2f}%')
    
    return voxel_volume
