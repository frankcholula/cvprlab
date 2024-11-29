"""
This module provides:
1. Camera visualization tools
2. Interactive point selection
3. Epipolar geometry visualization
4. Homography visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import Tuple, List, Dict, Optional, Callable
from .geometry import homogeneous_line_endpoints, to_homogeneous

class InteractivePointSelector:
    """Interactive tool for selecting corresponding points in images"""
    
    def __init__(self, img1: np.ndarray, img2: np.ndarray, n_points: int):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        self.n_points = n_points
        self.points1 = []
        self.points2 = []
        self.current_image = 1
        self.window_name = "Point Selection"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_image == 1:
                self.points1.append((x, y))
                cv2.circle(self.img1, (x, y), 3, (0, 255, 0), -1)
                self.current_image = 2
                cv2.imshow(self.window_name, self.img2)
            else:
                self.points2.append((x, y))
                cv2.circle(self.img2, (x, y), 3, (0, 255, 0), -1)
                self.current_image = 1
                cv2.imshow(self.window_name, self.img1)
                
    def select_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run point selection interface"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"Please select {self.n_points} corresponding points")
        print("Left click to select points")
        
        while len(self.points1) < self.n_points or len(self.points2) < self.n_points:
            if self.current_image == 1:
                cv2.imshow(self.window_name, self.img1)
            else:
                cv2.imshow(self.window_name, self.img2)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
        return (np.array(self.points1), np.array(self.points2))

def plot_camera_setup(ax: plt.Axes, cameras: List[Dict]) -> None:
    """
    Visualize camera setup in 3D.
    
    Args:
        ax: Matplotlib 3D axis
        cameras: List of camera parameter dictionaries
    """
    def draw_camera_frame(ax, R, t, scale=0.5):
        """Draw camera coordinate frame"""
        center = -R.T @ t
        
        # Draw coordinate axes
        for i, color in enumerate(['r', 'g', 'b']):
            direction = R.T[:, i]
            endpoint = center + direction * scale
            ax.plot([center[0], endpoint[0]],
                   [center[1], endpoint[1]],
                   [center[2], endpoint[2]],
                   color=color)
        
        # Draw camera frustum
        fx = scale * 0.7
        fy = scale * 0.5
        points = np.array([
            [fx, fy, scale],
            [fx, -fy, scale],
            [-fx, -fy, scale],
            [-fx, fy, scale]
        ])
        
        # Transform points to world coordinates
        points_world = [(R.T @ (p - t)) for p in points]
        
        # Draw frustum lines
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([points_world[i][0], points_world[j][0]],
                   [points_world[i][1], points_world[j][1]],
                   [points_world[i][2], points_world[j][2]],
                   'k-', alpha=0.5)
            ax.plot([center[0], points_world[i][0]],
                   [center[1], points_world[i][1]],
                   [center[2], points_world[i][2]],
                   'k-', alpha=0.5)
    
    # Plot each camera
    for i, cam in enumerate(cameras):
        # Draw camera frame
        draw_camera_frame(ax, cam['R'], cam['t'])
        
        # Add camera label
        center = -cam['R'].T @ cam['t']
        ax.text(center[0], center[1], center[2],
                f'Cam {i+1}',
                color='k')

def plot_epipolar_lines(img1: np.ndarray,
                       img2: np.ndarray,
                       F: np.ndarray,
                       points1: np.ndarray,
                       points2: np.ndarray) -> None:
    """
    Visualize epipolar lines and corresponding points.
    
    Args:
        img1: First image
        img2: Second image
        F: Fundamental matrix
        points1: Points in first image
        points2: Points in second image
    """
    def draw_line(img, line):
        """Draw epipolar line on image"""
        height, width = img.shape[:2]
        try:
            pt1, pt2 = homogeneous_line_endpoints(line, (height, width))
            cv2.line(img,
                    tuple(map(int, pt1)),
                    tuple(map(int, pt2)),
                    (0, 255, 0), 1)
        except ValueError:
            pass
    
    # Create copies for drawing
    img1_lines = img1.copy()
    img2_lines = img2.copy()
    
    # Draw points and corresponding epipolar lines
    for pt1, pt2 in zip(points1, points2):
        # Convert points to homogeneous coordinates
        pt1_h = to_homogeneous(pt1)
        pt2_h = to_homogeneous(pt2)
        
        # Calculate epipolar lines
        line2 = F @ pt1_h  # line in image 2
        line1 = F.T @ pt2_h  # line in image 1
        
        # Draw points
        cv2.circle(img1_lines, tuple(map(int, pt1)), 5, (0, 0, 255), -1)
        cv2.circle(img2_lines, tuple(map(int, pt2)), 5, (0, 0, 255), -1)
        
        # Draw epipolar lines
        draw_line(img1_lines, line1)
        draw_line(img2_lines, line2)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image 1 with epipolar lines')
    ax2.set_title('Image 2 with epipolar lines')
    plt.tight_layout()
    plt.show()

class HomographyVisualizer:
    """Interactive tool for visualizing homography mappings"""
    
    def __init__(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        self.H = H
        self.window_name = "Homography Visualization"
        
        # Create combined image
        self.height = max(img1.shape[0], img2.shape[0])
        self.combined_img = np.zeros((self.height,
                                    img1.shape[1] + img2.shape[1],
                                    3), dtype=np.uint8)
        self.combined_img[:img1.shape[0], :img1.shape[1]] = img1
        self.combined_img[:img2.shape[0], img1.shape[1]:] = img2
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only process clicks in left image
            if x < self.img1.shape[1]:
                # Get point in homogeneous coordinates
                pt = np.array([x, y, 1])
                
                # Apply homography
                pt_mapped = self.H @ pt
                pt_mapped = pt_mapped / pt_mapped[2]
                pt_mapped = pt_mapped[:2].astype(int)
                
                # Adjust x coordinate for display
                pt_mapped[0] += self.img1.shape[1]
                
                # Draw points and line
                img_display = self.combined_img.copy()
                cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
                cv2.circle(img_display, tuple(pt_mapped), 3, (0, 0, 255), -1)
                cv2.line(img_display, (x, y), tuple(pt_mapped), (255, 0, 0), 1)
                
                cv2.imshow(self.window_name, img_display)
    
    def run(self):
        """Run interactive visualization"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Click points in left image to see mappings")
        print("Press 'q' to quit")
        
        while True:
            cv2.imshow(self.window_name, self.combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

def select_corresponding_points(img1: np.ndarray,
                             img2: np.ndarray,
                             n_points: int,
                             title: str = "Select corresponding points") -> Tuple[np.ndarray, np.ndarray]:
    """
    Interactive point selection interface
    
    Args:
        img1: First image
        img2: Second image
        n_points: Number of point pairs to select
        title: Window title
        
    Returns:
        points1, points2: Arrays of corresponding points
    """
    selector = InteractivePointSelector(img1, img2, n_points)
    return selector.select_points()

def draw_coordinate_frame(ax: plt.Axes,
                        origin: np.ndarray,
                        R: np.ndarray,
                        scale: float = 1.0,
                        label: Optional[str] = None) -> None:
    """
    Draw coordinate frame on 3D axis
    
    Args:
        ax: Matplotlib 3D axis
        origin: Origin of coordinate frame
        R: Rotation matrix defining orientation
        scale: Size of coordinate axes
        label: Optional text label
    """
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        direction = R[:, i]
        endpoint = origin + direction * scale
        ax.plot([origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color)
    
    if label is not None:
        ax.text(origin[0], origin[1], origin[2], label)