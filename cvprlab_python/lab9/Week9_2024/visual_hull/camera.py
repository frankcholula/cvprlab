"""
Camera calibration and parameters module.
"""

import numpy as np

class Camera:
    """Camera class to store calibration parameters"""
    def __init__(self, R, T, fx, fy, cx, cy, imgsize):
        self.R = R  # Rotation matrix (3x3)
        self.T = T  # Translation vector (3x1)
        self.fx = fx  # Focal length x
        self.fy = fy  # Focal length y
        self.cx = cx  # Principal point x
        self.cy = cy  # Principal point y
        self.imgsize = imgsize  # Image dimensions (width, height)

def load_camera():
    """
    Load camera calibration parameters from calibration.txt.
    Returns list of Camera objects.
    """
    cameras = []
    
    try:
        with open('mvdata/calibration.txt', 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            # Read rotation matrix
            R = np.zeros((3, 3))
            for j in range(3):
                R[j] = [float(x) for x in lines[i+j].strip().split()]
            i += 3
            
            # Read translation vector
            T = np.array([float(x) for x in lines[i].strip().split()])
            i += 1
            
            # Read intrinsic parameters
            fx = float(lines[i].strip())
            fy = float(lines[i+1].strip())
            cx = float(lines[i+2].strip())
            cy = float(lines[i+3].strip())
            i += 4
            
            # Read image size
            imgsize = tuple(map(int, lines[i].strip().split()))
            i += 1
            
            cameras.append(Camera(R, T, fx, fy, cx, cy, imgsize))
            
        return cameras
        
    except Exception as e:
        print(f"Error loading calibration file: {str(e)}")
        return None
