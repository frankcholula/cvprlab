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
        with open("mvdata/calibration.txt", "r") as f:
            lines = [line.strip() for line in f.readlines()]

        num_cams = int(lines[0].split()[0])
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

            cameras.append(Camera(R, T, fx, fy, cx, cy, imgsize))
            i += 7

        return cameras
    except Exception as e:
        print(f"Error loading calibration file: {str(e)}")
        return None
