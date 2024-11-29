# CVPR Lab: Multiview Geometry
A Python implementation of the Computer Vision and Pattern Recognition lab focusing on multiview geometry, camera calibration, and 3D reconstruction.
## Notes
- Based on original MATLAB code by Miroslaw Bober



## Directory Structure
```
Week10_2020/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── main.py                  # Main lab interface
│
├── exercises/
│   ├── __init__.py
│   ├── fundamental.py       # Fundamental matrix exercises (Ex 1-4)
│   ├── triangulation.py     # 3D reconstruction exercises (Ex 5-6)
│   └── homography.py        # Homography exercises (Ex 7-8)
│
├── utils/
│   ├── __init__.py
│   ├── camera.py           # Camera geometry utilities
│   ├── geometry.py         # Geometric computations
│   └── visualization.py    # Plotting and UI tools
│
└── data/
    ├── calibration.txt     # Camera calibration data
    ├── points2D_cam1.txt   # 2D point observations
    └── images/
        ├── view1.jpg
        └── view2.jpg

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the lab:
   ```bash
   python main.py
   ```

3. Follow the interactive prompts to complete exercises

## Lab Exercises
1. Fundamental Matrix Estimation (8-point algorithm)
2. Understanding Epipolar Geometry
3. Visual Verification of Epipolar Lines
4. Mathematical Verification
5. 3D Point Triangulation
6. Multiple Point Triangulation
7. Homography Estimation
8. Interactive Homography Visualization

## Requirements
### Core Requirements
- Python 3.8 or newer
- NumPy
- SciPy
- OpenCV-Python
- Matplotlib
- Pillow

### Installation
```bash
pip install -r requirements.txt
```
