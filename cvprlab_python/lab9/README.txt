Visual Hull Lab
===================
This lab package implements visual hull reconstruction from multiple calibrated camera views, 
allowing 3D reconstruction of objects using silhouette information.

Directory Structure
-----------------
Week9_2020/
│
├── visual_hull/                # Main package
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Main lab script
│   ├── camera.py              # Camera calibration functions
│   ├── visualization.py        # Visualization tools
│   └── reconstruction.py       # Visual hull reconstruction
│
├── mvdata/                     # Data directory
│   ├── calibration.txt        # Camera parameters
│   ├── cam0.png               # Camera view 0
│   ├── cam1.png               # Camera view 1
│   └── ...                    # Additional views
│
├── README.md                   # This file
├── requirements.txt            # Python package requirements 
└── requirements_conda.txt      # Conda environment requirements

Setup Instructions
----------------
Windows Setup:
1. Using pip:
   - Open Command Prompt as administrator
   - Navigate to the project directory:
     > cd path\to\Week7_2020
   - Create a virtual environment:
     > python -m venv venv
   - Activate the virtual environment:
     > venv\Scripts\activate
   - Install requirements:
     > pip install -r requirements.txt

2. Using Conda:
   - Open Anaconda Prompt
   - Navigate to project directory:
     > cd path\to\Week7_2020
   - Create conda environment:
     > conda create --name visual_hull python=3.8
   - Activate environment:
     > conda activate visual_hull
   - Install requirements:
     > conda install --file requirements_conda.txt

Linux Setup:
1. Using pip:
   - Open terminal
   - Navigate to project directory:
     $ cd path/to/Week7_2020
   - Create virtual environment:
     $ python3 -m venv venv
   - Activate virtual environment:
     $ source venv/bin/activate
   - Install requirements:
     $ pip install -r requirements.txt

2. Using Conda:
   - Open terminal
   - Navigate to project directory:
     $ cd path/to/Week7_2020
   - Create conda environment:
     $ conda create --name visual_hull python=3.8
   - Activate environment:
     $ conda activate visual_hull
   - Install requirements:
     $ conda install --file requirements_conda.txt

Running the Lab
-------------
1. Ensure your virtual environment is activated
2. Run the main script:
   Windows: > python -m visual_hull.main
   Linux:   $ python3 -m visual_hull.main

Available Exercises
-----------------
1. Visualize Cameras (5 minutes)
   - Load calibration data
   - View 3D camera positions
   - Interactive visualization

2. Generate Masks (15 minutes)
   - Process multi-view images
   - Create binary masks
   - Background segmentation

3. Generate 3D Reconstruction (25 minutes)
   - Implement visual hull algorithm
   - Project points to camera views
   - Create 3D reconstruction

Data Requirements
---------------
- Create folder 'mvdata'
- Required files:
  * calibration.txt: Camera parameters
  * cam0.png through cam7.png: Camera view images
- Image format: PNG
- Camera calibration format: See example calibration.txt

Troubleshooting
-------------
Common issues:

1. ImportError: No module named 'cv2'
   - Ensure OpenCV is installed: 
     > pip install opencv-python

2. matplotlib display issues:
   - Linux: Install: sudo apt-get install python3-tk
   - Windows: Try running with pythonw instead of python

3. File not found errors:
   - Check mvdata directory exists and contains required files
   - Verify file permissions

4. Memory errors during reconstruction:
   - Reduce resolution by increasing step sizes
   - Decrease reconstruction volume bounds

5. Permission errors:
   - Windows: Run Command Prompt as administrator
   - Linux: Use sudo for system-wide installation

For additional help, contact course staff or refer to lab documentation.