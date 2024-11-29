Shape Recognition Lab
===================

This lab package contains implementations for shape recognition using Fourier descriptors 
and eigenmodel-based classification.

Directory Structure
-----------------
Week7_2020/
│
├── shape_recognition_lab.py    # Main lab file
├── README.txt                  # This file
├── requirements.txt            # Python package requirements
├── requirements_conda.txt      # Conda environment requirements
│
└── utils/                      # Utility functions
    ├── __init__.py
    ├── draw_shape.py
    ├── chaincode.py
    ├── chaincode_rasterize.py
    ├── sample_polygon_perimeter.py
    ├── fourier_descriptors.py
    └── eigenmodel.py

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
     > conda create --name shape_lab python=3.8
   - Activate environment:
     > conda activate shape_lab
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
     $ conda create --name shape_lab python=3.8
   - Activate environment:
     $ conda activate shape_lab
   - Install requirements:
     $ conda install --file requirements_conda.txt

Running the Lab
-------------
1. Ensure your virtual environment is activated
2. Run the main script:
   Windows: > python shape_recognition_lab.py
   Linux:   $ python3 shape_recognition_lab.py

Available Exercises
-----------------
1. Paper exercise – predict Fourier Descriptors
2. Computing Fourier Descriptors demo
3. Code understanding exercise
4. Angular Fourier Descriptors
5. Interactive Shape Recognition
6. Batch Shape Recognition
7. Optional comparison of descriptors

Data Requirements
---------------
- Create a folder named 'shapeimages'
- Extract shape image dataset in this folder
- Supported image format: BMP
- Expected naming convention: category####.bmp (e.g., circle0001.bmp)

Troubleshooting
-------------
Common issues:

1. ImportError: No module named 'cv2'
   - Ensure OpenCV is installed: pip install opencv-python

2. matplotlib display issues:
   - Linux: Install: sudo apt-get install python3-tk
   - Windows: Try running with pythonw instead of python

3. Permission errors:
   - Windows: Run Command Prompt as administrator
   - Linux: Use sudo for system-wide installation
