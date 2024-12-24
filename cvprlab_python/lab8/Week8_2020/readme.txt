Contour Lab
===========

This lab implements various curve and contour algorithms for teaching Computer Vision concepts.

Directory Structure
-----------------
contour-lab/
├── main.py           # Main lab file with all exercises
├── utils.py          # Helper functions
├── requirements.txt  # Python package requirements
└── data/            # Folder for images
    ├── hand.jpg
    └── snakefield.bmp

Setup Instructions
----------------

Windows Setup:
1. Open Command Prompt
2. Navigate to the project directory:
   > cd path\to\contour-lab
3. Create a virtual environment:
   > python -m venv venv
4. Activate the virtual environment:
   > venv\Scripts\activate
5. Install requirements:
   > pip install -r requirements.txt

Linux/Mac Setup:
1. Open terminal
2. Navigate to project directory:
   $ cd path/to/contour-lab
3. Create virtual environment:
   $ python3 -m venv venv
4. Activate virtual environment:
   $ source venv/bin/activate
5. Install requirements:
   $ pip install -r requirements.txt

Running the Lab
-------------
1. Ensure your virtual environment is activated
2. Run the main script:
   Windows: > python main.py
   Linux/Mac: $ python3 main.py

Available Exercises
-----------------
1. Hermite Curves
   - Demonstrates C1 continuous curve segments
   - Understand curve construction and tangent effects
   - Visualize how control points and tangents affect curve shape

2. Bezier Curves
   - Interactive demo for placing control points
   - Visualize control polygon and resulting curve
   - Understand approximating vs interpolating behavior
   - Shows relationship between control points and curve shape

3. Spline Demo
   - Compare Catmull-Rom and B-spline behaviors
   - Interactive control point placement
   - Visualize different spline properties
   - Understand effect of different blending matrices

4. Snake Demo (Synthetic Data)
   - Active contour on synthetic test image
   - Demonstrates effect of different energy terms:
     * α (alpha): controls continuity
     * β (beta): controls smoothness
     * γ (gamma): controls image force
   - Visualize snake evolution and convergence

5. Snake Demo (Hand Image)
   - Apply active contours to real image data
   - Hand contour extraction example
   - Parameter adjustment for optimal fitting
   - Real-world application of snakes algorithm

Controls
--------
- Left mouse click: Place control points (Ex 1-3)
- Close window: Proceed to next step
- Menu selection: Choose exercises

Data Requirements
---------------
Ensure the following files are in the data/ folder:
- hand.jpg (for snake demo on real image)
- snakefield.bmp (for snake demo on synthetic data)

Required Packages
---------------
- numpy: Array operations and numerical computing
- matplotlib: Plotting and visualization
- opencv-python: Image processing and computer vision
- scipy: Scientific computing and optimization

Troubleshooting
-------------
Common issues:

1. ImportError: No module named 'cv2'
   Solution: Run 'pip install opencv-python'

2. matplotlib display issues:
   - Linux: Install with 'sudo apt-get install python3-tk'
   - Windows: Try running with pythonw instead of python

3. Image loading errors:
   - Check that images exist in data/ folder
   - Verify image format is correct
   - Check file permissions

4. Display issues:
   - Close all figures between exercises
   - Use plt.show() to force display updates

5. Snake convergence issues:
   - Try adjusting α, β, γ parameters
   - Increase number of iterations
   - Check image edge map quality

Notes
-----
- This is a Python implementation of the Contour lab
- Each exercise builds on concepts from lectures
- Interactive demonstrations help understand theoretical concepts
- Parameter experimentation is encouraged for better understanding