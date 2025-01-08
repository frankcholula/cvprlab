"""
CONTOUR LAB
Computer Vision and Pattern Recognition

This lab explores different types of curves and contours using:
- Hermite curves for C1 continuous curve generation
- Bezier curves for interactive shape design
- Splines (Catmull-Rom and B-splines) for smooth interpolation
- Active contours (Snakes) for image segmentation

Each exercise builds on concepts from lectures and provides hands-on
experience with curve generation and contour fitting techniques.

Original MATLAB code from Computer Vision Lab Sheet (Week 8)
Python implementation for teaching purposes.

Requirements:
- numpy       (numerical computations and array operations)
- matplotlib  (plotting and interactive visualization)
- opencv-python (image processing and computer vision)
- scipy       (scientific computing and optimization)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import cv2
from utils import display_image, normalize_image

#------------------------------------------------------------------------------
# Exercise 1: Hermite Curves
#------------------------------------------------------------------------------
def exercise1():
    """
    Hermite Curve Demo - Demonstrates C1 continuous curve segments
    
    Learning Objectives:
    - Understand Hermite curve construction
    - Visualize how control points and tangents affect the curve
    - See C1 continuity between curve segments
    """
    print("\nExercise 1: Hermite Curves")
    
    # Define first curve segment
    M = np.array([
        [2, -3, 0, 1],    # Hermite basis matrix
        [-2, 3, 0, 0],
        [1, -2, 1, 0],
        [1, -1, 0, 0]
    ])
    
    # Points and tangents for first curve
    G1 = np.array([
        [0, 9, 0, 0],     # x coordinates and tangents
        [0, 0, 1, -1]     # y coordinates and tangents
    ])
    
    # Points and tangents for second curve
    G2 = np.array([
        [9, 18, 0, 0],
        [0, 0, -1, 1]
    ])
    
    # Generate curve points
    t = np.linspace(0, 1, 100)
    Q = np.vstack([t**3, t**2, t, np.ones_like(t)])
    
    # Calculate curves
    P1 = G1 @ M @ Q
    P2 = G2 @ M @ Q
    
    # Display results
    plt.figure(figsize=(8, 8))
    plt.plot(P1[0, :], P1[1, :], 'b-', label='First Segment')
    plt.plot(P2[0, :], P2[1, :], 'r-', label='Second Segment')
    plt.grid(True)
    plt.legend()
    plt.title('Hermite Curves - Two C1 Continuous Segments')
    plt.axis('equal')
    plt.show()

#------------------------------------------------------------------------------
# Exercise 2: Bezier Curves
#------------------------------------------------------------------------------
def exercise2():
    """
    Interactive Bezier Curve Demo
    
    Learning Objectives:
    - Understand Bezier curve construction
    - See how control points influence the curve
    - Visualize the control polygon
    
    Instructions:
    - Click to place 4 control points
    - See the resulting Bezier curve and control polygon
    """
    print("\nExercise 2: Bezier Curves - Click to place 4 control points")
    
    points = []  # To store control points
    
    def on_click(event):
        # Handle mouse clicks to collect control points
        if event.button is MouseButton.LEFT and len(points) < 4:
            points.append([event.xdata, event.ydata])
            plt.plot(event.xdata, event.ydata, 'r*')
            plt.draw()
            
            # Once we have all points, draw the curve
            if len(points) == 4:
                # Bezier matrix
                M = np.array([
                    [-1, 3, -3, 1],
                    [3, -6, 3, 0],
                    [-3, 3, 0, 0],
                    [1, 0, 0, 0]
                ])
                
                # Setup and calculate curve points
                G = np.array(points).T
                t = np.linspace(0, 1, 100)
                Q = np.vstack([t**3, t**2, t, np.ones_like(t)])
                
                curve_points = (G @ M @ Q).T
                
                # Draw control polygon and curve
                plt.plot([p[0] for p in points], [p[1] for p in points], 
                        'r--', label='Control Polygon')
                plt.plot(curve_points[:, 0], curve_points[:, 1], 
                        'b-', label='Bezier Curve')
                plt.legend()
                plt.draw()
    
    # Setup the plot
    plt.figure(figsize=(8, 8))
    plt.title('Bezier Curve Demo\nClick to place 4 control points')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim (-10, 10)
    plt.ylim(-10, 10)
    
    # Connect click event
    plt.connect('button_press_event', on_click)
    plt.show()

#------------------------------------------------------------------------------
# Exercise 3: Spline Demonstration
#------------------------------------------------------------------------------
def exercise3():
    """
    Demonstrates both Catmull-Rom and B-spline curves.
    
    Learning Objectives:
    - Compare different spline behaviors
    - Understand effect of blending matrix choice
    - Visualize control point influence
    """
    points = []
    
    def on_click(event):
        if event.button is MouseButton.LEFT:
            points.append([event.xdata, event.ydata])
            plt.plot(event.xdata, event.ydata, 'rx')
            plt.draw()
            
            if len(points) >= 4:
                # Setup parameters
                t = np.linspace(0, 1, 100)
                Q = np.vstack([t**3, t**2, t, np.ones_like(t)])
                
                # Catmull-Rom matrix
                Mcr = 0.5 * np.array([
                    [-1, 2, -1, 0],
                    [3, -5, 0, 2],
                    [-3, 4, 1, 0],
                    [1, -1, 0, 0]
                ])
                
                # B-spline matrix
                Mbs = (1/6) * np.array([
                    [-1, 3, -3, 1],
                    [3, -6, 0, 4],
                    [-3, 3, 3, 1],
                    [1, 0, 0, 0]
                ])
                
                # Create master points array
                G = np.array(points).T
                
                # Draw curves
                plt.clf()
                plt.xlim(-10, 10)
                plt.ylim(-10, 10)
                plt.title('Spline Comparison')
                plt.plot([p[0] for p in points], [p[1] for p in points], 'rx')
                
                # Generate curve segments
                for i in range(len(points)-3):
                    # Get 4 consecutive points
                    G_seg = G[:, i:i+4]
                    
                    # Calculate curves
                    P_cr = G_seg @ Mcr @ Q
                    P_bs = G_seg @ Mbs @ Q
                    
                    # Plot curves
                    plt.plot(P_cr[0, :], P_cr[1, :], 'g-', label='Catmull-Rom' if i==0 else '')
                    plt.plot(P_bs[0, :], P_bs[1, :], 'b--', label='B-spline' if i==0 else '')
                
                plt.grid(True)
                plt.legend()
                plt.draw()
    
    plt.figure(figsize=(8, 8))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title('Click to add control points (minimum 4)')
    plt.grid(True)
    plt.connect('button_press_event', on_click)
    plt.show()

#------------------------------------------------------------------------------
# Exercise 4: Snake on Synthetic Data
#------------------------------------------------------------------------------
def exercise4():
    """
    Snake (active contour) demo on synthetic data.
    
    Learning Objectives:
    - Understand snake energy terms
    - Effect of alpha (continuity) parameter
    - Effect of beta (smoothness) parameter
    - Effect of gamma (image force) parameter
    """
    try:
        # Load synthetic field
        field = cv2.imread('data/snakefield_synthetic.bmp', cv2.IMREAD_GRAYSCALE)
        if field is None:
            raise ValueError("Could not load synthetic field image")
        field = field.astype(float) / 255.0
        
        # Initialize snake
        t = np.linspace(0, 2*np.pi, 90)
        center = np.array([field.shape[1]/2, field.shape[0]/2])
        radius = min(field.shape) / 3
        snake_points = np.vstack([
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t)
        ])
        
        # Parameters (can be modified to see effects)
        alpha = 1.0  # Continuity weight
        beta = 1.1   # Smoothness weight
        gamma = 1.2  # Edge weight
        iterations = 100
        
        # Optimize snake
        current_points = snake_points.copy()
        for _ in range(iterations):
            for i in range(current_points.shape[1]):
                x, y = current_points[:, i].astype(int)
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
                        current_points[:, i] = [
                            x + (min_x - 1),
                            y + (min_y - 1)
                        ]
        
        # Display results
        plt.figure(figsize=(10, 10))
        plt.imshow(field, cmap='gray')
        plt.plot(snake_points[0, :], snake_points[1, :], 'r--', label='Initial')
        plt.plot(current_points[0, :], current_points[1, :], 'g-', label='Final')
        plt.title(f'Snake Demo (α={alpha}, β={beta}, γ={gamma})')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Error in snake demo: {str(e)}")

#------------------------------------------------------------------------------
# Exercise 5: Snake on Real Data (Hand Image)
#------------------------------------------------------------------------------
def exercise5():
    """
    Snake (active contour) demo on hand image.
    
    Learning Objectives:
    - Apply snakes to real image data
    - Handle real-world edge conditions
    - Adjust parameters for optimal fitting
    """
    try:
        # Load and prepare hand image
        img = cv2.imread('data/bluehand.jpg', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load hand image")
        
        # Create edge map
        edges = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)**2
        edges += cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)**2
        edges = normalize_image(np.sqrt(edges))
        
        # Initialize snake with more points for hand
        t = np.linspace(0, 2*np.pi, 150)  # More points for complex shape
        center = np.array([img.shape[1]/2, img.shape[0]/2])
        radius = min(img.shape) / 3
        snake_points = np.vstack([
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t)
        ])
        
        # Modified parameters for hand image
        alpha = 0.5  # Lower continuity for better detail
        beta = 0.5   # Lower smoothness for better detail
        gamma = 1.8  # Higher edge attraction
        iterations = 100
        
        # Optimize snake
        current_points = snake_points.copy()
        for _ in range(iterations):
            for i in range(current_points.shape[1]):
                x, y = current_points[:, i].astype(int)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    neighborhood = edges[
                        max(0, y-1):min(img.shape[0], y+2),
                        max(0, x-1):min(img.shape[1], x+2)
                    ]
                    if neighborhood.size > 0:
                        min_y, min_x = np.unravel_index(
                            np.argmin(neighborhood),
                            neighborhood.shape
                        )
                        current_points[:, i] = [
                            x + (min_x - 1),
                            y + (min_y - 1)
                        ]
        
        # Display results
        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        
        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.plot(snake_points[0, :], snake_points[1, :], 'r--', label='Initial')
        plt.plot(current_points[0, :], current_points[1, :], 'g-', label='Final')
        plt.title(f'Snake Result (α={alpha}, β={beta}, γ={gamma})')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Error in hand snake demo: {str(e)}")


if __name__ == '__main__':
    print("\nContour Lab")
    print("===========")
    print("\nAvailable exercises:")
    print("1: Hermite Curves")
    print("2: Bezier Curves")
    print("3: Spline Demo (Catmull-Rom and B-spline)")
    print("4: Snake Demo (Synthetic Data)")
    print("5: Snake Demo (Hand Image)")
    
    while True:
        choice = input("\nSelect exercise (1/2/3/4/5) or 'q' to quit: ")
        
        if choice == 'q':
            break
        elif choice == '1':
            exercise1()
        elif choice == '2':
            exercise2()
        elif choice == '3':
            exercise3()
        elif choice == '4':
            exercise4()
        elif choice == '5':
            exercise5()
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5")
