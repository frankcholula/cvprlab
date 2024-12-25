"""
SHAPE RECOGNITION LAB
Computer Vision and Pattern Recognition

This lab explores shape recognition using:
- Fourier descriptors for shape representation
- Eigenmodels for pattern classification
- Interactive and batch testing methods

Each exercise builds on concepts from lectures and provides hands-on
experience with shape analysis techniques.

Original MATLAB code by John Collomosse (J.Collomosse@surrey.ac.uk)
Python implementation for teaching purposes.

Requirements:
- numpy
- matplotlib
- opencv-python
- scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import traceback

# Import all required functions from utils package
from utils import (
    draw_shape,
    chaincode,
    chaincode_rasterize,
    sample_polygon_perimeter,
    compute_fd_angular,
    eigen_build,
    eigen_deflate,
    eigen_mahalanobis
)

def display_image(img, title="Image", cmap=None):
    """Helper function to display images with consistent formatting"""
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('on')
    plt.show()

#------------------------------------------------------------------------------
# Exercise 1: Paper Exercise - Understanding Fourier Descriptors
#------------------------------------------------------------------------------
def exercise1_explanation():
    """
    Paper-based exercise explanation - no code required
    Students should:
    1. Sketch the centroid for given shapes
    2. Draw a ray from centroid to edge at angle θ
    3. Plot d(θ) variation for 0-360 degrees
    4. Predict frequency spectrum F(u)
    """
    print("\nExercise 1: Paper exercise - predict central distance Fourier Descriptors")
    print("For each shape (triangle, square, circle, etc.):")
    print("1. Mark the centroid")
    print("2. Draw rays from centroid to boundary")
    print("3. Plot how ray length changes with angle")
    print("4. Sketch expected frequency spectrum")

#------------------------------------------------------------------------------
# Exercise 2: Computing Fourier Descriptors
#------------------------------------------------------------------------------
def fourier_descriptor_demo():
    """
    Interactive demo for computing and visualizing Fourier descriptors.
    
    This demo:
    1. Lets user draw a shape
    2. Computes uniform boundary sampling
    3. Calculates Fourier descriptors
    4. Visualizes both shape and descriptors
    """
    # Let user draw a shape
    print("\nExercise 2: Computing Fourier Descriptors")
    print("Draw a shape (left click to add points, right click to finish)")
    
    try:
        mask, polygon = draw_shape(200)
        
        # Step 1: Sample the polygon perimeter uniformly
        # This ensures consistent descriptor computation
        sampled_points = sample_polygon_perimeter(polygon, 100)
        
        # Step 2: Compute Fourier descriptors
        # We use indices 2-17 to skip the DC component
        descriptors = compute_fd_angular(sampled_points, np.arange(2, 18))
        
        # Step 3: Visualize results
        plt.figure(figsize=(12, 5))
        
        # Plot the sampled shape
        plt.subplot(1, 2, 1)
        plt.plot(sampled_points[0, :], sampled_points[1, :], 'rx-')
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.xlim(0, 200)
        plt.ylim(200, 0)

        plt.title("Your Shape\n(red points = uniform sampling)")
        plt.xlabel("X")
        plt.ylabel("Y")

        
        # Plot the Fourier descriptors
        plt.subplot(1, 2, 2)
        plt.bar(range(len(descriptors)), descriptors)
        plt.title("Shape's Fourier Descriptors\n(frequency components)")
        plt.xlabel("Descriptor Index")
        plt.ylabel("Magnitude")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in Fourier descriptor demo: {str(e)}")
        print("Please try again")

#------------------------------------------------------------------------------
# Exercise 5: Interactive Shape Recognition
#------------------------------------------------------------------------------
def shape_demo_interactive(path='shapeimages'):
    """
    Interactive shape classification using eigenmodels.
    
    Process:
    1. Load training shapes from images
    2. Compute Fourier descriptors for each shape
    3. Build eigenmodel for each category
    4. Let user draw test shape
    5. Classify using Mahalanobis distance
    
    Args:
        path: Directory containing training shape images
    """
    print("\nExercise 5: Interactive Shape Recognition")
    
    # Configuration
    categories = ['arch', 'fish', 'triangle', 'cross', 'square', 'circle']
    SAMPLES = 30  # Training examples per category
    FDMAX = 8    # Number of Fourier descriptors to use

    try:
        # Step 1: Load training data
        print("Loading training shapes...")
        samples = {cat: [] for cat in categories}
        for cat in categories:
            for n in range(1, SAMPLES + 1):
                fname = os.path.join(path, f"{cat}{n:04d}.bmp")
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"Missing training file: {fname}")
                    
                img = cv2.imread(fname, 0)  # Read as grayscale
                mask = img > 0
                cc, start = chaincode(mask)
                samples[cat].append((cc, start))
        
        # Step 2: Compute shape descriptors
        print("Computing shape descriptors...")
        observations = {cat: [] for cat in categories}
        for cat in categories:
            print(f"Processing {cat} shapes...")
            for cc, start in samples[cat]:
                # Convert chain code to polygon and compute descriptors
                mask, polyg = chaincode_rasterize(cc)
                polyg = sample_polygon_perimeter(polyg, 100)
                D = compute_fd_angular(polyg, np.arange(2, FDMAX + 2))
                observations[cat].append(D)
            observations[cat] = np.array(observations[cat]).T
        
        # Step 3: Train eigenmodels
        print("Training classifiers...")
        eigenmodels = {}
        for cat in categories:
            E = eigen_build(observations[cat])
            E = eigen_deflate(E, 'keepf', 0.97)  # Keep 97% of variance
            eigenmodels[cat] = E
        
        # Step 4: Get user's test shape
        print("\nDraw a shape to classify...")
        mask, polygon = draw_shape(200)
        
        # Step 5: Compute query descriptors
        sampled_query = sample_polygon_perimeter(polygon, 100)
        query_fd = compute_fd_angular(sampled_query, np.arange(2, FDMAX + 2))
        
        # Step 6: Classify shape
        scores = []
        for cat in categories:
            score = eigen_mahalanobis(query_fd.reshape(-1, 1), eigenmodels[cat])
            scores.append(score)
        
        # Convert distances to probabilities
        scores = np.array(scores).flatten()
        probabilities = 1 / (scores + 1e-10)  # Add small epsilon to avoid division by zero
        probabilities = probabilities / np.sum(probabilities)
        
        # Display results
        print("\nClassification probabilities:")
        for cat, prob in zip(categories, probabilities):
            print(f"{cat}: {prob:.3f}")
        print(f"\nBest match: {categories[np.argmax(probabilities)]}")
        
    except Exception as e:
        print(f"Error in shape recognition: {str(e)}")
        print("Please check the path to your training images and try again")

#------------------------------------------------------------------------------
# Exercise 6: Batch Shape Recognition
#------------------------------------------------------------------------------
def shape_demo_batch(path='shapeimages', test_proportion=0.25, num_trials=10):
    """
    Batch testing with cross-validation.
    
    This demonstrates:
    - Train/test splitting
    - Multiple evaluation trials
    - Confusion matrix analysis
    - Performance metrics
    
    Args:
        path: Directory containing shape images
        test_proportion: Fraction of data to use for testing (default: 0.25)
        num_trials: Number of cross-validation trials (default: 10)
    """
    print("\nExercise 6: Batch Shape Recognition")
    
    categories = ['arch', 'fish', 'triangle', 'cross', 'square', 'circle']
    SAMPLES = 30
    FDMAX = 8

    try:
        # Load and preprocess all data
        print("Loading and preprocessing shapes...")
        all_data = {cat: [] for cat in categories}
        
        for cat in categories:
            print(f"Processing {cat} shapes...")
            for n in range(1, SAMPLES + 1):
                fname = os.path.join(path, f"{cat}{n:04d}.bmp")
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"Missing training file: {fname}")
                
                # Load and process image
                img = cv2.imread(fname, 0)
                mask = img > 0
                
                # Compute shape descriptors
                cc, start = chaincode(mask)
                mask, polyg = chaincode_rasterize(cc)
                polyg = sample_polygon_perimeter(polyg, 100)
                fd = compute_fd_angular(polyg, np.arange(2, FDMAX + 2))
                all_data[cat].append(fd)
        
        # Run multiple trials
        confusion_matrices = []
        accuracies = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")
            
            # Split data into train/test sets
            train_data = {cat: [] for cat in categories}
            test_data = {cat: [] for cat in categories}
            
            for cat in categories:
                data = np.array(all_data[cat])
                n_test = int(len(data) * test_proportion)
                indices = np.random.permutation(len(data))
                
                test_idx = indices[:n_test]
                train_idx = indices[n_test:]
                
                test_data[cat] = data[test_idx]
                train_data[cat] = data[train_idx]
            
            # Train eigenmodels
            eigenmodels = {}
            for cat in categories:
                E = eigen_build(train_data[cat].T)
                E = eigen_deflate(E, 'keepf', 0.97)
                eigenmodels[cat] = E
            
            # Test classification
            confusion_matrix = np.zeros((len(categories), len(categories)))
            
            for true_cat_idx, true_cat in enumerate(categories):
                for test_sample in test_data[true_cat]:
                    # Classify test sample
                    scores = []
                    for cat in categories:
                        score = eigen_mahalanobis(test_sample.reshape(-1, 1), eigenmodels[cat])
                        scores.append(score)
                    
                    pred_cat_idx = np.argmin(scores)
                    confusion_matrix[true_cat_idx, pred_cat_idx] += 1
            
            # Normalize confusion matrix
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            confusion_matrices.append(confusion_matrix)
            accuracies.append(np.trace(confusion_matrix) / len(categories))
        
        # Compute and display final statistics
        avg_confusion = np.mean(confusion_matrices, axis=0)
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print("\nFinal Results:")
        print(f"Average accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"Random chance would give: {1/len(categories):.3f}")
        
        print("\nConfusion Matrix (rows=true class, columns=predicted class):")
        print("True \\ Pred |", " | ".join(f"{cat:^7}" for cat in categories))
        print("-" * 80)
        for i, cat in enumerate(categories):
            print(f"{cat:^10} |", " | ".join(f"{avg_confusion[i,j]:7.3f}" for j in range(len(categories))))
            
    except Exception as e:
        print(f"Error in batch recognition: {str(e)}")
        print("Please check your setup and try again")

#------------------------------------------------------------------------------
# Main Program
#------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nShape Recognition Lab")
    print("=====================")
    print("\nAvailable exercises:")
    print("1: Paper Exercise (explanation)")
    print("2: Fourier Descriptor Demo")
    print("5: Interactive Shape Recognition")
    print("6: Batch Shape Recognition")
    
    while True:
        choice = input("\nSelect exercise (1/2/5/6) or 'q' to quit: ")
        
        if choice == 'q':
            break
        elif choice == '1':
            exercise1_explanation()
        elif choice == '2':
            fourier_descriptor_demo()
        elif choice == '5':
            path = input("Enter path to shape images (default: 'shapeimages'): ") or 'shapeimages'
            shape_demo_interactive(path)
        elif choice == '6':
            path = input("Enter path to shape images (default: 'shapeimages'): ") or 'shapeimages'
            test_prop = float(input("Enter test proportion (default: 0.25): ") or "0.25")
            num_trials = int(input("Enter number of trials (default: 10): ") or "10")
            shape_demo_batch(path, test_prop, num_trials)
        else:
            print("Please choose exercise 1, 2, 5, or 6")