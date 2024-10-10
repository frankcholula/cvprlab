import cv2
import numpy as np
import os

def extract_color_descriptor(img):
    # Compute the average red, green, and blue values as a basic color descriptor
    R = np.mean(img[:, :, 2])  # Note: OpenCV uses BGR format
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 0])
    return np.array([R, G, B])

# Ex1: Download the Skeleton code and Dataset
DATASET_FOLDER = 'path_to_dataset'  # Modify this path as needed
OUT_FOLDER = 'path_to_output'  # Modify this path as needed
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

# Ex2: Extract Image Descriptors from the Dataset
Follow instructions from the Lab Worksheet 3 

# Ex3: Run a Visual Search
Follow instructions from the Lab Worksheet 3

# Print results of Ex3: Display the filenames of the top 20 matching images
print("Top 20 matching images:")
for index in top_indices:
    print(image_files[index])

# Ex4: Modify the Distance measure
# Here we can use Euclidean distance. For other measures modify this part accordingly.

# Ex5: Modify the Descriptor Computation
# The function extract_color_descriptor already computes a basic color descriptor.
# For enhancements, modify the extract_color_descriptor function.

# Ex6: (Optional) Modify the Descriptor to a Global Colour Histogram
# This step would involve modifying the extract_color_descriptor to calculate histograms.
# This is placeholder and needs actual implementation if required.

