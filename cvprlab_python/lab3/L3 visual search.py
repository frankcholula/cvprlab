import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Helper function to display images
def display_image(img, title="Image", cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('on')
    plt.show()

# Ex1: Finding Modes of Variation in Data
print("Ex1: Generating 3D data and applying PCA")
# Create 5000 random 2D points in the range [0,1]
pt = np.random.rand(2, 5000)

# Plot 2D points
plt.scatter(pt[0, :], pt[1, :], c='b', marker='x')
plt.title("2D points")
plt.show()

# Turn them into 3D points (z=0 for all)
pt = np.vstack((pt, np.zeros((1, 5000))))

# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pt[0, :], pt[1, :], pt[2, :], c='r', marker='x')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# Multiply the y-coordinate by 5
pt[1, :] = pt[1, :] * 5

# Plot modified 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pt[0, :], pt[1, :], pt[2, :], c='r', marker='x')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# Ex2: Manually Computing the Eigenmodel (PCA)
print("Ex2: Manually computing the PCA")
# Compute the mean of the 3D points
org = np.mean(pt, axis=1).reshape(-1, 1)

# Subtract the mean
ptsub = pt - org

# Compute the covariance matrix
C = np.cov(ptsub)

# Decompose the covariance matrix into eigenvectors and eigenvalues
val, vct = np.linalg.eig(C)

# Print the results
print("Mean:\n", org)
print("Eigenvalues:\n", val)
print("Eigenvectors:\n", vct)

# Ex3: Distance from an Eigenmodel
print("Ex3: Building color model and finding similar pixels")
# Load the target image and normalize it
target = cv2.imread('testimages/target_yellow.bmp').astype(np.float64) / 255.0

# Create a matrix of RGB pixel values (each column is a pixel)
target_obs = np.vstack([target[:, :, i].reshape(1, -1) for i in range(3)])

# Build the Eigenmodel (PCA)
org = np.mean(target_obs, axis=1).reshape(-1, 1)
ptsub = target_obs - org
C = np.cov(ptsub)
val, vct = np.linalg.eig(C)

# Load the test image and normalize it
test = cv2.imread('testimages/kitchen.bmp').astype(np.float64) / 255.0

# Create a matrix of RGB values for the test image
test_obs = np.vstack([test[:, :, i].reshape(1, -1) for i in range(3)])

# Compute Mahalanobis distance manually
invC = np.linalg.inv(C)
xsub = test_obs - org
#mdist_squared = np.sum(xsub.T @ invC @ xsub)

# Compute pixel-wise Mahalanobis distance
mdist_squared = np.sum((xsub.T @ invC) * xsub.T, axis=1)  # Compute distances for each pixel
mdist = np.sqrt(mdist_squared)

# Reshape to match the original image size
result = mdist.reshape(test.shape[:2])

# Normalize result and display
nresult = result / np.max(result)
display_image(nresult, "Mahalanobis Distance Map", cmap='gray')

# Threshold the result at 3 standard deviations
display_image(result < 3, "Thresholded Mahalanobis Distance Map", cmap='gray')

# Ex4: Manually Computing Mahalanobis Distance
print("Ex4: Manually computing Mahalanobis distance for one pixel")
# Select one pixel from test_obs (e.g., first column)
x = test_obs[:, 0].reshape(-1, 1)

# Subtract the mean
xsub = x - org

# Compute Mahalanobis distance manually
mdist_squared = xsub.T @ np.linalg.inv(np.diag(val)) @ vct.T @ vct @ xsub
mdist = np.sqrt(mdist_squared)

# Compare with precomputed Mahalanobis distance
print(f"Manually computed Mahalanobis distance for the first pixel: {mdist[0][0]}")

# Ex5: Now it is your turn to write some Python code, based on Ex.3 :-) 
