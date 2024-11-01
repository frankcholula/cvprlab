import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to images
IMAGE1 = 'testimages/wall1.jpg'
IMAGE2 = 'testimages/wall2.jpg'

# Load images
img1 = cv2.imread(IMAGE1)
img2 = cv2.imread(IMAGE2)

# Convert images to grayscale
gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT detector
sift = cv2.SIFT_create()
keypoints1, descr1 = sift.detectAndCompute(gimg1, None)
keypoints2, descr2 = sift.detectAndCompute(gimg2, None)

# Extract coordinates of the keypoints
keypoints1_coords = np.array([kp.pt for kp in keypoints1]).T
keypoints2_coords = np.array([kp.pt for kp in keypoints2]).T

# Convert descriptors to uint8 type
descr1 = (descr1 * 512).astype(np.uint8)
descr2 = (descr2 * 512).astype(np.uint8)

# Harris corner detection
thresh = 1000  # Number of top corners to keep
corners1 = cv2.goodFeaturesToTrack(gimg1, maxCorners=thresh, qualityLevel=0.01, minDistance=10)
corners2 = cv2.goodFeaturesToTrack(gimg2, maxCorners=thresh, qualityLevel=0.01, minDistance=10)

corners1 = corners1.reshape(-1, 2).T  # Convert to 2D array for plotting
corners2 = corners2.reshape(-1, 2).T  # Convert to 2D array for plotting

# Plot keypoints and corners for image 1
plt.figure(1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(keypoints1_coords[0], keypoints1_coords[1], color='yellow', marker='x', label='SIFT Keypoints')
plt.scatter(corners1[0], corners1[1], color='blue', marker='o', label='Harris Corners')
plt.title(IMAGE1)
plt.legend()

# Plot keypoints and corners for image 2
plt.figure(2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(keypoints2_coords[0], keypoints2_coords[1], color='yellow', marker='x', label='SIFT Keypoints')
plt.scatter(corners2[0], corners2[1], color='blue', marker='o', label='Harris Corners')
plt.title(IMAGE2)
plt.legend()

# Display plots
plt.show()

# Save results
np.savez('siftresults.npz', keypoints1=keypoints1_coords, keypoints2=keypoints2_coords, descr1=descr1, descr2=descr2)
