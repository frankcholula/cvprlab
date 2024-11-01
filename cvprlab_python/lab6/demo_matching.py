import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
IMAGE1 = 'testimages/wall1.jpg'
IMAGE2 = 'testimages/wall2.jpg'

img1 = cv2.imread(IMAGE1)
img2 = cv2.imread(IMAGE2)

# Convert images to grayscale
gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Load the SIFT keypoints and descriptors (assuming they were saved as .npz files)
sift_data = np.load('siftresults.npz')
keypoints1 = sift_data['keypoints1']
keypoints2 = sift_data['keypoints2']
descr1 = sift_data['descr1']
descr2 = sift_data['descr2']

# Match the SIFT features between images using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descr1, descr2)
matches = sorted(matches, key=lambda x: x.distance)

# Plot the matched keypoints on concatenated images
bothimages = np.hstack((img1, img2))
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(bothimages, cv2.COLOR_BGR2RGB))
plt.title('Matched Keypoints Between Images')

# Draw lines between matched points
image2offset = img1.shape[1]  # Width of the first image

STEPSIZE = 50  # Draw 1 in 50 matches
for i in range(0, len(matches), STEPSIZE):
    m = matches[i]
    img1_idx = m.queryIdx
    img2_idx = m.trainIdx

    # Coordinates of the matched keypoints
    img1x, img1y = keypoints1[:, img1_idx]
    img2x, img2y = keypoints2[:, img2_idx] + [image2offset, 0]

    # Plot matched keypoints and lines between them
    plt.plot([img1x, img2x], [img1y, img2y], 'r-')
    plt.scatter([img1x, img2x], [img1y, img2y], c='yellow', marker='x')

plt.axis('off')
plt.show()