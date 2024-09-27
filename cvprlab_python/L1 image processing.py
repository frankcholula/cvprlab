import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Helper function to display images
def display_image(img, title="Image", cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('on')
    plt.show()

# Ex1: Loading an image
img = cv2.imread('testimages/sphinx.jpg')  # Make sure to use your actual image path
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
display_image(img, "Original Image")

# Ex2: Colour image representation
print(f"Image size: {img.shape}")  # Shows (height, width, channels)

# Access RGB value of pixel (2,3)
p = img[2, 3]  # OpenCV uses (y,x) format
print(f"RGB value at (2,3): {p}")

# Ex3: Working with ranges of pixels (setting part of the image to white)
img_copy = img.copy()  # Create a copy of the image
img_copy[200:300, 100:150] = [255, 255, 255]  # Set to white
display_image(img_copy, "Modified Image")

# Crop part of the image
subimg = img[200:300, 100:150]
display_image(subimg, "Cropped Image")

# Ex4: Saving an image
cv2.imwrite('out.jpg', cv2.cvtColor(subimg, cv2.COLOR_RGB2BGR))  # Save as JPEG
print("Cropped image saved as 'out.jpg'")

# Ex5: Normalized images
norm_img = img / 255.0  # Normalize the pixel values to the range 0-1
display_image(norm_img, "Normalized Image")

# Ex6: Grayscale conversion
grey_img = 0.30 * norm_img[:, :, 0] + 0.59 * norm_img[:, :, 1] + 0.11 * norm_img[:, :, 2]  # Convert to grayscale
display_image(grey_img, "Grayscale Image", cmap='gray')

# Ex7: Filter for blur (low-pass filter)
# Create a 3x3 mean filter
K = np.ones((3, 3)) / 9
blurred_img = convolve2d(grey_img, K, mode='same')  # Apply the filter using convolution
display_image(blurred_img, "Blurred Image", cmap='gray')

# Ex8: Filter for edge detection (Sobel)
# Sobel filter kernels
Kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4
Ky = np.transpose(Kx)

# Convolve to get derivatives
dx = convolve2d(grey_img, Kx, mode='same')
dy = convolve2d(grey_img, Ky, mode='same')

# Compute the magnitude of the gradient
mag = np.sqrt(dx**2 + dy**2)

# Display the edge-detected image
display_image(mag, "Edge Detection (Sobel)", cmap='gray')

# Ex9: Thresholding an image
threshold = 0.15
thresholded_img = (mag > threshold).astype(np.uint8)

# Display the thresholded image (binary mask)
display_image(thresholded_img, "Thresholded Image", cmap='gray')

