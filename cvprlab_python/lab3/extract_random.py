
import numpy as np

def extract_random(img):
    # Generate a random row vector with 30 elements
    F = np.random.rand(1, 30)
    # Returns a row [rand rand .... rand] representing an image descriptor
    # computed from image 'img'
    # Note img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
    return F


def extract_rgb(img):
    # Compute the average red value
    red = img[:, :, 0]
    average_red = np.mean(red)
    
    # Compute the average green value
    green = img[:, :, 1]
    average_green = np.mean(green)
    
    # Compute the average blue value
    blue = img[:, :, 2]
    average_blue = np.mean(blue)
    
    # Concatenate the average values to form the feature vector
    F = np.array([average_red, average_green, average_blue])
    
    return F
