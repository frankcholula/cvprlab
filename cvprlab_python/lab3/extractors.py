import numpy as np

class Extractors:
    @staticmethod
    def extract_random(img):
        """
        Generate a random row vector with 30 elements.
        
        This function returns a row vector [rand rand .... rand] representing 
        an image descriptor computed from the image 'img'.
        
        Note: 'img' is expected to be a normalized RGB image (colors range [0,1] not [0,255]).
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A random row vector with 30 elements.
        """
        F = np.random.rand(1, 30)
        return F
    
    @staticmethod
    def extract_rgb(img):
        """
        Compute the average red, green, and blue values as a basic color descriptor.
        
        This function calculates the average values for the blue, green, and red channels
        of the input image and returns them as a feature vector.
        
        Note: OpenCV uses BGR format, so the channels are accessed in the order B, G, R.
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A feature vector containing the average B, G, and R values.
        """
        img = img / 255.0  # normalize the image to [0, 1]
        B = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        R = np.mean(img[:, :, 2])
        return np.array([R, G, B])