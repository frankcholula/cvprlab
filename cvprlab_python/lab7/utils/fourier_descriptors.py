import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def compute_fd(p, n=None, debug=False):
    """
    Compute the centroid distance Fourier descriptors of a closed polygon.

    Args:
        p (np.ndarray): Regularly sampled polygon vertices [x1,x2..;y1,y2..]
        n (np.ndarray): Optional indices of descriptors to compute
        debug (bool): Optional flag to plot signal pre-FFT

    Returns:
        np.ndarray: Fourier descriptors
    """
    if p.shape[1] < 3:
        raise ValueError("Polygon must have at least 3 points")

    # Handle optional argument n
    if n is None:
        n = np.arange(1, p.shape[1] // 2 + 1)

    # Compute centroid
    mn = np.mean(p, axis=1, keepdims=True)

    # Compute distance of each point to centroid
    p_centered = p - mn
    centdist = np.sqrt(np.sum(p_centered**2, axis=0))

    if debug:
        plt.figure()
        plt.plot(centdist)
        plt.title("Centroid Distance Signal")
        plt.show()

    # Fourier decomposition of signal "centdist"
    F = np.fft.fft(centdist)

    # Compute magnitude of each frequency component
    mag = np.sqrt(F.real**2 + F.imag**2)

    # Return the requested descriptors
    return mag[n]


def compute_fd_angular(p, n=None, debug=False):
    """
    Compute angular Fourier descriptors of a closed polygon.

    Args:
        p (np.ndarray): Regularly sampled polygon vertices [x1,x2..;y1,y2..]
        n (np.ndarray): Optional indices of descriptors to compute
        debug (bool): Optional flag to plot signal pre-FFT

    Returns:
        np.ndarray: Fourier daescriptors
    """
    if p.shape[1] < 3:
        raise ValueError("Polygon must have at least 3 points")

    if n is None:
        n = np.arange(1, p.shape[1] // 2 + 1)

    # Compute vectors around polygon
    v = np.hstack((p[:, 1:], p[:, :1])) - p

    # Normalize vectors
    v_norm = np.sqrt(np.sum(v**2, axis=0))
    v = v / v_norm

    # Add z-coordinate for cross product
    v = np.vstack((v, np.zeros(v.shape[1])))

    # Compute cross product between normalized vectors
    v_next = np.hstack((v[:, 1:], v[:, :1]))
    cross_v = np.cross(v_next.T, v.T)

    # Extract sine of angles
    sin_theta = cross_v[:, 2]

    if debug:
        plt.figure()
        plt.plot(sin_theta)
        plt.title("Angle Signal")
        plt.show()

    # Fourier decomposition
    F = np.fft.fft(sin_theta)

    # Compute magnitude
    mag = np.sqrt(F.real**2 + F.imag**2)

    return mag[n]
