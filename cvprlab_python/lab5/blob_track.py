# 3032 Labs - Python version - Mirek Bober

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label
import argparse

def aviread(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def Eigen_Build(samps):
    mean = np.mean(samps, axis=1)
    cov = np.cov(samps)
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov

def Eigen_Mahalanobis(samps, eigenmodel):
    mean, inv_cov = eigenmodel
    diff = samps - mean[:, None]
    dst = np.sqrt(np.sum(np.dot(inv_cov, diff) * diff, axis=0))
    return dst

def track_object(file_path):
    F = aviread(file_path)
    if len(F) == 0:
        print("Failed to load video.")
        return

    # Display first frame for region selection
    img = cv2.cvtColor(F[0], cv2.COLOR_BGR2RGB) / 255.0
    img_height, img_width = img.shape[:2]

    plt.imshow(img)
    plt.title("Select top-left and bottom-right corners of the region")
    plt.show(block=False)

    # Capture two clicks for the region
    print("Select the top-left and bottom-right corners of the tracking region.")
    click_points = plt.ginput(2)
    plt.close()

    # Assign the coordinates correctly
    x1, y1 = int(round(click_points[0][0])), int(round(click_points[0][1]))
    x2, y2 = int(round(click_points[1][0])), int(round(click_points[1][1]))

    # Define the top-left and bottom-right points correctly
    topleft = [min(x1, x2), min(y1, y2)]
    botright = [max(x1, x2), max(y1, y2)]

    print(f"Top-left (selected): {topleft}, Bottom-right (selected): {botright}")

    # Initialize samples array as a list of RGB triplets
    samps = []
    for y in range(topleft[1], botright[1]):
        for x in range(topleft[0], botright[0]):
            r, g, b = img[y, x, :]
            samps.append([r, g, b])

    samps = np.array(samps).T  # Ensure samps is a 3 x N array
    if samps.size == 0:
        print("No samples were collected. Check your selected region.")
        return

    e = Eigen_Build(samps)  # Build eigenmodel
    lastpos = np.round((np.array(topleft) + np.array(botright)) / 2)

    # Track from frame 2 onward
    history = []
    for frame in F[1:]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        samps = np.vstack([img[:, :, 0].ravel(), img[:, :, 1].ravel(), img[:, :, 2].ravel()])
        dst = Eigen_Mahalanobis(samps, e)
        dst = dst.reshape(img.shape[:2])
        thresholded = dst < 3
        map = label(thresholded)
        reglabels = np.setdiff1d(np.unique(map), 0)
        print(f'There are {len(reglabels)} connected components')

        possible_centroids = {}
        for cc in reglabels:
            mask = (map == cc)
            y, x = np.where(mask)
            possible_centroids[cc] = np.array([np.mean(x), np.mean(y)])  # Ensure numpy array

        bestdist = np.inf
        bestidx = -1
        for cc in reglabels:
            thisdist = np.linalg.norm(possible_centroids[cc] - lastpos)  # Subtracting numpy arrays
            if thisdist < bestdist:
                bestdist = thisdist
                bestidx = cc

        currentpos = possible_centroids[bestidx]
        history.append(currentpos)

        # Display tracking
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(np.dstack([thresholded.astype(np.uint8)]*3))  # Display thresholded in RGB
        history_arr = np.array(history)
        plt.plot(history_arr[:, 0], history_arr[:, 1], 'y-')
        plt.plot(currentpos[0], currentpos[1], 'm*')
        plt.title(f'Frame {len(history) + 1}')
        plt.draw()
        plt.pause(0.01)

        lastpos = currentpos

# Command Line Interface (CLI) support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track an object in a video file.")
    parser.add_argument("video_file", type=str, help="Path to the video file (e.g., 'video.avi')")
    
    args = parser.parse_args()
    
    # Call the function with the video file provided as an argument
    track_object(args.video_file)

