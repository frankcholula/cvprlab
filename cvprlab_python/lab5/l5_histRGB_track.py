import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, square
from skimage.measure import label
import argparse

def getTrackDescriptor(img, Q=4):
    """
    Create a track descriptor from an image window using an RGB histogram.
    """
    # Split the image into RGB channels and reshape each to a flat array
    red, green, blue = img[:, :, 0].ravel(), img[:, :, 1].ravel(), img[:, :, 2].ravel()

    # Quantize the RGB values and create histogram bins
    sf = 1.0 / Q
    red_index = np.floor(red / sf).astype(int)
    green_index = np.floor(green / sf).astype(int)
    blue_index = np.floor(blue / sf).astype(int)

    # Create a unique bin index for each quantized RGB triplet
    N = red_index * (Q**2) + green_index * Q + blue_index
    bins = Q ** 3
    histogram, _ = np.histogram(N, bins=np.arange(bins + 1))
    
    # Normalize the histogram
    return histogram / np.sum(histogram)

def aviread(file_path):
    """
    Load frames from a video file.
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def track_object(file_path):
    """
    Track an object in a video based on a selected region.
    """
    F = aviread(file_path)
    if len(F) == 0:
        print("Failed to load video.")
        return

    # Display first frame for region selection
    img = (F[0] / 255.0).astype(np.float32)  # Convert to float32
    img_height, img_width = img.shape[:2]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    targetwnd = img[topleft[1]:botright[1], topleft[0]:botright[0], :]
    targetdesc = getTrackDescriptor(targetwnd)
    winh, winw, _ = targetwnd.shape
    lastpos = np.array(topleft) + (np.array(botright) - np.array(topleft)) / 2

    # Track from frame 2 onwards
    history = [lastpos]
    for i in range(1, len(F)):
        img = (F[i] / 255.0).astype(np.float32)  # Convert to float32
        scores = np.ones((img.shape[0], img.shape[1])) * np.inf
        halfwinw, halfwinh = winw // 2, winh // 2
        step = 2

        for x in range(halfwinw, img.shape[1] - halfwinw, step):
            for y in range(halfwinh, img.shape[0] - halfwinh, step):
                # Extract window and get descriptor
                wnd = img[y - halfwinh:y + halfwinh, x - halfwinw:x + halfwinw, :]
                thisdesc = getTrackDescriptor(wnd)
                # Calculate Euclidean distance between descriptors
                eucdst = np.sqrt(np.sum((targetdesc - thisdesc) ** 2))
                scores[y, x] = eucdst

        # Threshold the descriptor distances
        thresholded = scores < 0.25
        thresholded = dilation(thresholded, square(3))

        # Find all the centroids of the connected components
        map = label(thresholded)
        reglabels = np.setdiff1d(np.unique(map), 0)
        print(f'There are {len(reglabels)} connected components')

        possible_centroids = []
        for cc in reglabels:
            mask = (map == cc)
            y, x = np.where(mask)
            centroid = np.array([np.mean(x), np.mean(y)])
            possible_centroids.append(centroid)

        # Find the best match based on distance to last position
        bestdist = np.inf
        bestidx = -1
        for idx, centroid in enumerate(possible_centroids):
            thisdist = np.linalg.norm(centroid - lastpos)
            if thisdist < bestdist:
                bestdist = thisdist
                bestidx = idx

        # If no best match is found, skip the frame
        if bestidx == -1:
            print('No match found! Skipping frame.')
            continue

        # Update current position and add to history
        currentpos = possible_centroids[bestidx]
        history.append(currentpos)
        
        # Display tracking
        plt.clf()
        thresholded_rgb = np.dstack([thresholded.astype(np.uint8)]*3)
        combined_img = np.hstack((img, thresholded_rgb))
        plt.imshow(combined_img)
        history_arr = np.array(history)
        plt.plot(history_arr[:, 0], history_arr[:, 1], 'y-')
        plt.plot(currentpos[0], currentpos[1], 'm*')
        plt.title(f'Frame {i+1}')
        plt.draw()
        plt.pause(0.1)

        lastpos = currentpos

# Command Line Interface (CLI) support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track an object in a video file.")
    parser.add_argument("video_file", type=str, help="Path to the video file (e.g., 'video.avi')")
    args = parser.parse_args()
    track_object(args.video_file)

