import os
import numpy as np
import cv2
import scipy.io as sio

DATASET_FOLDER = 'c:/visiondemo/cwsolution/MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = 'c:/visiondemo/cwsolution/descriptors'
OUT_SUBFOLDER = 'globalRGBhisto'

# Ensure the output directory exists
os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

# Iterate through all BMP files in the dataset folder
for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
    if filename.endswith(".bmp"):
        print(f"Processing file {filename}")
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
        img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
        fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
        
        # Call extractRandom (or another feature extraction function) to get the descriptor
        F = extract_random(img)
        
        # Save the descriptor to a .mat file
        sio.savemat(fout, {'F': F})

