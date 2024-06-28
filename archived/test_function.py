import os
import cv2
import pandas as pd
import base64
from utils.box_counting import BOXCOUNTING
from utils.cluster import cluster_boxes
# Initialize the BOXCOUNTING model
boxcounting = BOXCOUNTING('models/19junv12.pt',show=True)

import numpy as np
def fill(image):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create a mask of the same size as the image, initialized to zero (black)
    mask = np.zeros_like(image)

    # Fill half of the image with black (left half in this example)
    image[:, :width // 2 + 200] = 0  # Left half
    # image[:, width // 2:] = 0  # Right half
    # image[:height // 2, :] = 0  # Top half
    # image[height // 2:, :] = 0  # Bottom half

    # Apply the mask to the image
    return image
# Define the directory containing the images
image_dir = 'outputs\CevaCrop'
video_path = 'inputs\DJI_0711_processed.MP4'
# List to store results
results = []
cap = cv2.VideoCapture(video_path)
# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    boxcounting.count(fill(frame))








# # Iterate through each file in the directory
# for filename in os.listdir(image_dir):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         image_path = os.path.join(image_dir, filename)
#         image = cv2.imread(image_path)
        
#         if image is not None:
#             result = boxcounting.count(image)
#             print(result)
#             results.append({'filename': filename, 'count': result})
#         else:
#             print(f"Error reading image {filename}")

# # Create a DataFrame from the results
# df = pd.DataFrame(results)

# # Save the DataFrame to a CSV file
# output_csv = 'box_counting_results.csv'
# df.to_csv(output_csv, index=False)

# print(f"Results saved to {output_csv}")
