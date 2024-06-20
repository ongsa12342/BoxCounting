import os
import cv2
import pandas as pd
import base64
from utils.box_counting import BOXCOUNTING

# Initialize the BOXCOUNTING model
boxcounting = BOXCOUNTING('models/19junv12.pt',show=True)


# Define the directory containing the images
image_dir = 'outputs/capO/'

# List to store results
results = []

# Iterate through each file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        if image is not None:
            result = boxcounting.count(image)
            print(result)
            results.append({'filename': filename, 'count': result})
        else:
            print(f"Error reading image {filename}")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_csv = 'box_counting_results.csv'
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
