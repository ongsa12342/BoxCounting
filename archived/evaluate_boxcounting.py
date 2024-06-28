import os
import cv2
import pandas as pd
from utils.box_counting import BOXCOUNTING

# Initialize the BOXCOUNTING model
boxcounting = BOXCOUNTING('models/19junv12.pt',show=False     , conf_threshold=0.3)

boxcounting2 = BOXCOUNTING('models/19junv12.pt',show=True     , conf_threshold=0.3)
# Define the directory containing the images
image_dir = 'outputs\CevaCrop'

# List to store results
results = []
true_count = 0
false_count = 0
import numpy as np


# Iterate through each file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        
        
        if image is not None:
            print(f"{filename}: { boxcounting.count(image)}")
            result = boxcounting2.count(image)
            

            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('t'):
                true_count += 1
                results.append({'filename': filename, 'count': result, 'label': 'true'})
                cv2.putText(image, 'True', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif key == ord('f'):
                false_count += 1
                results.append({'filename': filename, 'count': result, 'label': 'false'})
                cv2.putText(image, 'False', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("Invalid key pressed. Please press 't' for true or 'f' for false.")
                continue

            # Display the updated image with the label
            cv2.imshow('Image', cv2.resize(image, (1080, 720)))
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        else:
            print(f"Error reading image {filename}")

# Calculate accuracy
total_count = true_count + false_count
accuracy = (true_count / total_count) * 100 if total_count > 0 else 0

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)

# Print summary
print(f"Total images: {total_count}")
print(f"True: {true_count}")
print(f"False: {false_count}")
print(f"Accuracy: {accuracy:.2f}%")
