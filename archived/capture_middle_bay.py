import os
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd

def extract_tags(file_path, tab, target):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Extract direction and target
    direction_filter = target.split('-')[1]
    target = target.split('-')[0]

    # Load the specified sheet
    pa_zone_df = pd.read_excel(file_path, sheet_name=tab)

    # Filter columns based on direction
    if direction_filter == 'R':
        pa_zone_df = pa_zone_df.loc[:, ~pa_zone_df.columns.str.contains('L')]
    elif direction_filter == 'L':
        pa_zone_df = pa_zone_df.loc[:, ~pa_zone_df.columns.str.contains('R')]

    # Filter rows containing the target string in the 'direction' column and offset 0.0 in the 'OFFSET:' column
    filtered_pa_zone_df = pa_zone_df[
        (pa_zone_df['direction'].astype(str).str.contains(target)) & 
        (pa_zone_df['OFFSET:'] == 0.0)
    ]
    
    # Extract numerical tags from these filtered rows, excluding 0.0
    filtered_tags = []
    for _, row in filtered_pa_zone_df.iterrows():
        for item in row:
            if isinstance(item, (int, float)) and item != 0.0 and not pd.isnull(item):
                filtered_tags.append(item)
    
    # Create a DataFrame from the filtered tags
    filtered_tags_df = pd.DataFrame(filtered_tags, columns=['Tag'])
    
    return filtered_tags_df

def detect_aruco_in_center(frame, aruco_dict, parameters, tag_list, range_percentage=0.01):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        # Determine center region
        frame_width = frame.shape[1]
        center_x = frame_width // 2
        delta_x = int(frame_width * range_percentage)
        x_min = center_x - delta_x
        x_max = center_x + delta_x
        for i in range(len(ids)):
            x_center = int(corners[i][0][:, 0].mean())
            if x_min <= x_center <= x_max and ids[i][0] in tag_list:
                return ids[i][0]
    return None

# File paths and parameters
file_path = 'TAGMAP_DHL-HD-Bangkok2-NEW.xlsx'
tab = 'VB ZONE'
target = 'VBF-R'

# Extract tags from the Excel file
tag_list = extract_tags(file_path, tab, target)['Tag'].tolist()

# Load the predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters()

# Path to the video file
video_file_path = '2024-05-27 - DHL-HD-Bangkok2_385_3857A_1_of_1.mp4'

# Path to the output folder for images
output_folder_path = 'extracted_images'
os.makedirs(output_folder_path, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_file_path}.")
else:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 24
    previous_tag = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ArUco tags in the center region of the frame
        detected_tag_id = detect_aruco_in_center(frame, aruco_dict, parameters, tag_list)
        if detected_tag_id:
            if detected_tag_id != previous_tag:
                # Save the frame as an image with tag ID in the filename
                image_filename = os.path.join(output_folder_path, f"frame_{frame_count:05d}_{detected_tag_id:03d}.png")
                print(f'save frame_{frame_count:05d}_{detected_tag_id:03d}.png')
                cv2.imwrite(image_filename, frame)
                frame_count += 1

            previous_tag = detected_tag_id

        # cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
print("Processing complete.")
