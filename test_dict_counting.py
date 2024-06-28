import torch
import numpy as np
import cv2
from ultralytics import YOLO
from utils.cluster import cluster_boxes

def letterbox_image(image, target_size):
        """
        Resize image with unchanged aspect ratio using padding.
        """
        height, width, _ = image.shape
        scale = min(target_size / width, target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image with the computed scale
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create a new image with the target size and fill it with black (zeros)
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Compute top-left corner for the resized image to be centered
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2

        # Place the resized image in the padded image
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, :] = resized_image

        return padded_image

def box_counting(image_tensor, model_path='models/v13.pt', conf_threshold=0.3):
    # Load the YOLO model
    model = YOLO(model_path)

    # Perform inference on the image
    results = model(image_tensor)

    # Extract bounding boxes and convert to dictionary format for clustering
    boxes_by_class = {}

    for result in results:  # Iterate through the results
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract the bounding box coordinates
            conf = box.conf[0]  # Extract the confidence score
            cls = int(box.cls[0])  # Extract the class index
            label = model.names[cls]  # Get the class label name
            # Apply confidence threshold
            if conf >= conf_threshold:
                if cls not in boxes_by_class:
                    boxes_by_class[cls] = []
                boxes_by_class[cls].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls, 'conf': conf})

    # Cluster the boxes by class
    clusters_by_class = {}
    for cls, boxes in boxes_by_class.items():
        clusters_by_class[cls] = cluster_boxes(boxes)

    cbbox, bbox, conf, count = [], [], [], []
    for cls, clusters in clusters_by_class.items():
        for cluster in clusters:
            x1 = min(min(box['x1'], box['x2']) for box in cluster)
            y1 = min(min(box['y1'], box['y2']) for box in cluster)
            x2 = max(max(box['x1'], box['x2']) for box in cluster)
            y2 = max(max(box['y1'], box['y2']) for box in cluster)

            cbbox.append([x1, y1, x2, y2])
            for box in cluster:
                bbox.append([box['x1'], box['y1'], box['x2'], box['y2']])
                conf.append(float(box['conf']))
            count.append(len(cluster))


    return {
        "cbbox": cbbox,
        "count": count,
        "bbox": bbox,
        "conf": conf
    }

# Read the image
image = cv2.imread('inputs/ceva/4549_33.png')
# Convert the image to tensor and resize it
target_size = 1280
padded_image = letterbox_image(image, target_size)
image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).unsqueeze(0)
# Ensure the tensor is of type float and normalized
image_tensor = image_tensor.float() / 255.0
print(box_counting(image_tensor))

