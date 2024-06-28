import torch
from torchvision import transforms
import numpy as np
import cv2
from ultralytics import YOLO
from utils.cluster import cluster_boxes

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

def cv2_to_torch_image(image, target_size=1280, stride=32):

    
    # Step 2: Resize the image while maintaining the aspect ratio
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(target_size * width / height)
    else:
        new_width = target_size
        new_height = int(target_size * height / width)
    
    # Ensure new dimensions are divisible by the stride
    new_height = (new_height // stride) * stride
    new_width = (new_width // stride) * stride
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Step 3: Convert color space from BGR to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Step 4: Normalize pixel values to [0, 1]
    normalized_image = rgb_image / 255.0
    
    # Step 5: Convert the image to a PyTorch tensor and add batch dimension
    tensor_image = torch.tensor(normalized_image).permute(2, 0, 1).float().unsqueeze(0)
    
    return tensor_image
# Read the image
image = cv2.imread('inputs/ceva/4549_33.png')
image_tensor = cv2_to_torch_image(image)
result_dict = box_counting(image_tensor)
print(result_dict)


def draw_boxes(image, result_dict):
    for box in result_dict["cbbox"]:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    for box in result_dict["bbox"]:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    return image


# Draw bounding boxes on the image
output_image = draw_boxes(image, result_dict)

# Display the output image with bounding boxes
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the output image
cv2.imwrite('output_with_boxes.png', output_image)