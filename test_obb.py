import cv2
import numpy as np
from ultralytics import YOLO

def draw_oriented_bbox(image, bbox, label, color):
    cx, cy, w, h, angle = bbox
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    box_points = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    rotated_box_points = cv2.transform(np.array([box_points]), rotation_matrix)[0]
    rotated_box_points[:, 0] += cx
    rotated_box_points[:, 1] += cy
    rotated_box_points = np.int32(rotated_box_points)
    cv2.polylines(image, [rotated_box_points], isClosed=True, color=color, thickness=2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = int(cx - text_width / 2)
    text_y = int(cy - h / 2 - baseline - 5)
    text_bg_x2 = text_x + text_width
    text_bg_y2 = text_y + text_height + baseline
    cv2.rectangle(image, (text_x, text_y), (text_bg_x2, text_bg_y2), color, -1)
    cv2.putText(image, label, (text_x, text_y + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Load the YOLOv8 OBB model
model = YOLO('models/yolov8obb.pt')

# Perform inference
image_path = 'inputs/ceva/4549_1.png'
image = cv2.imread(image_path)
results = model(image_path)

# Define colors for classes
colors = [
    (240, 228, 66),
    (230, 159, 0)  
]


for result in results:  # Iterate through the results
    for box in result.boxes:
        print(box)
        cls = box.cls
        conf = box.conf
        xywhtheta = box.xywhn if hasattr(box, 'xywhn') else box.xywh
        x, y, w, h, theta = xywhtheta  # Assuming these attributes are provided by the model
        color = colors[int(cls) % len(colors)]
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        draw_oriented_bbox(image, (x, y, w, h, theta), label, color)

# Display the image
cv2.imshow('Image withOriented Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
