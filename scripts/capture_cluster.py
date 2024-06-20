import cv2
from ultralytics import YOLO
import os
import numpy as np

Class = ['f', 't']

    # Pastel colors for bounding boxes
colors = [
    (240, 228, 66),
    (230, 159, 0)  
]
def extract_corners(box):
    return [
        (box['x1'], box['y1']),
        (box['x1'], box['y2']),
        (box['x2'], box['y1']),
        (box['x2'], box['y2'])
    ]

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def is_point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    return x1 <= x <= x2 and y1 <= y <= y2

def is_any_point_inside(box1, box2):
    corners1 = extract_corners(box1)
    return any(is_point_inside_box(corner, box2) for corner in corners1)


def are_boxes_near_or_inside(box1, box2, threshold=10):
    corners1 = extract_corners(box1)
    corners2 = extract_corners(box2)
    near = any(distance(c1, c2) <= threshold for c1 in corners1 for c2 in corners2)
    inside = is_any_point_inside(box1, box2) or is_any_point_inside(box2, box1)
    return near or inside

def merge_clusters(clusters, threshold):
    merged = True
    while merged:
        merged = False
        new_clusters = []
        while clusters:
            cluster = clusters.pop(0)
            for i, other_cluster in enumerate(clusters):
                if any(are_boxes_near_or_inside(box, other_box, threshold) for box in cluster for other_box in other_cluster):
                    cluster.extend(other_cluster)
                    clusters.pop(i)
                    merged = True
                    break
            new_clusters.append(cluster)
        clusters = new_clusters
    return clusters

def cluster_boxes(boxes, threshold=10):
    clusters = []
    for box in boxes:
        added_to_cluster = False
        for cluster in clusters:
            if any(are_boxes_near_or_inside(box, other_box, threshold) for other_box in cluster):
                cluster.append(box)
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([box])

    clusters = merge_clusters(clusters, threshold)
    return clusters
def save_clustered_frame(image, cluster, output_dir, base_name, index, margin=25):
    x1 = min(min(box['x1'], box['x2']) for box in cluster) - margin
    y1 = min(min(box['y1'], box['y2']) for box in cluster) - margin
    x2 = max(max(box['x1'], box['x2']) for box in cluster) + margin
    y2 = max(max(box['y1'], box['y2']) for box in cluster) + margin

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])

    cluster_image = image[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"{base_name}_cluster_{index}.jpg")
    cv2.imwrite(output_path, cluster_image)

def process_image(model_path, image_path, output_dir,  conf_threshold, show_disp):
    # Load the YOLO model
    model = YOLO(model_path)

    # Read the image
    image = cv2.imread(image_path)
    


    # Perform inference on the image
    results = model(image)

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
                boxes_by_class[cls].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls, 'label': label, 'conf': conf})

    # Cluster the boxes by class
    clusters_by_class = {}
    for cls, boxes in boxes_by_class.items():
        clusters_by_class[cls] = cluster_boxes(boxes)

    # Draw bounding boxes and clusters on the image
    # for cls, boxes in boxes_by_class.items():
    #     for box in boxes:
    #         x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    #         color = colors[cls % len(colors)]
    #         label = f"{box['label']}: {box['conf']:.2f}"
    #         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    #         # Calculate the text size
    #         (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    #         # Draw the text background rectangle just below the top left corner inside the bounding box
    #         text_x = x1
    #         text_y = y1 + text_height + baseline
    #         cv2.rectangle(image, (text_x, y1), (text_x + text_width, text_y), color, -1)

    #         # Draw the text on top of the rectangle
    #         cv2.putText(image, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for cls, clusters in clusters_by_class.items():
        for index, cluster in enumerate(clusters):
            save_clustered_frame(image, cluster, output_dir, os.path.splitext(os.path.basename(image_path))[0], index)
        for cluster in clusters:
            x1 = min(min(box['x1'], box['x2']) for box in cluster)
            y1 = min(min(box['y1'], box['y2']) for box in cluster)
            x2 = max(max(box['x1'], box['x2']) for box in cluster)
            y2 = max(max(box['y1'], box['y2']) for box in cluster)
            color = colors[cls % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
            cluster_label = f"{Class[cls]}:{len(cluster)}"

            # Calculate the text size
            (text_width, text_height), baseline = cv2.getTextSize(cluster_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Draw the text background rectangle
            cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)

            # Draw the text on top of the rectangle
            cv2.putText(image, cluster_label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save the processed image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    # cv2.imwrite(output_path, image)

    # Display the image if show_disp is True
    if show_disp:
        cv2.imshow('YOLOv8 Detection', cv2.resize(image, (1080, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model_path, video_path, skip_frames, output_dir,  conf_threshold, show_disp):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    

    i = 0
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        

        # Perform inference on the frame
        results = model(frame)

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
                    boxes_by_class[cls].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls, 'label': label, 'conf': conf})

        # Cluster the boxes by class
        clusters_by_class = {}
        for cls, boxes in boxes_by_class.items():
            clusters_by_class[cls] = cluster_boxes(boxes)

        # Draw bounding boxes and clusters on the frame
        # for cls, boxes in boxes_by_class.items():
            
        #     for box in boxes:
        #         x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        #         color = colors[cls % len(colors)]
        #         label = f"{box['label']}: {box['conf']:.2f}"
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        #         # Calculate the text size
        #         (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        #         # Draw the text background rectangle just below the top left corner inside the bounding box
        #         text_x = x1
        #         text_y = y1 + text_height + baseline
        #         cv2.rectangle(frame, (text_x, y1), (text_x + text_width, text_y), color, -1)

        #         # Draw the text on top of the rectangle
        #         cv2.putText(frame, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for cls, clusters in clusters_by_class.items():
            for index, cluster in enumerate(clusters):
                i += 1
                save_clustered_frame(frame, cluster, output_dir, os.path.splitext(os.path.basename(output_dir))[0] + str(i), index)
            for cluster in clusters:
                x1 = min(min(box['x1'], box['x2']) for box in cluster)
                y1 = min(min(box['y1'], box['y2']) for box in cluster)
                x2 = max(max(box['x1'], box['x2']) for box in cluster)
                y2 = max(max(box['y1'], box['y2']) for box in cluster)
                color = colors[cls % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cluster_label = f"{Class[cls]}:{len(cluster)}"

                # Calculate the text size
                (text_width, text_height), baseline = cv2.getTextSize(cluster_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Determine if there is enough space above the box for the label
                label_y1 = y1 - text_height - baseline
                label_y2 = y1

                if label_y1 < 0:  # Not enough space above, plot below the box
                    label_y1 = y2 + baseline
                    label_y2 = y2 + text_height + baseline

                # Draw the text background rectangle
                cv2.rectangle(frame, (x1, label_y1), (x1 + text_width, label_y2), color, -1)

                # Draw the text on top of the rectangle
                cv2.putText(frame, cluster_label, (x1, label_y2 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame if show_disp is True
        if show_disp:
            cv2.imshow('YOLOv8 Detection', cv2.resize(frame, (1080, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows if show_disp is True
    if show_disp:
        cv2.destroyAllWindows()

def main(model_path, source_path,output_dir ,skip_frames,conf, show_disp):

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(source_path):
        # Process each file in the folder
        for filename in os.listdir(source_path):
            print(filename)
            if filename.endswith((".mp4",".MP4")):
                
                video_path = os.path.join(source_path, filename)
                process_video(model_path, video_path, skip_frames, output_dir, conf, show_disp)
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(source_path, filename)
                process_image(model_path, image_path, output_dir, conf, show_disp)
    else:
        # Process a single file
        if source_path.endswith((".mp4",".MP4")):
            process_video(model_path, source_path, skip_frames, output_dir, conf, show_disp)
        elif source_path.endswith(('.jpg', '.jpeg', '.png')):
            process_image(model_path, source_path, output_dir, conf, show_disp)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 image and video processing with frame skipping and folder support.")
    parser.add_argument("--model", help="Path to the YOLOv8 model file.", default='models/19junv12.pt')
    parser.add_argument("--source", help="Path to the video or image file or folder containing files.", default='inputs/DJI_0711_processed.MP4')
    parser.add_argument("--output", help="Path for output.", default='outputs/capO')
    parser.add_argument("--skip", type=int, default=1, help="Number of frames to skip.")
    parser.add_argument("--confidence", type=float, default=0.3, help="confident level.")
    parser.add_argument("--show", type=bool, default=False, help="Display the processed images and videos.")
    
    args = parser.parse_args()
    
    main(args.model, args.source,args.output, args.skip, args.confidence, args.show)
