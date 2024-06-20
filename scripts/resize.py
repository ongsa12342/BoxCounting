import cv2
import os
import sys
from tqdm import tqdm
import argparse

def resize_image(image_path, output_path, new_width=1280):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False

    original_height, original_width = image.shape[:2]
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, resized_image)
    print(f"Successfully resized image: {image_path}")
    return True

def resize_video(video_path, output_path, new_width=1280):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = new_width / frame_width
    new_height = int(frame_height * aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc=f"Processing video: {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"Successfully resized video: {video_path}")
    return True

def process_folder(input_folder, output_folder, is_video=False, new_width=1280):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if is_video:
            resize_video(input_path, output_path, new_width)
        else:
            resize_image(input_path, output_path, new_width)

def main(input_path, output_path=None, new_width=1280):
    if not output_path:
        if os.path.isfile(input_path):
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_processed{ext}"
        elif os.path.isdir(input_path):
            output_path = f"{input_path}_processed"

    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            resize_video(input_path, output_path, new_width)
        else:
            resize_image(input_path, output_path, new_width)
    elif os.path.isdir(input_path):
        is_video_folder = any(fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for fname in os.listdir(input_path))
        process_folder(input_path, output_path, is_video=is_video_folder, new_width=new_width)
    else:
        print("Invalid input path. Please provide a valid file or folder path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images and videos.')
    parser.add_argument('input', type=str, help='Path to the input file or folder.')
    parser.add_argument('--output', type=str, help='Path to the output file or folder.', default=None)
    parser.add_argument('--width', type=int, default=1280, help='New width for resizing. Default is 1280 pixels.')

    args = parser.parse_args()

    main(args.input, args.output, args.width)
