import cv2
import os

# Function to save the current frame
def save_frame(frame, frame_number, save_folder, video_filename):
    base_filename = os.path.splitext(video_filename)[0]
    filename = os.path.join(save_folder, f'{base_filename}_frame_{frame_number}.png')
    cv2.imwrite(filename, frame)
    print(f'Saved {filename}')

# Path to the folder containing videos
video_folder = 'lf'
save_folder = 'saved_frames'

# Create save folder if it does not exist
os.makedirs(save_folder, exist_ok=True)

# Check if video folder exists
if not os.path.exists(video_folder):
    print(f"Error: Video folder '{video_folder}' does not exist.")
else:
    # List all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    print(f"Found {len(video_files)} video files.")

    if len(video_files) == 0:
        print("No video files found in the specified folder.")

    for video_filename in video_files:
        # Load the video
        video_path = os.path.join(video_folder, video_filename)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue

        print(f"Processing video file {video_path}")

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {video_filename}: {total_frames}")

        # Create a window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 1280, 720)  # Set window to HD size

        # Play the video
        current_pos = 0
        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print(f"Finished reading {video_filename}")
                    break

                # Resize the frame to HD size
                frame_hd = cv2.resize(frame, (1280, 720))

                # Display the current frame
                cv2.imshow('Video', frame_hd)

                # Increment the current position for normal playback
                current_pos += 1
                if current_pos >= total_frames:
                    current_pos = 0  # Loop the video

            # Wait for a short period and get key press
            key = cv2.waitKey(30)  # Wait for 30ms

            if key == 27:  # Esc key to stop
                break
            elif key == ord('a'):  # 'a' key for back 30 frames
                current_pos = max(current_pos - 30, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                paused = False
            elif key == ord('d'):  # 'd' key for next 30 frames
                current_pos = min(current_pos + 30, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                paused = False
            elif key == ord('s'):  # 's' key to save the current frame
                save_frame(frame, current_pos, save_folder, video_filename)
            elif key == 32:  # Space bar to pause/resume
                paused = not paused

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
