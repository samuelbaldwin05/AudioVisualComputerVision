import cv2
import os
import shutil

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video and saves them to the output directory.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        frame_rate (int): Number of frames to extract per second.
    """
    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        # Remove all files in the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the subdirectory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file:", video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count // frame_interval} frames to {output_dir}")