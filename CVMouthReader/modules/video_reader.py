import cv2
import os
import shutil
from CVMouthReader.modules.utils.timer import timer

@timer
def extract_frames(video_path, output_dir, extract_fps=1):
    """
    Extracts frames from a video and saves them to the output directory.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        extract_fps (int/None): Frames per second to extract (None = all frames)
    """
    # Clear or create output directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval based on extraction FPS
    if extract_fps is None:
        save_every_n_frames = 1  # Save every frame
        extract_fps = video_fps
    else:
        save_every_n_frames = max(1, int(video_fps / extract_fps))

    frame_index = 0
    saved_frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save frame if it matches our interval
        if frame_index % save_every_n_frames == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_index += 1

    cap.release()
    
    print(f"Extracted {saved_frame_count} frames from {total_frames} total frames")
    print(f"Video FPS: {video_fps:.2f}, Extraction FPS: {extract_fps or video_fps:.2f}")
    print(f"Saved frames to: {output_dir}")