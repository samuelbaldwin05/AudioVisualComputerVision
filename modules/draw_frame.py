import cv2
import pandas as pd
import numpy as np
from modules.utils.timer import timer
from tqdm import tqdm
import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

@timer
def draw_boxes_on_frame(video_path, csv_path, frame_number):
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter rows for the given frame number
    frame_data = df[df['Frame'] == frame_number]
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Set the video to the specific frame
    current_frame = 0
    while current_frame < frame_number:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_number}.")
            return None
        current_frame += 1
    
    # Copy
    frame = frame.copy()

    # Get the scene number
    scene_number = frame_data['Scene'].iloc[0]
    
    # Draw the scene number on the top-left corner
    cv2.putText(frame, f"Scene: {scene_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Loop through each row in the filtered data
    for index, row in frame_data.iterrows():
        x, y, w, h = int(row['X']), int(row['Y']), int(row['W']), int(row['H'])
        mouth_openness = row['Mouth Openness']
        person_num = row['Person']
        
        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Annotate the mouth openness above the bounding box
        cv2.putText(frame, f"Person {person_num}: {mouth_openness:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Release the video capture object
    cap.release()
    
    # Return the processed frame
    if frame is not None:
    # Save or display the frame
        cv2.imwrite('CVMouthReader/data/output/output_frame2.jpg', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process the frame.")

@timer
def draw_boxes_and_save_video(video_path, csv_path, output_video_path):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process frames with progress bar
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw bounding boxes for the current frame
            frame_data = df[df['Frame'] == frame_index]
            for _, row in frame_data.iterrows():
                x, y, w, h = int(row['X']), int(row['Y']), int(row['W']), int(row['H'])
                mouth_openness = row['Mouth Openness']
                person_num = row['Person']

                # Draw boxes and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {person_num}: {mouth_openness:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Write processed frame to video
            out.write(frame)
            frame_index += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()

    print(f"Video saved to: {output_video_path}")



def add_subtitles_with_scene_detection(video_path, subtitle_csv, output_path, 
                                      scene_threshold=20, font_scale=1.0, 
                                      text_color=(255, 255, 255), bg_color=(0, 0, 0, 128)):
    """
    Adds subtitles to a video and detects scene changes, displaying scene numbers.
    Preserves the original audio.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    subtitle_csv : str
        Path to CSV file containing subtitles with start_time and end_time columns
    output_path : str
        Path where the output video will be saved
    scene_threshold : float, optional
        Threshold for scene change detection (higher = less sensitive)
    font_scale : float, optional
        Size of the subtitle text (default: 1.0)
    text_color : tuple, optional
        RGB color of the subtitle text (default: white)
    bg_color : tuple, optional
        RGBA color of the background behind text (default: semi-transparent black)
        
    Returns:
    --------
    str
        Path to the output video file with subtitles, scene numbers, and preserved audio
    """
    
    # Create a temporary file for the video without audio
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Load subtitle data
    subtitles_df = pd.read_csv(subtitle_csv)
    
    # Clean column names by stripping whitespace
    subtitles_df.columns = [col.strip() for col in subtitles_df.columns]
    
    # Find the correct column names
    time_cols = ['start_time', 'end_time']
    for col in time_cols:
        if col not in subtitles_df.columns:
            # Try case-insensitive match
            matching_cols = [c for c in subtitles_df.columns if c.lower() == col.lower()]
            if matching_cols:
                subtitles_df = subtitles_df.rename(columns={matching_cols[0]: col})
    
    # Find the subtitle/text column
    text_col = None
    for possible_name in ['subtitle', 'text', 'word', 'caption']:
        if possible_name in subtitles_df.columns:
            text_col = possible_name
            break
    
    if text_col is None:
        raise ValueError("Could not find subtitle column in CSV")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object for temporary video without audio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # Prepare subtitle lookup dictionary: frame_number -> subtitle
    subtitle_frames = {}
    for _, row in subtitles_df.iterrows():
        start_frame = int(row['start_time'] * fps)
        end_frame = int(row['end_time'] * fps)
        text = str(row[text_col])
        
        for frame_num in range(start_frame, end_frame + 1):
            subtitle_frames[frame_num] = text
    
    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    padding = 10  # Padding around text
    
    # Scene detection variables
    prev_frame = None
    scene_count = 1
    scene_frames = {}  # Frame number -> scene number
    
    # Process video frame by frame
    frame_number = 0
    
    # Use tqdm for progress tracking
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for scene detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Scene detection
        if prev_frame is not None:
            # Calculate difference between current and previous frame
            frame_diff = cv2.absdiff(gray_frame, prev_frame)
            mean_diff = np.mean(frame_diff)
            
            # If difference is above threshold, it's a new scene
            if mean_diff > scene_threshold:
                scene_count += 1
                print(f"Scene change detected at frame {frame_number} (diff: {mean_diff:.2f})")
        
        # Store scene number for current frame
        scene_frames[frame_number] = scene_count
        
        # Update previous frame for next iteration
        prev_frame = gray_frame.copy()  # Make sure to create a copy
        
        # Check if this frame should have a subtitle
        if frame_number in subtitle_frames:
            text = subtitle_frames[frame_number]
            
            # Get text size to calculate background size
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            
            # Position for subtitle text (top-left corner with padding)
            text_x = padding
            text_y = padding + text_h
            
            # Create overlay for semi-transparent background
            overlay = frame.copy()
            
            # Draw the background rectangle for subtitle
            cv2.rectangle(
                overlay, 
                (text_x - padding, text_y - text_h - padding), 
                (text_x + text_w + padding, text_y + padding),
                bg_color[:3],  # RGB portion of the color
                -1  # Filled rectangle
            )
            
            # Apply transparency to the background
            alpha = bg_color[3] / 255.0
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw the subtitle text
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
        
        # Add scene number text (top-right corner)
        scene_text = f"Scene: {scene_frames[frame_number]}"
        scene_text_size, _ = cv2.getTextSize(scene_text, font, font_scale, font_thickness)
        scene_text_w, scene_text_h = scene_text_size
        
        # Position for scene text (top-right corner with padding)
        scene_x = width - scene_text_w - padding
        scene_y = padding + scene_text_h
        
        # Create overlay for scene number background
        overlay = frame.copy()
        
        # Draw the background rectangle for scene number
        cv2.rectangle(
            overlay, 
            (scene_x - padding, scene_y - scene_text_h - padding), 
            (scene_x + scene_text_w + padding, scene_y + padding),
            bg_color[:3],  # RGB portion of the color
            -1  # Filled rectangle
        )
        
        # Apply transparency to the background
        alpha = bg_color[3] / 255.0
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the scene number text
        cv2.putText(
            frame,
            scene_text,
            (scene_x, scene_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        
        # Write the frame to output video
        out.write(frame)
        
        # Update progress bar and frame count
        frame_number += 1
        pbar.update(1)
    
    # Clean up OpenCV resources
    pbar.close()
    cap.release()
    out.release()
    
    print(f"Processed {frame_number} frames with {scene_count} scene changes")
    print("Adding audio to the video...")
    
    # Use moviepy to merge the video with the original audio
    try:
        # Load the original video to extract its audio
        original_video = VideoFileClip(video_path)
        # Load the newly created video without audio
        new_video = VideoFileClip(temp_video)
        
        # Add the original audio to the new video
        final_video = new_video.set_audio(original_video.audio)
        
        # Write the final video with audio
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Close the clips to release resources
        original_video.close()
        new_video.close()
        final_video.close()
        
        # Remove the temporary file
        os.remove(temp_video)
        
        print(f"Subtitled video with scene numbers and audio saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error adding audio: {e}")
        print(f"Video without audio saved to: {temp_video}")
        # If there's an error with the audio, just rename the temp file
        if os.path.exists(temp_video):
            os.rename(temp_video, output_path)
            return output_path