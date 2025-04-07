import cv2
import pandas as pd
from utils.timer import timer
from tqdm import tqdm

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


# video_path = 'CVMouthReader/data/input/smithscut.mp4'
# csv_path = 'CVMouthReader/data/output/final.csv'
# frame_number = 150

# # Get the processed frame
# processed_frame = draw_boxes_on_frame(video_path, csv_path, frame_number)


video_path = 'CVMouthReader/data/input/smithscut.mp4'
csv_path = 'CVMouthReader/data/output/final2.csv'
output_video_path = 'CVMouthReader/data/output/processed_output.mp4'

draw_boxes_and_save_video(video_path, csv_path, output_video_path)
