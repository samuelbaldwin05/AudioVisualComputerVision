import cv2
import mediapipe as mp
from mtcnn import MTCNN
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Initialize MediaPipe Face Mesh for mouth detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)

# Initialize MTCNN
mtcnn_detector = MTCNN()

def calculate_mouth_openness(landmarks, image_shape):
    """
    Calculates the mouth openness ratio based on upper and lower lip landmarks.
    Args:
        landmarks (list): List of face mesh landmarks.
        image_shape (tuple): Shape of the image for coordinate scaling.
    Returns:
        float: Mouth openness ratio.
    """
    ih, iw, _ = image_shape

    # Define key mouth landmarks (MediaPipe Face Mesh indices)
    upper_lip_idx = 13  # Center of upper lip
    lower_lip_idx = 14  # Center of lower lip

    # Get coordinates
    upper_lip = np.array([landmarks[upper_lip_idx].x * iw, landmarks[upper_lip_idx].y * ih])
    lower_lip = np.array([landmarks[lower_lip_idx].x * iw, landmarks[lower_lip_idx].y * ih])

    # Calculate mouth openness (Euclidean distance between upper and lower lip)
    mouth_openness = np.linalg.norm(upper_lip - lower_lip)

    return mouth_openness

def detect_faces_and_mouths(frames_dir, output_csv, output_frames_dir, output_frame_interval=None):
    """
    Detects faces using MTCNN, extracts the face region, and uses MediaPipe Face Mesh
    to calculate mouth openness. Tracks the same person across frames and determines
    if they are talking based on mouth openness variations.
    Args:
        frames_dir (str): Directory containing extracted frames.
        output_csv (str): Path to save CSV output.
        output_frames_dir (str): Directory to save frames with drawn rectangles.
        fps (int): Frames per second of the video.
        output_frame_interval (int, optional): If provided, only save every Nth frame to the output directory.
    """
    # Initialize list to store results
    results = []

    # Create output directory if it doesn't exist
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    # Dictionary to track people across frames
    person_tracker = defaultdict(lambda: {"frames": [], "mouth_openness": []})

    # Get list of frames and sort them
    frame_files = sorted(os.listdir(frames_dir))
    frame_count = 0

    # Process each frame
    for frame_name in frame_files:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)

        # Convert frame to RGB (MTCNN and MediaPipe require RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)

        # Process each detected face
        for result in mtcnn_results:
            x, y, w, h = result['box']
            confidence = result['confidence']

            # Ensure the face region is valid
            if w > 0 and h > 0:  # Skip invalid face regions
                # Extract the face region
                face_region = rgb_frame[y:y+h, x:x+w]

                # Use MediaPipe Face Mesh to detect landmarks in the face region
                face_mesh_results = face_mesh.process(face_region)

                mouth_openness = None
                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        # Calculate mouth openness
                        mouth_openness = calculate_mouth_openness(face_landmarks.landmark, face_region.shape)

                # Assign a unique ID to the person based on their bounding box position
                person_id = f"person_{x}_{y}"

                # Update the person's data
                person_tracker[person_id]["frames"].append(frame_name)
                person_tracker[person_id]["mouth_openness"].append(mouth_openness)

                # Draw rectangle around the face and display mouth openness
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{person_id} {mouth_openness if mouth_openness is not None else 'N/A'}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Store results in the list
                results.append([frame_name, person_id, x, y, w, h, confidence, "mtcnn", mouth_openness])

        # Save the frame with drawn rectangles (if output_frame_interval is None or the frame is selected)
        if output_frame_interval is None or frame_count % output_frame_interval == 0:
            output_frame_path = os.path.join(output_frames_dir, frame_name)
            cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["Frame", "Person ID", "X", "Y", "W", "H", "Confidence", "Detector", "Mouth Openness"])

    # Add a "Talking" column based on mouth openness variations
    df["Talking"] = False

    # Analyze mouth openness over time for each person
    for person_id, data in person_tracker.items():
        mouth_openness = data["mouth_openness"]
        # Filter out None values
        valid_mouth_openness = [mo for mo in mouth_openness if mo is not None]
        if len(valid_mouth_openness) > 1:  # Need at least 2 frames to detect variation
            # Calculate the standard deviation of mouth openness
            std_dev = np.std(valid_mouth_openness)
            # If the standard deviation is above a threshold, the person is talking
            if std_dev > 1.0:  # Adjust this threshold as needed
                df.loc[df["Person ID"] == person_id, "Talking"] = True

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(frame_files)} frames. Results saved to {output_csv}")
    print(f"Frames with drawn rectangles saved to {output_frames_dir}")
