import cv2
import mediapipe as mp
from mtcnn import MTCNN
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from CVMouthReader.modules.utils.timer import timer

# METASYNTATIC VARIABLES
MEDIAPIPE_MIN_CONFIDENCE = 0.2
MTCNN_THRESH = 0.95
FRAME_DIFF = 10

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Initialize MediaPipe Face Mesh for mouth detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=MEDIAPIPE_MIN_CONFIDENCE)

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

@timer
def collect_data(frames_dir, output_csv, output_frames_dir, output_frame_interval=None):
    """
    Detects faces using MTCNN, extracts the face region, and uses MediaPipe Face Mesh
    to calculate mouth openness. Tracks the same person across frames and saves the data.
    Args:
        frames_dir (str): Directory containing extracted frames.
        output_csv (str): Path to save CSV output.
        output_frames_dir (str): Directory to save frames with drawn rectangles.
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

    # Initialize frame count for returning frames 
    frame_count = 0

    # Initialize scene tracking
    scene_number = 0
    prev_frame = None

    # Process each frame
    for frame_name in frame_files:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)

        # Convert frame to RGB (MTCNN and MediaPipe require RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect drastic changes between frames to update the scene
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, rgb_frame)
            diff_mean = np.mean(diff)
            if diff_mean > 30:  # Threshold for scene change (adjust as needed)
                scene_number += 1
                person_tracker.clear()  # Reset person tracking for the new scene

        prev_frame = rgb_frame

        # Detect faces using MTCNN
        mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)

        # Process each detected face
        for result in mtcnn_results:
            x, y, w, h = result['box']
            confidence = result['confidence']

            # Skip this face if confidence is below the threshold
            if confidence < MTCNN_THRESH:
                continue  # Skip to the next face

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

                # Check if this person is already being tracked within ±10 pixels
                found_person = False
                person_number = None
                for existing_person, data in person_tracker.items():
                    existing_x, existing_y, existing_w, existing_h = data["last_position"]
                    if (abs(existing_x - x) <= FRAME_DIFF and
                        abs(existing_y - y) <= FRAME_DIFF and
                        abs(existing_w - w) <= FRAME_DIFF and
                        abs(existing_h - h) <= FRAME_DIFF):
                        found_person = True
                        person_number = existing_person
                        break

                # If not found, assign a new person number
                if not found_person:
                    person_number = len(person_tracker) + 1  # Start from 1 for each scene
                    person_tracker[person_number] = {"last_position": (x, y, w, h), "frames": [], "mouth_openness": []}

                # Update the person's data
                person_tracker[person_number]["frames"].append(frame_name)
                person_tracker[person_number]["mouth_openness"].append(mouth_openness)
                person_tracker[person_number]["last_position"] = (x, y, w, h)

                # Draw rectangle around the face and display person number and talking status
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                talking_status = "Talking" if mouth_openness is not None and mouth_openness > 1.0 else "Not Talking"
                cv2.putText(frame, f"Person {person_number} {talking_status}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Store results in the list
                results.append([frame_name, scene_number, person_number, x, y, w, h, confidence, "mtcnn", mouth_openness])

        # Draw scene number in the top-left corner
        cv2.putText(frame, f"Scene {scene_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the frame with drawn rectangles (if output_frame_interval is None or the frame is selected)
        if output_frame_interval is not None and frame_count % output_frame_interval == 0:
            output_frame_path = os.path.join(output_frames_dir, frame_name)
            cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["Frame", "Scene", "Person", "X", "Y", "W", "H", "Confidence", "Detector", "Mouth Openness"])

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(frame_files)} frames. Results saved to {output_csv}")
    print(f"Frames with drawn rectangles saved to {output_frames_dir}")

    return df