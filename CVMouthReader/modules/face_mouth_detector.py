import cv2
import mediapipe as mp
import multiprocessing as mproc
from mtcnn import MTCNN
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.utils.timer import timer

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Face Mesh for mouth detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)

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

def process_frame(frame, frame_number, scene_number, person_tracker, FRAME_DIFF=10, MTCNN_THRESH=0.985):
    """
    Processes a single frame to detect faces, track people, and calculate mouth openness.
    Args:
        frame (np.array): The frame to process.
        frame_number (int): The frame number.
        scene_number (int): The current scene number.
        person_tracker (dict): Dictionary to track people across frames.
        FRAME_DIFF (int): Threshold for tracking people across frames.
        MTCNN_THRESH (float): Confidence threshold for MTCNN face detection.
    Returns:
        list: Results for the frame.
    """
    results = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)

    # Process each detected face
    for result in mtcnn_results:
        x, y, w, h = result['box']
        confidence = result['confidence']

        # Skip this face if confidence is below the threshold
        if confidence < MTCNN_THRESH:
            continue

        # Ensure the face region is valid
        if w > 0 and h > 0:
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
            person_tracker[person_number]["frames"].append(frame_number)
            person_tracker[person_number]["mouth_openness"].append(mouth_openness)
            person_tracker[person_number]["last_position"] = (x, y, w, h)

            # Store results in the list
            results.append([frame_number, scene_number, person_number, x, y, w, h, confidence, "mtcnn", mouth_openness])

    return results

def process_frame_batch(frame_batch, scene_number, person_tracker):
    """
    Processes a batch of frames.
    Args:
        frame_batch (list): List of tuples containing (frame_number, frame).
        scene_number (int): The current scene number.
        person_tracker (dict): Dictionary to track people across frames.
    Returns:
        list: Results for the batch.
    """
    results = []
    prev_frame = None

    for frame_number, frame in frame_batch:
        # Detect drastic changes between frames to update the scene
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            diff_mean = np.mean(diff)
            if diff_mean > 30:  # Threshold for scene change
                scene_number += 1
                person_tracker.clear()  # Reset person tracking for the new scene

        prev_frame = frame
        frame_results = process_frame(frame, frame_number, scene_number, person_tracker)
        results.extend(frame_results)

    return results

@timer
def process_video(video_path, output_csv, extract_fps=1, num_processes=5):
    """
    Processes a video frame-by-frame using multiprocessing.
    Args:
        video_path (str): Path to the input video file.
        output_csv (str): Path to save CSV output.
        extract_fps (int): Frames per second to extract.
        num_processes (int): Number of processes to use for multiprocessing.
    """
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

    # Initialize scene tracking
    scene_number = 0

    # Initialize results list
    all_results = []

    # Use a multiprocessing Manager to create a shared dictionary for person_tracker
    manager = mproc.Manager()
    person_tracker = manager.dict()

    # Multiprocessing setup
    pool = mproc.Pool(processes=num_processes)
    frame_batches = []
    current_batch = []
    batch_size = 10  # Number of frames per batch

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % save_every_n_frames == 0:
            current_batch.append((frame_index, frame))
            if len(current_batch) >= batch_size:
                frame_batches.append(current_batch)
                current_batch = []

        frame_index += 1

    if current_batch:
        frame_batches.append(current_batch)

    # Process frames in parallel
    for results in pool.starmap(process_frame_batch, [(batch, scene_number, person_tracker) for batch in frame_batches]):
        all_results.extend(results)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results, columns=["Frame", "Scene", "Person", "X", "Y", "W", "H", "Confidence", "Detector", "Mouth Openness"])

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(all_results)} mouth detections from {total_frames} total frames")
    print(f"Video FPS: {video_fps:.2f}, Extraction FPS: {extract_fps or video_fps:.2f}")
    print(f"Saved results to: {output_csv}")

    cap.release()
    pool.close()
    pool.join()