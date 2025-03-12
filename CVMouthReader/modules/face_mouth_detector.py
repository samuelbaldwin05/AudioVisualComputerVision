import cv2
import mediapipe as mp
from mtcnn import MTCNN
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.utils.timer import timer
from tqdm import tqdm  # Import tqdm

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Face Mesh for mouth detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)

# Initialize MTCNN
mtcnn_detector = MTCNN()

def calculate_mouth_openness(landmarks, image_shape):
    """Calculates the distance between the upper and lower lip points."""
    ih, iw, _ = image_shape
    upper_lip_idx = 13
    lower_lip_idx = 14
    upper_lip = np.array([landmarks[upper_lip_idx].x * iw, landmarks[upper_lip_idx].y * ih])
    lower_lip = np.array([landmarks[lower_lip_idx].x * iw, landmarks[lower_lip_idx].y * ih])
    mouth_openness = np.linalg.norm(upper_lip - lower_lip)
    return mouth_openness

def process_frame(frame, frame_number, scene_number, person_tracker, FRAME_DIFF=10, MTCNN_THRESH=0.985):
    """Processes a video frame to detect faces, track individuals, and estimate mouth openness."""
    results = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)

    for result in mtcnn_results:
        x, y, w, h = result['box']
        confidence = result['confidence']
        if confidence < MTCNN_THRESH:
            continue

        if w > 0 and h > 0:
            face_region = rgb_frame[y:y+h, x:x+w]
            face_mesh_results = face_mesh.process(face_region)
            mouth_openness = None

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    mouth_openness = calculate_mouth_openness(face_landmarks.landmark, face_region.shape)

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

            if not found_person:
                person_number = len(person_tracker) + 1
                person_tracker[person_number] = {"last_position": (x, y, w, h), "frames": [], "mouth_openness": []}

            person_tracker[person_number]["frames"].append(frame_number)
            person_tracker[person_number]["mouth_openness"].append(mouth_openness)
            person_tracker[person_number]["last_position"] = (x, y, w, h)

            results.append([frame_number, scene_number, person_number, x, y, w, h, confidence, "mtcnn", mouth_openness])
    return results

@timer
def process_video(video_path, output_csv, extract_fps=1):
    """Processes a video frame-by-frame, tracks faces, and saves results to a CSV file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if extract_fps is None:
        save_every_n_frames = 1
        extract_fps = video_fps
    else:
        save_every_n_frames = max(1, int(video_fps / extract_fps))

    # Calculate the total number of frames to process
    total_frames_to_process = total_frames // save_every_n_frames

    scene_number = 0
    all_results = []
    person_tracker = {}
    prev_frame = None
    frame_index = 0

    # Use tqdm to show a progress bar
    with tqdm(total=total_frames_to_process, desc="Processing Video", unit="frame") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % save_every_n_frames == 0:
                # Check for scene changes
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_mean = np.mean(diff)
                    if diff_mean > 30:
                        scene_number += 1
                        person_tracker.clear()

                # Process current frame
                frame_results = process_frame(frame, frame_index, scene_number, person_tracker)
                all_results.extend(frame_results)
                prev_frame = frame

                # Update the progress bar
                pbar.update(1)

            frame_index += 1

    cap.release()

    # Convert and save results
    df = pd.DataFrame(all_results, columns=["Frame", "Scene", "Person", "X", "Y", "W", "H", 
                                         "Confidence", "Detector", "Mouth Openness"])
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(all_results)} mouth detections from {total_frames} total frames")
    print(f"Video FPS: {video_fps:.2f}, Extraction FPS: {extract_fps or video_fps:.2f}")
    print(f"Saved results to: {output_csv}")
