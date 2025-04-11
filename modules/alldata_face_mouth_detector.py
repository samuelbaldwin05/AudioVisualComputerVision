import cv2
import mediapipe as mp
from mtcnn import MTCNN
import os
import numpy as np
import pandas as pd
import multiprocessing as mp_proc
from functools import partial
from modules.utils.timer import timer

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Face Mesh for mouth detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)

# Initialize MTCNN for face detection
mtcnn_detector = MTCNN()

def calculate_mouth_openness(landmarks, image_shape):
    ih, iw, _ = image_shape
    upper_lip_idx, lower_lip_idx = 13, 14
    upper_lip = np.array([landmarks[upper_lip_idx].x * iw, landmarks[upper_lip_idx].y * ih])
    lower_lip = np.array([landmarks[lower_lip_idx].x * iw, landmarks[lower_lip_idx].y * ih])
    return np.linalg.norm(upper_lip - lower_lip)

def process_frame(frame, frame_number, scene_number):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)
    results = []
    for result in mtcnn_results:
        x, y, w, h = result['box']
        confidence = result['confidence']

        if w > 0 and h > 0:
            face_region = rgb_frame[y:y+h, x:x+w]
            face_mesh_results = face_mesh.process(face_region)

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    mouth_openness = calculate_mouth_openness(face_landmarks.landmark, face_region.shape)
                    flattened_landmarks = [coord for lm in face_landmarks.landmark for coord in (lm.x * w + x, lm.y * h + y, lm.z * w)]
                    if len(flattened_landmarks) == 1404:
                        results.append([frame_number, scene_number, x, y, w, h, confidence, mouth_openness] + flattened_landmarks)
    return results

def process_chunk(video_path, chunk_range, extract_fps=1, progress=None, progress_lock=None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_range[0])
    chunk_results = []

    for frame_number in range(chunk_range[0], chunk_range[1]):
        success, frame = cap.read()
        if not success:
            break
        if frame_number % extract_fps == 0:
            chunk_results.extend(process_frame(frame, frame_number, 0))
            if progress is not None:
                with progress_lock:
                    progress.value += 1

    cap.release()
    return chunk_results

@timer
def process_video_parallel(video_path, output_csv, num_processes=4, extract_fps=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_size = total_frames // num_processes
    chunk_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]

    with mp_proc.Manager() as manager:
        progress = manager.Value('i', 0)
        progress_lock = manager.Lock()

        with mp_proc.Pool(num_processes) as pool:
            results = pool.map(partial(process_chunk, video_path, extract_fps=extract_fps, progress=progress, progress_lock=progress_lock), chunk_ranges)

    merged_results = [item for sublist in results for item in sublist]
    landmark_columns = [f"Lm_{i}_{axis}" for i in range(468) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(merged_results, columns=["Frame", "Scene", "X", "Y", "W", "H", "Confidence", "Mouth Openness"] + landmark_columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")
