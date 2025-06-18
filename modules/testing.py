import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import torch
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue, cpu_count, Value, Lock
from tqdm import tqdm
import time
import logging
import warnings

# === Configure logging ===
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings even more aggressively

# === Suppress ONNX Runtime warnings ===
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 3 = WARNING, 4 = ERROR

# === Environment setup ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable ONEDNN optimizations for better compatibility

# Initialize MediaPipe face mesh module at module level
mp_face_mesh = mp.solutions.face_mesh

# Global detector instance for sharing across processes
_global_detector = None

def get_face_detector():
    """Get or initialize the RetinaFace detector as a singleton."""
    global _global_detector
    if _global_detector is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing RetinaFace on {device}...")
        _global_detector = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'],
            allowed_modules=['detection']  # Only load detection module to reduce memory usage
        )
        # Increased detection size for better A100 utilization
        _global_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(960, 960))
    return _global_detector

def calculate_mouth_openness(landmarks, image_shape):
    """Calculate vertical distance between upper and lower lip landmarks."""
    ih, iw, _ = image_shape
    upper_lip_idx = 13
    lower_lip_idx = 14
    if len(landmarks) > max(upper_lip_idx, lower_lip_idx):
        upper_lip = np.array([landmarks[upper_lip_idx].x * iw, landmarks[upper_lip_idx].y * ih])
        lower_lip = np.array([landmarks[lower_lip_idx].x * iw, landmarks[lower_lip_idx].y * ih])
        return np.linalg.norm(upper_lip - lower_lip)
    return None

def process_frame(frame, frame_number, scene_number, face_mesh, face_detector):
    """Detect face, extract landmarks, and calculate mouth openness."""
    results = []
    try:
        # Skip small or corrupt frames
        if frame.size == 0 or frame.shape[0] < 20 or frame.shape[1] < 20:
            logging.warning(f"Frame {frame_number} is too small or corrupt, skipping")
            return results
            
        # Resize large frames to improve performance
        max_dim = 1280
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        
        # Use try/except for face detection to handle potential errors
        try:
            faces = face_detector.get(rgb_frame)  # Detect faces using RetinaFace
        except Exception as e:
            logging.warning(f"Face detection failed on frame {frame_number}: {e}")
            return results
            
        # Process ALL detected faces (not just the highest confidence one)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)  # Get face bounding box
            
            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rgb_frame.shape[1], x2), min(rgb_frame.shape[0], y2)
            
            # Skip invalid boxes that are too small
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
            confidence = face.det_score  # Get detection confidence
                
            face_region = rgb_frame[y1:y2, x1:x2]  # Crop face region

            # Get facial landmarks using MediaPipe
            try:
                mesh_results = face_mesh.process(face_region)
                mouth_openness = None
                
                if mesh_results and mesh_results.multi_face_landmarks:
                    face_landmarks = mesh_results.multi_face_landmarks[0]
                    mouth_openness = calculate_mouth_openness(face_landmarks.landmark, face_region.shape)

                    # Flatten landmark coordinates
                    flattened_landmarks = []
                    for lm in face_landmarks.landmark:
                        px, py, pz = lm.x * (x2 - x1) + x1, lm.y * (y2 - y1) + y1, lm.z * (x2 - x1)
                        flattened_landmarks.extend([px, py, pz])

                    # Store all extracted features
                    results.append([frame_number, scene_number, x1, y1, x2 - x1, y2 - y1, confidence, mouth_openness] + flattened_landmarks)
            except Exception as e:
                logging.warning(f"Landmark detection failed on face in frame {frame_number}: {e}")
                
    except Exception as e:
        logging.error(f"Error processing frame {frame_number}: {e}")
    
    return results

def process_worker(worker_id, frame_queue, output_queue, total_processed, face_mesh_lock, verbose=False):
    """Worker that processes frames and puts the results in output queue."""
    # Suppress messages if not verbose
    if not verbose:
        logging.getLogger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Initialize MediaPipe face mesh (with low resource usage)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,  # Disable for better performance
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    processed_frames = 0
    
    while True:
        try:
            item = frame_queue.get(timeout=5)  # Get next frame with timeout
        except:
            # If queue is empty for too long, assume end of processing
            break
            
        if item is None:
            break  # End of stream
            
        frame_number, frame = item
        
        # Get singleton detector instance
        with face_mesh_lock:
            # Process frame with shared detector instance
            results = process_frame(frame, frame_number, 0, face_mesh, get_face_detector())
        
        output_queue.put(results)  # Send results back
        processed_frames += 1
        
        # Update progress atomically
        with total_processed.get_lock():
            total_processed.value += 1
    
    face_mesh.close()  # Clean up
    if verbose:
        logging.info(f"Worker {worker_id} processed {processed_frames} frames")

def get_total_frames(video_path):
    """Get the total number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def video_reader(video_path, frame_queue, num_workers, start_frame=0, end_frame=None):
    """Reads video file and pushes frames into the frame queue."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        if start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: start_frame ({start_frame}) must be less than end_frame ({end_frame})")
            
        frames_to_process = end_frame - start_frame
        
        print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
        print(f"Processing frames {start_frame} to {end_frame-1} ({frames_to_process} frames)")
        
        # Disabled frame skipping - process every frame regardless of length
        # We have the computational power with A100 to handle this
        process_every_nth = 1
        
        # Skip to the start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        frame_number = start_frame
        processed_count = 0
        
        while cap.isOpened() and frame_number < end_frame:
            success, frame = cap.read()  # Read frame
            if not success:
                break  # End of video
                
            # Only process every nth frame if needed
            if (frame_number - start_frame) % process_every_nth == 0:
                # Wait if queue is getting too full to prevent memory issues
                # Increased queue size limit for high-memory system
                while frame_queue.qsize() > 200:  
                    time.sleep(0.05)
                    
                frame_queue.put((frame_number, frame))  # Queue the frame with original frame number
                processed_count += 1
                
            frame_number += 1
            
            # Print progress occasionally
            if frame_number % 1000 == 0:
                print(f"Read {frame_number - start_frame}/{frames_to_process} frames ({(frame_number - start_frame)/frames_to_process*100:.1f}%)")
            
        cap.release()
        
        print(f"Finished reading {processed_count} frames (from {start_frame} to {frame_number-1})")
        
        # Send termination signals to workers
        for _ in range(num_workers):
            frame_queue.put(None)  # Signal workers to stop
        
        return processed_count
        
    except Exception as e:
        logging.error(f"Error in video reader: {e}")
        # Make sure workers terminate even if reader fails
        for _ in range(num_workers):
            frame_queue.put(None)
        return 0

def selective_video_reader(video_path, frame_queue, num_workers, frames_to_process, start_frame=0, end_frame=None):
    """Reads only the specific frames from a video file and pushes them into the frame queue."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        # Filter the frames to process to ensure they're within range
        frames_to_process = [f for f in frames_to_process if start_frame <= f < end_frame]
        frames_to_process.sort()  # Ensure frames are in order
        
        if not frames_to_process:
            print("No frames to process after filtering!")
            # Send termination signals to workers
            for _ in range(num_workers):
                frame_queue.put(None)
            return 0
            
        print(f"Processing {len(frames_to_process)} specific frames within range {start_frame} to {end_frame-1}")
        
        processed_count = 0
        last_frame_read = -1
        
        for frame_number in frames_to_process:
            # If we need to skip many frames, use seek
            if frame_number > last_frame_read + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
            # Read the frame
            success, frame = cap.read()
            if not success:
                print(f"Failed to read frame {frame_number}")
                continue
                
            last_frame_read = frame_number
            
            # Wait if queue is getting too full to prevent memory issues
            while frame_queue.qsize() > 200:
                time.sleep(0.05)
                
            frame_queue.put((frame_number, frame))
            processed_count += 1
            
            # Print progress occasionally
            if processed_count % 1000 == 0:
                print(f"Read {processed_count}/{len(frames_to_process)} frames ({processed_count/len(frames_to_process)*100:.1f}%)")
            
        cap.release()
        
        print(f"Finished reading {processed_count} frames")
        
        # Send termination signals to workers
        for _ in range(num_workers):
            frame_queue.put(None)
        
        return processed_count
        
    except Exception as e:
        logging.error(f"Error in selective video reader: {e}")
        # Make sure workers terminate even if reader fails
        for _ in range(num_workers):
            frame_queue.put(None)
        return 0

def process_video(video_path, output_csv, word_csv=None, buffer_frames=0, start_frame=0, end_frame=None, num_workers=None, batch_size=1000, progress_bar=True, verbose=False):
    """ 
    Orchestrates video reading, multiprocessing, and saves results to CSV.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_csv : str
        Path where the output CSV will be saved
    word_csv : str, optional
        Path to CSV file containing words with frame numbers (start_frame, end_frame)
    buffer_frames : int, optional
        Number of frames before and after each word to include in processing
    start_frame : int, optional
        First frame to process (inclusive, 0-indexed)
    end_frame : int, optional
        Last frame to process (exclusive). If None, processes until the end
    num_workers : int, optional
        Number of worker processes to use. If None, will use up to 8 CPU cores
    batch_size : int, optional
        Number of results to accumulate before writing to CSV
    progress_bar : bool, optional
        Whether to display a progress bar during processing
    verbose : bool, optional
        Whether to display detailed initialization messages
    
    Returns:
    --------
    int
        Number of frames processed
    """
    # Suppress non-essential messages if not verbose
    if not verbose:
        logging.getLogger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress even more TF logs
        
    if num_workers is None:
        num_workers = max(1, min(8, cpu_count() // 2))  # Increase to 8 workers for A100 GPU
    
    print(f"Processing video with {num_workers} workers...")
    
    # Get total frame count and validate frame range
    total_frames = get_total_frames(video_path)
    
    # If word_csv is provided, determine which frames to process
    frames_to_process_set = None
    word_data = None
    
    if word_csv is not None:
        try:
            # Load the word CSV
            word_df = pd.read_csv(word_csv)
            
            # Clean column names by stripping whitespace
            word_df.columns = [col.strip() for col in word_df.columns]
            
            # Verify required columns exist - now expecting start_frame and end_frame 
            # instead of time-based columns
            required_columns = ['start_frame', 'end_frame']
            for col in required_columns:
                if col not in word_df.columns:
                    # Try case-insensitive match
                    matching_cols = [c for c in word_df.columns if c.lower() == col.lower()]
                    if matching_cols:
                        # Rename the column to the expected name
                        word_df = word_df.rename(columns={matching_cols[0]: col})
                    else:
                        raise ValueError(f"Required column '{col}' not found in word CSV")
            
            # Check if 'word' or 'subtitle' column exists
            word_column = None
            for possible_name in ['word', 'subtitle', 'Word', 'Subtitle']:
                if possible_name in word_df.columns:
                    word_column = possible_name
                    break
            
            if word_column is None:
                raise ValueError("Word CSV must contain a 'word' or 'subtitle' column")
                
            print(f"Loaded word CSV with {len(word_df)} entries")
            
            # Create an empty set to collect all frames that need processing
            frames_to_process_set = set()
            
            # Create a mapping from frame number to word data
            frame_to_word_map = {}
            
            # For each word, calculate its frame range
            for _, row in word_df.iterrows():
                # Direct use of frame numbers rather than converting from time
                start_frame_word = int(row['start_frame'])
                end_frame_word = int(row['end_frame'])
                word = row[word_column]
                
                # Calculate frame numbers with buffer
                start_frame_word_with_buffer = max(0, start_frame_word - buffer_frames)
                end_frame_word_with_buffer = min(total_frames, end_frame_word + buffer_frames + 1)  # +1 for exclusive range
                
                # Add all frames in this range to the set of frames to process
                frames_in_range = list(range(start_frame_word_with_buffer, end_frame_word_with_buffer))
                frames_to_process_set.update(frames_in_range)
                
                # Map each frame to the word data (for frames that actually contain the word, not buffer frames)
                word_data_dict = {
                    'word': word,
                    'start_frame': start_frame_word,
                    'end_frame': end_frame_word
                }
                
                # Add any other columns from the word CSV that might be useful
                for col in word_df.columns:
                    if col not in word_data_dict and col != word_column:
                        word_data_dict[col] = row[col]
                
                # Map each frame to this word data
                for frame in range(start_frame_word, end_frame_word + 1):
                    frame_to_word_map[frame] = word_data_dict
            
            # Convert frame set to sorted list for processing
            frames_to_process_list = sorted(list(frames_to_process_set))
            
            # Store frame to word mapping and the list of frames to process
            word_data = {
                'frame_to_word_map': frame_to_word_map,
                'frames_to_process': frames_to_process_list
            }
            
            print(f"Will process {len(frames_to_process_list)} frames containing words (with buffer)")
            
        except Exception as e:
            logging.error(f"Error processing word CSV: {e}")
            raise ValueError(f"Failed to process word CSV: {e}")
    
    # Validate frame range after potential adjustment from word CSV
    if start_frame < 0:
        start_frame = 0
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range: start_frame ({start_frame}) must be less than end_frame ({end_frame})")
    
    # Calculate the actual number of frames that will be processed
    frames_to_process = 0
    if word_data is not None:
        # Only count frames that are in both the word list and the specified range
        frames_in_range = [f for f in word_data['frames_to_process'] if start_frame <= f < end_frame]
        frames_to_process = len(frames_in_range)
    else:
        # Process all frames in range
        frames_to_process = end_frame - start_frame
    
    print(f"Total video frames: {total_frames}")
    print(f"Processing frame range: {start_frame} to {end_frame-1} ({frames_to_process} frames)")
    
    # Using shared memory for progress tracking
    total_processed = Value('i', 0)
    
    # Create a lock for mediapipe face mesh access
    face_mesh_lock = Lock()
    
    # Initialize model once before starting workers if verbose
    if verbose:
        _ = get_face_detector()  # Initialize the shared detector
    
    frame_queue = Queue(maxsize=300)  # Increased queue size for high-memory system
    output_queue = Queue()  # Queue for results
    
    # Create and start reader process with frame range and specific frames to process
    if word_data is not None:
        # Modified video reader to only read specific frames
        reader = Process(target=selective_video_reader, args=(
            video_path, frame_queue, num_workers, word_data['frames_to_process'], start_frame, end_frame))
    else:
        # Original video reader for all frames
        reader = Process(target=video_reader, args=(video_path, frame_queue, num_workers, start_frame, end_frame))
    
    reader.start()
    
    # Create and start worker processes
    workers = []
    for i in range(num_workers):
        w = Process(target=process_worker, args=(i, frame_queue, output_queue, total_processed, face_mesh_lock, verbose))
        workers.append(w)
        w.start()
    
    # Create dataframe for results
    landmark_columns = [f'L_{i}_{axis}' for i in range(468) for axis in ['x', 'y', 'z']]
    
    # Add word-related columns if we have word data
    result_columns = ['Frame', 'Scene', 'X', 'Y', 'W', 'H', 'Confidence_RetinaFace', 'Mouth_Openness']
    
    if word_data is not None:
        # Add word-related columns to the beginning for better visibility
        word_columns = ['subtitle', 'start_frame', 'end_frame']
        # Add any other columns from the word CSV
        all_extra_columns = set()
        if 'frame_to_word_map' in word_data and word_data['frame_to_word_map']:
            for frame_data in word_data['frame_to_word_map'].values():
                for col in frame_data:
                    if col not in word_columns and col != 'word':
                        all_extra_columns.add(col)
        
        # Add all found columns
        for col in all_extra_columns:
            if col not in word_columns:
                word_columns.append(col)
        
        columns = word_columns + result_columns + landmark_columns
    else:
        columns = result_columns + landmark_columns
    
    # Initialize CSV file with headers
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)
    
    # Process results with progress bar
    all_results = []
    last_processed = 0
    
    if progress_bar:
        # Use the correctly calculated frames_to_process for the progress bar
        pbar = tqdm(total=frames_to_process, desc="Processing frames")
    
    try:
        while total_processed.value < frames_to_process:
            # Update progress bar based on the shared counter
            current = total_processed.value
            if current > last_processed:
                if progress_bar:
                    pbar.update(current - last_processed)
                last_processed = current
            
            # Collect results (with timeout to prevent hanging)
            try:
                # Try to get results with timeout
                result = output_queue.get(timeout=0.1)
                if result:
                    # If we have word data, add it to the results
                    if word_data is not None and 'frame_to_word_map' in word_data:
                        for r in result:
                            frame_num = r[0]  # Frame number is the first element
                            if frame_num in word_data['frame_to_word_map']:
                                word_info = word_data['frame_to_word_map'][frame_num]
                                
                                # Create a new result with word data at the beginning
                                word_values = []
                                for col in word_columns:
                                    if col == 'subtitle':
                                        word_values.append(word_info.get('word', ''))
                                    else:
                                        word_values.append(word_info.get(col, None))
                                
                                new_r = word_values + r
                                all_results.append(new_r)
                            else:
                                # Frame doesn't have a word (it's a buffer frame)
                                # Add empty values for word columns
                                empty_word_values = [None] * len(word_columns)
                                new_r = empty_word_values + r
                                all_results.append(new_r)
                    else:
                        all_results.extend(result)
            except:
                # Queue might be empty, continue
                pass
                
            # Write results to CSV in batches to avoid memory issues
            if len(all_results) >= batch_size:
                df_batch = pd.DataFrame(all_results, columns=columns)
                df_batch.to_csv(output_csv, mode='a', header=False, index=False)
                all_results = []
                
            # Check if processing is complete - all workers done and no frames left
            if not any(w.is_alive() for w in workers) and frame_queue.empty() and output_queue.empty():
                break
            
            time.sleep(0.05)  # Small sleep to reduce CPU usage
    
        # Write any remaining results
        if all_results:
            df_batch = pd.DataFrame(all_results, columns=columns)
            df_batch.to_csv(output_csv, mode='a', header=False, index=False)
    
        # Ensure the progress bar shows completion
        if progress_bar:
            pbar.update(frames_to_process - pbar.n)
            pbar.close()
    except KeyboardInterrupt:
        print("\nInterrupted by user, finishing up...")
    finally:
        # Clean up
        if 'pbar' in locals() and progress_bar:
            pbar.close()
            
        reader.join(timeout=5)
        for w in workers:
            w.join(timeout=5)
    
    print(f"\nProcessing complete. Results saved to: {output_csv}")
    return total_processed.value