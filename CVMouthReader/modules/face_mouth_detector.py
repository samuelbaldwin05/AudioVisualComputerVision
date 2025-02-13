import cv2
import mediapipe as mp
from mtcnn import MTCNN
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Initialize MTCNN
mtcnn_detector = MTCNN()

def detect_faces_and_mouths(frames_dir, output_dir):
    """
    Detects faces and mouth movements in each frame.
    Args:
        frames_dir (str): Directory containing extracted frames.
        output_dir (str): Directory to save face and mouth detection results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each frame
    for frame_name in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)

        # Detect faces using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mediapipe_results = face_detection.process(rgb_frame)

        faces = []
        if mediapipe_results.detections:
            for detection in mediapipe_results.detections:
                confidence = detection.score[0]
                if confidence >= 0.7:  # MediaPipe confidence threshold
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    faces.append((x, y, w, h, confidence, "mediapipe"))

        # Fall back to MTCNN if MediaPipe is uncertain
        if not faces:
            mtcnn_results = mtcnn_detector.detect_faces(rgb_frame)
            for result in mtcnn_results:
                x, y, w, h = result['box']
                confidence = result['confidence']
                faces.append((x, y, w, h, confidence, "mtcnn"))

        # Save face and mouth detection results
        output_path = os.path.join(output_dir, frame_name.replace(".jpg", ".txt"))
        with open(output_path, "w") as f:
            for (x, y, w, h, confidence, detector) in faces:
                f.write(f"{x},{y},{w},{h},{confidence},{detector}\n")

    print(f"Processed {len(os.listdir(frames_dir))} frames. Results saved to {output_dir}")