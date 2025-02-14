from modules.video_reader import extract_frames
from modules.face_mouth_detector import detect_faces_and_mouths
from modules.transcript_processor import process_transcript

def main():
    # Paths
    video_path = "CVMouthReader/data/input/testvideo.mp4"
    frames_dir = "CVMouthReader/data/output/frames"
    transcript_path = "CVMouthReader/data/transcript.csv"
    face_mouth_dir = "CVMouthReader/data/output/face_mouth_data"
    output_csv = "CVMouthReader/data/results/final_results.csv"

    # # Step 1: Extract frames from the video
    # print("Extracting frames from the video...")
    # extract_frames(video_path, frames_dir)

    # Step 2: Detect faces and mouth movements
    print("Detecting faces and mouth movements...")
    detect_faces_and_mouths(frames_dir, face_mouth_dir)

    # # # Step 3: Process the transcript and generate final results
    # print("Processing transcript and generating final results...")
    # process_transcript(face_mouth_dir, output_csv)

if __name__ == "__main__":
    main()