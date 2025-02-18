from modules.video_reader import extract_frames
from modules.face_mouth_detector import detect_faces_and_mouths
from modules.data_analysis import analyze_talking, merge_onset_with_frames

def main():
    # Paths
    video_path = "CVMouthReader/data/input/smithscut.mp4"                   # Video
    frames_dir = "CVMouthReader/data/input/frames"                          # Frames folder
    transcript_path = "CVMouthReader/data/input/smithscut.csv"              # Transcript
    face_mouth_dir = "CVMouthReader/data/output/face_mouth_data"            # Frames with data folder
    output_mouth_csv = "CVMouthReader/data/output/face_mouth_data.csv"      # Output mouth openness per frame csv
    transcript_output_csv = "CVMouthReader/data/output/aligned_script.csv"  # Output aligned script 
    

    # Step 1: Extract frames from the video
    # print("Extracting frames from the video...")
    # extract_frames(video_path, frames_dir, extract_fps=30)

    # Step 2: Detect faces and mouth movements
    # print("Detecting faces and mouth movements...")
    # detect_faces_and_mouths(frames_dir, output_mouth_csv, face_mouth_dir, output_frame_interval=1)

    # Step 3: Process the transcript and generate final results
    print("Processing transcript and generating final results...")
    analyze_talking(transcript_path, video_path, transcript_output_csv, fps=30)

    # Step 4: Merge with transcript data

    # Step 5: Visualize
    

if __name__ == "__main__":
    main()