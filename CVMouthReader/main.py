from modules.face_mouth_detector import process_video
from modules.data_analysis import analyze_talking, merge_onset_with_frames

def main():
    # Paths
    video_path = "CVMouthReader/data/input/smithscut.mp4"                   # Video
    transcript_path = "CVMouthReader/data/input/smithscut.csv"              # Transcript
    output_mouth_csv = "CVMouthReader/data/output/face_mouth_data2.csv"     # Output mouth openness per frame CSV
    transcript_output_csv = "CVMouthReader/data/input/smithscutwords.csv"   # Output aligned script 
    final_csv = "CVMouthReader/data/output/final.csv"                       # Final CSV

    # Step 1: Process video and detect mouth openness
    print("Processing video and detecting mouth openness...")
    process_video(video_path, output_mouth_csv, extract_fps=30, num_processes=4)

    # Step 2: Process the transcript and generate final results
    print("Processing transcript and generating final results...")
    analyze_talking(transcript_path, video_path, transcript_output_csv, fps=30)

    # # Step 3: Merge with transcript data
    # print("Merging dataframes:")
    # merge_onset_with_frames(output_mouth_csv, transcript_output_csv, final_csv, fps=30)

    # Step 4: Visualize (if needed)
    # Add visualization logic here if required

if __name__ == "__main__":
    main()