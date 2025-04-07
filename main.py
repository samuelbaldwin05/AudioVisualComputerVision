from modules.alldata_face_mouth_detector import process_video
from modules.data_analysis import analyze_talking, analyze_talking_method2, merge_onset_with_frames

def main():
    # Paths
    video_path = "data/input/smithscut.mp4"                   # Video
    transcript_path = "data/input/smithscut.csv"              # Transcript
    output_mouth_csv = "data/output/face_mouth_data3.csv"     # Output mouth openness per frame CSV
    transcript_output_csv = "data/input/smithscutwords.csv"   # Output aligned script 
    final_csv = "data/output/final2.csv"                      # Final CSV

    # Step 1: Process video and detect mouth openness
    print("Processing video and detecting mouth openness...")
    process_video(video_path, output_mouth_csv, extract_fps=24)

    # Step 2: Generate audio visual boolean from mouth openess
    # print("Processing transcript and generating final results...")
    # analyze_talking_method2(final_csv, final_csv2)

    # # Step 3: Merge with transcript data
    # print("Merging dataframes:")
    # merge_onset_with_frames(transcript_output_csv, output_mouth_csv, final_csv, fps=24)



if __name__ == "__main__":
    main()