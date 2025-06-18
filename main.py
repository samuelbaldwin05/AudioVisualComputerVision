from modules.testing import process_video


def main():
    # Paths
    video_path = "data/input/500_days_of_summer.mp4"           # Video
    transcript_csv = "data/input/500_days_of_summer_frames.csv" # Transcript
    output_mouth_csv = "data/output/all_face_mouth_data6.csv"  # Output csv

    # Process video and detect mouth 
    process_video(video_path, output_mouth_csv,
                word_csv=transcript_csv, buffer_frames=10,
                start_frame=89500, end_frame=89999,
                batch_size=100, num_workers=8, progress_bar=True)

if __name__ == "__main__":
    main()