import pandas as pd
import os

def process_transcript(face_mouth_dir, output_csv, transcript_path=None):
    """
    Processes the transcript (if provided) and generates final results.
    If no transcript is provided, returns only the timestamp and mouth moving status.
    Args:
        face_mouth_dir (str): Directory containing face and mouth detection results.
        output_csv (str): Path to save the final results.
        transcript_path (str, optional): Path to the transcript CSV file. Defaults to None.
    """
    results = []

    # If transcript is provided, load it
    if transcript_path:
        transcript = pd.read_csv(transcript_path)
    else:
        # If no transcript is provided, process all frames in face_mouth_dir
        transcript = pd.DataFrame(columns=["Timestamp", "Speaker", "Word"])

    # Process each frame in face_mouth_dir
    for frame_file in os.listdir(face_mouth_dir):
        if frame_file.endswith(".txt"):
            # Extract timestamp from the filename
            timestamp = int(frame_file.replace("frame_", "").replace(".txt", ""))

            # Check if the mouth is moving
            face_mouth_file = os.path.join(face_mouth_dir, frame_file)
            mouth_moving = False
            if os.path.exists(face_mouth_file):
                with open(face_mouth_file, "r") as f:
                    for line in f:
                        x, y, w, h, confidence, detector = line.strip().split(",")
                        mouth_moving = True  # Simplified logic for demonstration

            # If transcript is provided, match the timestamp with the transcript
            if transcript_path:
                matching_rows = transcript[transcript["Timestamp"] == timestamp]
                for _, row in matching_rows.iterrows():
                    speaker = row["Speaker"]
                    word = row["Word"]
                    results.append([timestamp, speaker, word, mouth_moving])
            else:
                # If no transcript is provided, return only timestamp and mouth moving status
                results.append([timestamp, None, None, mouth_moving])

    # Save final results
    results_df = pd.DataFrame(results, columns=["Timestamp", "Speaker", "Word", "Mouth Moving"])
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")