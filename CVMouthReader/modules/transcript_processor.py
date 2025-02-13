import pandas as pd
import os

def process_transcript(transcript_path, face_mouth_dir, output_csv):
    """
    Processes the transcript and generates final results.
    Args:
        transcript_path (str): Path to the transcript CSV file.
        face_mouth_dir (str): Directory containing face and mouth detection results.
        output_csv (str): Path to save the final results.
    """
    # Load transcript
    transcript = pd.read_csv(transcript_path)

    # Process each word in the transcript
    results = []
    for _, row in transcript.iterrows():
        timestamp = row["Timestamp"]
        speaker = row["Speaker"]
        word = row["Word"]

        # Check if the speaker's mouth is moving at the timestamp
        frame_name = f"frame_{int(timestamp):05d}.jpg"
        face_mouth_file = os.path.join(face_mouth_dir, frame_name.replace(".jpg", ".txt"))

        mouth_moving = False
        if os.path.exists(face_mouth_file):
            with open(face_mouth_file, "r") as f:
                for line in f:
                    x, y, w, h, confidence, detector = line.strip().split(",")
                    mouth_moving = True  # Simplified logic for demonstration

        results.append([timestamp, speaker, word, mouth_moving])

    # Save final results
    results_df = pd.DataFrame(results, columns=["Timestamp", "Speaker", "Word", "Mouth Moving"])
    results_df.to_csv(output_csv, index=False)