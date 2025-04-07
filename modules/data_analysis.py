import pandas as pd
import numpy as np

#-----------------------
# Calculating if talking
#-----------------------

# METASYNTATIC VARIABLES
STDEV_THRESH = 0.7 

def analyze_talking(input_csv, output_csv):
    """
    Analyzes the collected data to determine if characters are talking.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the analyzed CSV file.
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Add a "Talking" column initialized to False
    df["Talking"] = False

    # Iterate over each scene and person to determine talking status
    for scene_number in df["Scene"].unique():
        for person_number in df[df["Scene"] == scene_number]["Person"].unique():
            mouth_openness = df[(df["Scene"] == scene_number) & (df["Person"] == person_number)]["Mouth Openness"].dropna()
            
            if len(mouth_openness) > 1:  # Need at least 2 frames to detect variation
                std_dev = np.std(mouth_openness)

                # If standard deviation exceeds the threshold, mark as talking
                if std_dev > STDEV_THRESH:
                    df.loc[(df["Scene"] == scene_number) & (df["Person"] == person_number), "Talking"] = True

    # Save the analyzed data
    df.to_csv(output_csv, index=False)
    print(f"Analyzed data saved to {output_csv}")

def analyze_talking_method2(input_csv, output_csv):
    """ Idea: when no word is spoken we KNOW that mouth is closed (theoretically), base talking 
    determination on if the mouth openness is at least one standard deviation away from this AND
    variation in mouth openness. Start tracking a new talking time (instead of storing 
    audio visual, store the person number talking). Create a confidence of person talking based on
    equation instead of just a threshold, research into ML models that work with finding variation.
    
    For each frame, find the difference in new frame mouth openness to previous frame mouth openness.
    If 
    
    """
    df = pd.read_csv(input_csv)

    # Add a "Talking" column initialized to False
    df["Talking"] = False
    df["Variance"] = 0
    # Iterate over each scene and person to determine talking status
    for scene_number in df["Scene"].unique():
        for person_number in df[df["Scene"] == scene_number]["Person"].unique():
            mouth_openness = df[(df["Scene"] == scene_number) & (df["Person"] == person_number)]["Mouth Openness"].dropna().reset_index()
            if len(mouth_openness) > 1:  # Need at least 2 frames to detect variation
                variance_over_duration = 0
                std_dev = np.std(mouth_openness)
                for i in range(len(mouth_openness) - 1):
                    variance = abs(mouth_openness.iloc[1]["Mouth Openness"] - mouth_openness.iloc[0]["Mouth Openness"])
                    print(variance)
                    print(df.loc[(df["Scene"] == scene_number) and (df["Person"] == person_number), "Variance"])
                    df.loc[(df["Scene"] == scene_number) and (df["Person"] == person_number), "Variance"] = variance
                    variance_over_duration += variance
                    if i % 12 == 0 & variance_over_duration >= 0.5 * std_dev:    # Every 12 frames 
                        df.loc[(df["Scene"] == scene_number) & (df["Person"] == person_number), "Talking"] = True

    # Save the analyzed data
    df.to_csv(output_csv, index=False)
    print(f"Analyzed data saved to {output_csv}")

#----------------------------
# Merging talking with script
#----------------------------

def merge_onset_with_frames(onset_csv, frame_csv, output_csv, fps):
    """
    Merges onset times with closest frame times and adds the word to every frame
    up until the end_time (if provided), or up to the next word or a maximum of 1 second.
    Frames without a word are retained with an empty subtitle.
    Saves the result to a new CSV.
    """
    # Load the data
    onset_df = pd.read_csv(onset_csv)  # Must contain 'start_time', 'end_time', and 'word'
    frame_df = pd.read_csv(frame_csv)  # Must contain 'Frame' and additional info

    # Calculate frame time for each frame number
    frame_df["frame_time"] = frame_df["Frame"] / fps

    # Sort onset_df by start_time to ensure correct order
    onset_df = onset_df.sort_values(by="start_time").reset_index(drop=True)

    # Initialize a new column in frame_df for subtitles, defaulting to empty string
    frame_df["subtitle"] = ""
    frame_df["start_time"] = None
    frame_df["end_time"] = None

    # Iterate through each onset row and assign words to frames in range
    for i, onset_row in onset_df.iterrows():
        start_time = onset_row["start_time"]
        end_time = onset_row.get("end_time")  # Use .get() to handle missing end_time
        word = onset_row["subtitle"]

        # Determine the end time for this word
        if pd.isna(end_time):  # If no end_time is provided
            next_start_time = onset_df.iloc[i + 1]["start_time"] if i + 1 < len(onset_df) else None
            end_time = min(start_time + 1.0, next_start_time) if next_start_time is not None else start_time + 1.0
        else:
            next_start_time = onset_df.iloc[i + 1]["start_time"] if i + 1 < len(onset_df) else None
            if next_start_time is not None:
                end_time = min(end_time, next_start_time)

        # Assign the word to frames in range
        frame_mask = (frame_df["frame_time"] >= start_time) & (frame_df["frame_time"] < end_time)
        frame_df.loc[frame_mask, "subtitle"] = word
        frame_df.loc[frame_mask, "start_time"] = start_time
        frame_df.loc[frame_mask, "end_time"] = end_time

    # Save the modified frame_df, ensuring all frames are retained
    frame_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to {output_csv}")
