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


#----------------------------
# Merging talking with script
#----------------------------

def merge_onset_with_frames(onset_csv, frame_csv, output_csv, fps):
    """Merges onset times with closest frame times and saves to a new CSV."""

    # Load the data
    onset_df = pd.read_csv(onset_csv)  # Must contain 'onset_time' and 'word'
    frame_df = pd.read_csv(frame_csv)  # Must contain 'frame_number' and additional info

    # Calculate frame time for each frame number
    frame_df["frame_time"] = frame_df["frame_number"] / fps

    # Merge by finding the closest frame time for each onset time
    merged_data = []

    for _, onset_row in onset_df.iterrows():
        # Find the closest frame time
        closest_frame = frame_df.iloc[(frame_df['frame_time'] - onset_row['onset_time']).abs().argsort()[:1]]
        
        # Merge row data
        merged_row = onset_row.to_dict() | closest_frame.iloc[0].to_dict()
        merged_data.append(merged_row)

    # Convert to DataFrame and save
    merged_df = pd.DataFrame(merged_data)
    merged_df.to_csv(output_csv, index=False)

    print(f"Merged data saved to {output_csv}")
