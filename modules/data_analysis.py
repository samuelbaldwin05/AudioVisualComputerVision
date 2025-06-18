import pandas as pd
import numpy as np

def save_df_to_csv(df, output_csv):
    """ """
    df.to_csv(output_csv, index=False)
    print(f"Analyzed data saved to {output_csv}")

def analyze_talking(input_csv, stdev_thresh):
    """Analyzes the collected data to determine if characters are talking."""
    df = pd.read_csv(input_csv)

    df["Talking"] = False

    # Iterate over each scene and person to determine talking status
    for scene_number in df["Scene"].unique():
        for person_number in df[df["Scene"] == scene_number]["Person"].unique():
            mouth_openness = df[(df["Scene"] == scene_number) & (df["Person"] == person_number)]["Mouth Openness"].dropna()
            
            if len(mouth_openness) > 1:  # Need at least 2 frames to detect variation
                std_dev = np.std(mouth_openness)

                # If standard deviation exceeds the threshold, mark as talking
                if std_dev > stdev_thresh:
                    df.loc[(df["Scene"] == scene_number) & (df["Person"] == person_number), "Talking"] = True

def analyze_talking_method2(input_csv):
    """ Idea: when no word is spoken we KNOW that mouth is closed (theoretically), base talking 
    determination on if the mouth openness is at least one standard deviation away from this AND
    variation in mouth openness. Start tracking a new talking time (instead of storing 
    audio visual, store the person number talking). Create a confidence of person talking based on
    equation instead of just a threshold, research into ML models that work with finding variation.
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

    return df


def merge_onset_with_frames(onset_csv, frame_csv, fps):
    """ Merges onset times with closest frame times and adds the word and all associated 
    metadata to every frame up until the end_time. """
    # Load the data
    onset_df = pd.read_csv(onset_csv)
    frame_df = pd.read_csv(frame_csv)
    
    # Clean column names by stripping whitespace
    onset_df.columns = [col.strip() for col in onset_df.columns]
    frame_df.columns = [col.strip() for col in frame_df.columns]
    
    # Print columns to help debugging
    print(f"Onset CSV columns: {list(onset_df.columns)}")
    print(f"Frame CSV columns: {list(frame_df.columns)}")
    
    # Verify required columns exist
    required_columns = ['start_time', 'end_time']
    for col in required_columns:
        if col not in onset_df.columns:
            # Try case-insensitive match
            matching_cols = [c for c in onset_df.columns if c.lower() == col.lower()]
            if matching_cols:
                # Rename the column to the expected name
                onset_df = onset_df.rename(columns={matching_cols[0]: col})
            else:
                raise ValueError(f"Required column '{col}' not found in onset CSV")
    
    # Check if 'word' column exists
    word_column = None
    for possible_name in ['word', 'subtitle', 'Word', 'Subtitle']:
        if possible_name in onset_df.columns:
            word_column = possible_name
            break
    
    if word_column is None:
        raise ValueError("Onset CSV must contain a 'word' or 'subtitle' column")
    
    # Calculate frame time for each frame number
    frame_df["frame_time"] = frame_df["Frame"] / fps

    # Sort onset_df by start_time to ensure correct order
    onset_df = onset_df.sort_values(by="start_time").reset_index(drop=True)

    # Define important word-related columns we want to carry over
    word_related_columns = [word_column, 'word_type', 'start_time', 'end_time', 'interval', 'song']
    word_related_columns = [col for col in word_related_columns if col in onset_df.columns]
    
    # Create new columns in the frame df for all word-related data
    for col in word_related_columns:
        if col not in frame_df.columns:
            frame_df[col] = None
    
    # Add subtitle column for compatibility if it doesn't exist yet
    if 'subtitle' not in frame_df.columns:
        frame_df['subtitle'] = ""

    # Iterate through each onset row and assign words and metadata to frames in range
    for i, onset_row in onset_df.iterrows():
        start_time = onset_row["start_time"]
        end_time = onset_row["end_time"] if pd.notna(onset_row["end_time"]) else None
        
        # Determine the end time for this word
        if pd.isna(end_time) or end_time is None:  # If no end_time is provided
            if i + 1 < len(onset_df):
                next_start_time = onset_df.iloc[i + 1]["start_time"]
                end_time = min(start_time + 1.0, next_start_time)
            else:
                end_time = start_time + 1.0  # For the last word without end_time
        else:
            if i + 1 < len(onset_df):
                next_start_time = onset_df.iloc[i + 1]["start_time"]
                end_time = min(end_time, next_start_time)

        # Assign the word and metadata to frames in range
        frame_mask = (frame_df["frame_time"] >= start_time) & (frame_df["frame_time"] < end_time)
        
        # Copy important columns from onset_row to frame_df where the mask is True
        for col in word_related_columns:
            frame_df.loc[frame_mask, col] = onset_row[col]
        
        # Set subtitle explicitly for compatibility
        frame_df.loc[frame_mask, 'subtitle'] = onset_row[word_column]

    # Identify columns to move to the front
    # 1. All word-related columns from the onset file
    front_columns = [col for col in word_related_columns if col in frame_df.columns]
    
    # 2. Add 'subtitle' if it's not already in front_columns
    if 'subtitle' not in front_columns:
        front_columns.append('subtitle')
    
    # 3. Add 'frame_time' to front columns
    if 'frame_time' not in front_columns:
        front_columns.append('frame_time')
    
    # 4. Ensure 'Frame' is one of the first columns after the word data
    remaining_columns = [col for col in frame_df.columns if col not in front_columns]
    
    # If Frame exists in remaining columns, move it to the beginning
    if 'Frame' in remaining_columns:
        remaining_columns.remove('Frame')
        remaining_columns = ['Frame'] + remaining_columns
    
    # Final column order: word data first, then frame number, then all other columns
    reordered_columns = front_columns + remaining_columns
    
    # Reorder the DataFrame columns
    frame_df = frame_df[reordered_columns]

    return frame_df

def convert_time_to_frames(onset_csv, output_csv, fps):
    """
    Converts time-based annotations to frame-based annotations.
    
    Takes a CSV with start_time and end_time (in seconds) and converts them
    to start_frame and end_frame based on the specified FPS, while preserving
    all other columns and data.
    """
    # Load the data
    onset_df = pd.read_csv(onset_csv)
    
    # Clean column names by stripping whitespace
    onset_df.columns = [col.strip() for col in onset_df.columns]
    
    # Print columns to help debugging
    print(f"Input CSV columns: {list(onset_df.columns)}")
    
    # Verify required columns exist
    required_columns = ['start_time', 'end_time']
    for col in required_columns:
        if col not in onset_df.columns:
            # Try case-insensitive match
            matching_cols = [c for c in onset_df.columns if c.lower() == col.lower()]
            if matching_cols:
                # Rename the column to the expected name
                onset_df = onset_df.rename(columns={matching_cols[0]: col})
            else:
                raise ValueError(f"Required column '{col}' not found in input CSV")
    
    # Convert time values to frame numbers
    onset_df['start_frame'] = (onset_df['start_time'] * fps).astype(int)
    onset_df['end_frame'] = (onset_df['end_time'] * fps).astype(int)
    
    # Make sure there are no zero-length frames
    onset_df.loc[onset_df['end_frame'] <= onset_df['start_frame'], 'end_frame'] = onset_df['start_frame'] + 1
    
    # Sort by start_frame for consistent ordering
    onset_df = onset_df.sort_values(by="start_frame").reset_index(drop=True)
    
    # Save the converted dataframe to CSV
    onset_df.to_csv(output_csv, index=False)
    
    print(f"Converted {len(onset_df)} entries from time to frames")
    print(f"Results saved to: {output_csv}")
    
    return onset_df