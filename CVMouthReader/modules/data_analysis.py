import pandas as pd
import numpy as np

# METASYNTATIC VARIABLES
STDEV_THRESH = 0.7

def determine_talking(df, scene_number):
    """
    Determines if a person is talking based on mouth openness variations.
    Args:
        df (pd.DataFrame): DataFrame containing results.
        scene_number (int): Current scene number.
    Returns:
        pd.DataFrame: Updated DataFrame with "Talking" column.
    """
    # Group by person and calculate standard deviation of mouth openness
    for person_number, group in df.groupby("Person"):
        mouth_openness = group["Mouth Openness"].dropna()
        if len(mouth_openness) > 1:  # Need at least 2 frames to detect variation
            std_dev = np.std(mouth_openness)
            # If the standard deviation is above a threshold, the person is talking
            if std_dev > STDEV_THRESH:
                df.loc[(df["Scene"] == scene_number) & (df["Person"] == person_number), "Talking"] = True
    return df

def analyze_data(input_csv, output_csv):
    """
    Analyzes the collected data to determine if characters are talking.
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the analyzed CSV file.
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Add a "Talking" column
    df["Talking"] = False

    # Determine if people are talking
    for scene_number in df["Scene"].unique():
        df = determine_talking(df, scene_number)

    # Save the analyzed data
    df.to_csv(output_csv, index=False)
    print(f"Analyzed data saved to {output_csv}")