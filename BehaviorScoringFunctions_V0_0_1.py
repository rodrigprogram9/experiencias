"""
BehaviorScoringFunctions.py

This module provides utility functions that support the Drosophila defensive behavior analysis pipeline.
It includes functions to:
    - Load and interact with experimental folder paths.
    - Preprocess and validate raw data files (e.g., alignment corrections, checking file processing status).
    - Transform data (e.g., filling gaps, cleaning binary columns, speed and orientation calculations).
    - Classify behaviors at various analysis layers (hierarchical, sustained, transient, and resistant behaviors).
    
These functions are designed to modularize the data processing, ensuring clear and efficient transformation 
of raw experimental data into a structured format for further analysis.
"""

import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


CHECKPOINT_ERRORS = {
    'WRONG_LOOM_COUNT': {
        'message': 'Wrong loom count detected.',
        'file_end': 'wrong_stim_count.csv'
    },
    'LOST_CENTROID_POSITION': {
        'message': 'Too many centroid NaNs detected.',
        'file_end': 'many_nans.csv'
    },
    'POSE_MISMATCH': {
        'message': 'Mismatch between tracked and pose data lengths.',
        'file_end': 'no_match_pose.csv'
    },
    'MISSING_POSE_FILE': {
        'message': 'Pose file is missing.',
        'file_end': 'missing_pose.csv'
    },
    'VIEW_NAN_EXCEEDED': {
        'message': 'Too many NaNs in view data.',
        'file_end': 'view_nan_exceeded.csv'
    },
    'UNASSIGNED_BEHAVIOR': {
        'message': 'Too many unassigned behaviors detected.',
        'file_end': 'unassigned_behaviors.csv'
    },
    'NO_EXPLORATION': {
        'message': 'Insufficient exploration during baseline period.',
        'file_end': 'too_little_exploration.csv'
    }
}


def load_path():
    """
    Load the experimental folder path.
    
    Returns:
        str: The selected experimental folder path.
    """
    # Initialize and configure the Tkinter window for the file dialog
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    
    # Open the file dialog for folder selection
    foldername = filedialog.askdirectory(parent=root, title='Choose experimental folder to load files')
    
    # Clean up the Tkinter instance
    root.destroy()
    return foldername


def is_file_already_processed(filename_tracked, group_name, pose_scoring, processed_counters, PATH):
    """
    Determine if a tracked file has already been processed or marked with an error.
    
    Parameters:
        filename_tracked (str): Name of the tracked file.
        group_name (str): Name of the output group folder.
        pose_scoring (bool): Flag indicating if pose scoring is enabled.
        processed_counters (dict): Dictionary with keys 'scored' and 'error' to track processed files.
        PATH (str): Root directory for the experimental data.
    
    Returns:
        bool: True if the file is already processed or labeled as an error, False otherwise.
    """
    # Determine the output folder and expected scored file name based on pose scoring
    if pose_scoring:
        scored_folder = os.path.join(PATH, 'ScoredPose', group_name)
        scored_file = filename_tracked.replace('tracked.csv', 'scored_pose.csv')
    else:
        scored_folder = os.path.join(PATH, 'Scored', group_name)
        scored_file = filename_tracked.replace('tracked.csv', 'scored.csv')
    
    # Check if the scored file already exists
    if scored_file in os.listdir(scored_folder):
        processed_counters['scored'] += 1
        return True

    # Check for any error file labels in the corresponding error folder
    error_folder = os.path.join(PATH, 'Error', group_name)
    for error_key, error_info in CHECKPOINT_ERRORS.items():
        error_file = filename_tracked.replace('tracked.csv', error_info['file_end'])
        if error_file in os.listdir(error_folder):
            processed_counters['error'] += 1
            return True

    return False


def fill_zeros(df, column, max_length):
    """
    Fill gaps in a binary column by setting isolated zeros to one when adjacent to ones.
    
    Parameters:
        df (DataFrame): The input data.
        column (str): The column name to be processed.
        max_length (int): Maximum number of zeros to fill in a sequence.
    
    Returns:
        None: The DataFrame is modified in place.
    """
    x = df[column].to_numpy()
    n = len(x)
    
    # Loop through the array and fill zeros in sequences of ones
    for i in range(n - max_length - 1):
        if x[i] == 1 and x[i + 1] == 0:
            if x[i + 1: i + max_length + 1].sum() > 0:
                x[i + 1] = 1
    df[column] = x


def clean_ones(df, column, min_length):
    """
    Clean isolated ones in a binary column by setting them to zero if their duration is below a threshold.
    
    Parameters:
        df (DataFrame): The input data.
        column (str): The column name to be processed.
        min_length (int): Minimum required consecutive ones to retain them.
    
    Returns:
        None: The DataFrame is modified in place.
    """
    x = df[column].to_numpy()
    n = len(x)
    
    # Loop through the array and clean isolated ones based on min_length criteria
    for i in range(n - min_length - 1):
        if x[i] == 0 and x[i + 1] == 1:
            if x[i + 1: i + 1 + min_length + 1].sum() < 3:
                x[i + 1] = 0
    df[column] = x


def calculate_speed(column_x, column_y, frame_span_sec):
    """
    Calculate the speed of a fly based on x and y coordinate differences.
    
    Parameters:
        column_x (Series): Series of x coordinates.
        column_y (Series): Series of y coordinates.
        frame_span_sec (float): Duration of a single frame.
    
    Returns:
        Series: Calculated speeds rounded to two decimal places.
    """
    # Compute coordinate differences
    diff_x = column_x.diff()
    diff_y = column_y.diff()
    
    # Calculate Euclidean distance and derive speed
    distance = np.sqrt(diff_x**2 + diff_y**2)
    speed = distance / frame_span_sec
    return round(speed, 2)


def determine_view(row):
    """
    Determine the view and corresponding position based on row confidence values.
    
    When all three key positions (Head, Thorax, and Abdomen) are available, the view is chosen
    based on the highest confidence value among 'Left', 'Right', and 'Top'.
    
    For vertical classification (when not all three key positions are available):
      - If at least one of Head or Abdomen coordinates is available (Thorax is optional):
          - If both Head and Abdomen are provided, use the Head coordinate.
          - If only Head is available, use it.
          - If only Abdomen is available, use it.
      - If neither Head nor Abdomen is available (or only the Thorax is provided),
        then the view and coordinates will be set to NaN.
    
    Parameters:
        row (Series): A row from a DataFrame containing position and confidence data.
    
    Returns:
        tuple: (selected_view (str or NaN), view_x (float), view_y (float))
    """
    # If full key positional data is available, select the view based on the confidence values
    if pd.notna(row.get('Head.Position.X')) and pd.notna(row.get('Thorax.Position.X')) and pd.notna(row.get('Abdomen.Position.X')):
        confidences = {
            'Left': row.get('Left.Confidence', 0),
            'Right': row.get('Right.Confidence', 0),
            'Top': row.get('Top.Confidence', 0)
        }
        selected_view = max(confidences, key=confidences.get)
        view_x = row.get(f'{selected_view}.Position.X', np.nan)
        view_y = row.get(f'{selected_view}.Position.Y', np.nan)
    else:
        # Apply new vertical classification:
        # Check if at least one of Head or Abdomen coordinates is present.
        if pd.notna(row.get('Head.Position.X')) or pd.notna(row.get('Abdomen.Position.X')):
            selected_view = 'Vertical'
            # If Head is present, use it; otherwise use Abdomen.
            if pd.notna(row.get('Head.Position.X')):
                view_x = row.get('Head.Position.X', np.nan)
                view_y = row.get('Head.Position.Y', np.nan)
            else:
                view_x = row.get('Abdomen.Position.X', np.nan)
                view_y = row.get('Abdomen.Position.Y', np.nan)
        else:
            # No valid coordinates for vertical (or only Thorax is available)
            selected_view = np.nan
            view_x = np.nan
            view_y = np.nan

    return selected_view, view_x, view_y


def calculate_orientation(pointA_x, pointA_y, pointB_x, pointB_y):
    """
    Calculate the orientation of a fly between two points.

    Parameters:
        pointA_x (float): x coordinate of the first point.
        pointA_y (float): y coordinate of the first point.
        pointB_x (float): x coordinate of the second point.
        pointB_y (float): y coordinate of the second point.

    Returns:
        float: Orientation in degrees (0-360), where 0 corresponds to North.
    """
    # Calculate differences in coordinates
    dx = pointB_x - pointA_x
    dy = pointA_y - pointB_y

    # Compute angle in radians and convert to degrees
    orientation = np.arctan2(dy, dx)
    orientation_degrees = np.degrees(orientation)

    # Normalize to [0, 360] and shift by 90 degrees to set 0 as North
    orientation_degrees = (orientation_degrees + 360) % 360
    orientation_degrees = (orientation_degrees + 90) % 360

    return np.round(orientation_degrees, 2)


def calculate_center_running_average(df, cols, output_cols, window_size):
    """
    Calculate the centered running average for specified DataFrame columns.

    Parameters:
        df (DataFrame): Input data.
        cols (list): List of column names to average.
        output_cols (list): Names of the columns to store the averaged results.
        window_size (int): Window size used for the running average.

    Returns:
        DataFrame: Modified DataFrame with new averaged columns added.
    """
    # Loop through each column and calculate the centered rolling mean
    for col, output_col in zip(cols, output_cols):
        df[output_col] = df[col].rolling(window=(window_size + 1), center=True).mean()
    return df


def hierarchical_classifier(df, columns):
    """
    Assign a single behavior per frame based on a hierarchical scheme.
    
    Parameters:
        df (DataFrame): Input data.
        columns (list): Columns representing different behaviors.
    
    Returns:
        DataFrame: Modified DataFrame with only one behavior scored per frame.
    """
    # Convert behavior columns to a numpy array and compute cumulative sum row-wise
    arr = df[columns].to_numpy(copy=True)
    cumsum = np.cumsum(arr, axis=1)
    
    # Zero out all values after the first non-zero entry in each row
    arr[cumsum > 1] = 0
    df[columns] = arr
    return df


def classify_layer_behaviors(df, average_columns):
    """
    Classify behaviors for each frame by selecting the column with the maximum averaged value.
    
    Parameters:
        df (DataFrame): Input data.
        average_columns (list): Columns containing averaged behavior data.
    
    Returns:
        list: Selected behavior for each frame (column name or NaN if none is above zero).
    """
    action_category_list = []
    
    # Iterate through each row to determine the behavior with highest value
    for i in range(len(df)):
        max_value = -1
        max_col = np.nan
        for col in average_columns:
            if df.loc[i, col] > max_value:
                max_value = df.loc[i, col]
                max_col = col

        action_category_list.append(max_col if max_value > 0 else np.nan)
    
    return action_category_list


def classify_resistant_behaviors(df, RESISTANT_COLUMNS, STARTLE_WINDOW_FRAMES):
    """
    Mark resistant bouts: layer‑2 behaviour that (i) overlaps the startle window
    and (ii) lasts at least MIN_PERSISTENT_DURATION_FRAMES.
    """
    for col in RESISTANT_COLUMNS:
        base = col.replace("resistant_", "")          # map to walk / stationary / freeze
        layer2 = f"layer2_{base}"
        df[col] = 0                                   # create output column

        # find onsets / offsets in the layer‑2 binary trace
        on = df[df[layer2].diff() == 1].index
        off = df[df[layer2].diff() == -1].index
        if len(off) < len(on):                        # handle open bout at file end
            off = np.hstack((off, len(df)))

        # flag bouts that satisfy both duration & overlap criteria
        for a, b in zip(on, off):
            overlaps    = df.loc[a:b, "Startle_window"].sum() >= STARTLE_WINDOW_FRAMES
            if overlaps:
                df.loc[a:b - 1, col] = 1
    return df