#%% CELL 00 - DESCRIPTION
"""
This script processes and classifies Drosophila defensive behaviors in response
to visual stimuli.  The experimental folder must contain:
    - 'Tracked': tracked‑data CSVs (…tracked.csv)
    - 'Pose'    (optional): pose CSVs if POSE_SCORING is enabled.

Pipeline overview
    1. Data Loading:
         - Read tracked data (and pose data when enabled).
    2. Pre‑processing & Validation:
         - Noise filtering, alignment corrections, QC checkpoints.
    3. Data Transformation:
         - Convert coordinates to millimetres; compute speed & motion; derive
           orientation when pose is available.
    4. Layered Behaviour Classification:
         - Multi‑layer thresholds & smoothing → transient/sustained/resistant.
    5. Output Generation:
         - Save scored data into 'Scored' / 'ScoredPose' and print a summary.
"""


#%% CELL 01 - IMPORT LIBRARIES AND SETUP LOGGING
"""
Imports the required libraries, sets up logging, and loads helper functions from the BehaviorScoringFunctions module.
These libraries support numerical operations, file handling, data manipulation, GUI interactions, and logging.
"""

def main_pipeline(experiment_path, EConfig, BSF):
    
    # Import necessary libraries and modules
    import numpy as np
    import os
    import pandas as pd
    import logging

    # Set up logging with timestamped messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    #%% CELL 02 - COMPUTE DERIVED VALUES
    """
    Calculate derived values for frame-based parameters used throughout the pipeline.
    Assumes experimental parameters are defined in the run file:
      - FRAME_RATE, EXPERIMENTAL_PERIODS,
        STARTLE_WINDOW_SEC, MIN_PERSISTENT_DURATION_SEC,
        LAYER2_AVG_WINDOW_SEC, LAYER3_AVG_WINDOW_SEC, LAYER4_AVG_WINDOW_SEC.
    
    Derived values include:
      - FRAME_SPAN_SEC: Duration of a single frame.
      - EXPERIMENTAL_PERIODS: Frame counts for each phase.
      - STARTLE_WINDOW_FRAMES: Number of frames defining the startle period.
      - MIN_PERSISTENT_DURATION_FRAMES: Minimum frames for sustained/resistant behavior.
      - NUMBER_FRAMES: Total number of frames in the trial.
      - LAYER2_AVG_WINDOW, LAYER3_AVG_WINDOW, LAYER4_AVG_WINDOW: Smoothing windows (in frames)
        for behavior classification in the respective layers.
    """
    
    # Calculate basic frame durations and counts
    FRAME_SPAN_SEC = 1 / EConfig.FRAME_RATE
    BASELINE_FRAMES    = EConfig.EXPERIMENTAL_PERIODS['Baseline']['duration_sec']    * EConfig.FRAME_RATE
    STIMULATION_FRAMES = EConfig.EXPERIMENTAL_PERIODS['Stimulation']['duration_sec'] * EConfig.FRAME_RATE
    RECOVERY_FRAMES    = EConfig.EXPERIMENTAL_PERIODS['Recovery']['duration_sec']    * EConfig.FRAME_RATE
    NUMBER_FRAMES = BASELINE_FRAMES + STIMULATION_FRAMES + RECOVERY_FRAMES
    
    # Compute smoothing windows (in frames) for each layer classification
    LAYER2_AVG_WINDOW = int(EConfig.LAYER2_AVG_WINDOW_SEC * EConfig.FRAME_RATE)


    #%% CELL 03 - DEFINE CHECKPOINT ERRORS & INITIALIZE COUNTERS
    """
    Define checkpoint error messages with unique file suffixes and initialize counters.
    These counters track error occurrences and successfully scored files.
    """
    
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
    
    # Initialize error and success counters
    wrong_loom_count = 0
    lost_centroid_position = 0
    pose_mismatch = 0
    missing_pose_file = 0
    unassigned_behavior = 0
    no_exploration = 0
    view_nan_exceeded = 0
    scored_files = 0

    
    #%% CELL 04 - LOAD EXPERIMENTAL FOLDER & SETUP DIRECTORIES
    """
    Prompt the user to select the experimental folder and initialize key directory paths.
    Expected folder structure:
      - 'Tracked': Contains tracked data files (ending with 'tracked.csv').
      - 'Pose': Contains corresponding pose data files (if POSE_SCORING is enabled).
      
    Depending on the POSE_SCORING flag, create the following output directories under the selected folder:
      - If POSE_SCORING is True: 'ScoredPose', 'Scored', 'Error'
      - Otherwise: 'Scored', 'Error'
      
    These directories store the processed/scored data and error logs.
    """
    
    # Use helper function to prompt folder selection
    PATH = os.path.join(experiment_path, "PostProcessing")
    
    # Define input folders for tracked and, optionally, pose data
    tracked_folder = os.path.join(PATH, 'Tracked')
    pose_folder = os.path.join(PATH, 'Pose')
    
    # Determine and create required output folders based on POSE_SCORING flag
    if EConfig.POSE_SCORING:
        output_folders = ['ScoredPose', 'Scored', 'Error']
    else:
        output_folders = ['Scored', 'Error']
    
    for folder in output_folders:
        os.makedirs(os.path.join(PATH, folder), exist_ok=True)
    
    #%% CELL 05 - PROCESS EACH FILE (CHECKPOINTS, TRANSFORMATIONS, CLASSIFICATION)
    """
    Process each tracked file in the 'Tracked' folder and execute the full pipeline for data validation, transformation, 
    and behavior classification. This cell is divided into subcells, each focusing on a specific stage of processing.
    """
    
    #%%% CELL 05a - FILE ITERATION & INITIAL CHECKS
    """
    Iterate over tracked files in the 'Tracked' folder and perform initial checks:
      - Process only files ending with 'tracked.csv'.
      - Skip files already processed (scored or marked with an error) using BSF.is_file_already_processed.
      - Display a live progress bar for pending files.
    """
    from tqdm import tqdm
    
    # Gather and count all tracked.csv files
    tracked_files = sorted(f for f in os.listdir(tracked_folder) if f.endswith('tracked.csv'))
    total_files = len(tracked_files)
    
    # Pre‑filter processed vs. pending files
    processed_counters = {'scored': 0, 'error': 0}
    skipped_count = 0
    to_process = []
    for filename_tracked in tracked_files:
        if BSF.is_file_already_processed(
            filename_tracked,
            "",
            EConfig.POSE_SCORING,
            processed_counters,
            PATH
        ):
            skipped_count += 1
        else:
            to_process.append(filename_tracked)
    
    logging.info(f"{skipped_count} files skipped; processing {len(to_process)} of {total_files}")
    
    # Process pending files with a tqdm progress bar
    processed_count = 0
    for filename_tracked in tqdm(
        to_process,
        desc="Scoring files",
        unit="file",
        bar_format="{desc}: {percentage:6.1f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]"
    ):
        processed_count += 1
        
        #%%% CELL 05b - LOAD TRACKED DATA & INITIAL ALIGNMENT
        """
        Load the tracked data from the current file, apply alignment corrections, and compute stimulus indices for quality control.
        
        Steps:
          - Read the tracked CSV file.
          - Apply alignment corrections on the alignment column (using BSF.fill_zeros and BSF.clean_ones).
          - Compute stimulus indices by detecting changes in the alignment column.
          - Verify that at least one stimulus exists.
          - Define first_stim as the first stimulus index.
          - Validate that the detected stimuli count matches the expected value.
          - Check for excessive missing centroid data using NaN thresholds.
        
        If any check fails, skip processing this file.
        """
        
        # Load tracked data from the current file
        tracked_file_path = os.path.join(tracked_folder, filename_tracked)
        try:
            tracked_df = pd.read_csv(tracked_file_path)
        except Exception as e:
            logging.error(f"Error reading tracked file {filename_tracked}: {e}")
            continue
    
        # Apply alignment corrections on the alignment column
        BSF.fill_zeros(tracked_df, EConfig.ALIGNMENT_COL, EConfig.NOISE_TOLERANCE)
        BSF.clean_ones(tracked_df, EConfig.ALIGNMENT_COL, EConfig.NOISE_TOLERANCE)
    
        # Compute stimulus indices by finding positive differences in the alignment column
        stim_indices = tracked_df.index[tracked_df[EConfig.ALIGNMENT_COL].diff() > 0].tolist()
        if not stim_indices:
            logging.error(f"No stimulus found in file: {filename_tracked}")
            continue
    
        # Define first_stim as the first stimulus index
        first_stim = stim_indices[0]
    
        # Validate that the stimulus count matches the expected value
        if len(stim_indices) != EConfig.EXPECTED_STIMULUS:
            wrong_loom_count += 1
            continue
    
        # Validate centroid data by checking the number of NaNs in 'NormalizedCentroidX'
        pos_nan_count = tracked_df['NormalizedCentroidX'].isna().sum()
        if pos_nan_count > (NUMBER_FRAMES * EConfig.NAN_TOLERANCE):
            lost_centroid_position += 1
            continue


        #%%% CELL 05c - PROCESS POSE DATA (IF ENABLED)
        """
        If POSE_SCORING is enabled, load and preprocess the corresponding pose data.
        
        Steps:
          - Construct the pose file name by replacing 'tracked.csv' with 'pose.csv'.
          - Validate that the pose file exists. If not, save the tracked data to the Error folder, log the error, and skip the file.
          - Load the pose data using pandas (with comma as the separator).
          - Temporarily drop bottom points from the pose data.
          - Validate that the tracked data length matches (pose data length - 1). If not, log the error,
            save the tracked data to the Error folder, increment the pose mismatch counter, and skip the file.
        """
        
        if EConfig.POSE_SCORING:
            filename_pose = filename_tracked.replace("tracked.csv", "pose.csv")
            pose_file_path = os.path.join(pose_folder, filename_pose)
            
            # Validate that the pose file exists
            missing_pose_file_name = filename_tracked.replace("tracked.csv", CHECKPOINT_ERRORS['MISSING_POSE_FILE']['file_end'])
            if not os.path.exists(pose_file_path):
                tracked_df.to_csv(os.path.join(PATH, 'Error', missing_pose_file_name), header=True)
                missing_pose_file += 1
                continue
    
            pose_df = pd.read_csv(pose_file_path, sep=',')
    
            # Temporarily drop bottom points from pose data
            def drop_bottom_points(df):
                columns_to_drop = [col for col in df.columns if 'Bottom' in col]
                return df.drop(columns=columns_to_drop)
            pose_df = drop_bottom_points(pose_df)
            
            # Validate that the length of the tracked data matches (pose data length - 1)
            if len(tracked_df) != len(pose_df) - 1:
                pose_mismatch_file = filename_tracked.replace("tracked.csv", CHECKPOINT_ERRORS['POSE_MISMATCH']['file_end'])
                tracked_df.to_csv(os.path.join(PATH, 'Error', pose_mismatch_file), header=True)
                pose_mismatch += 1
                continue


        #%%% CELL 05d - TRACKED DATA TRANSFORMATIONS
        """
        Transform tracked data into a format for further analysis.
        
        Steps:
          - Align and clean stimulus columns (VisualStim, Stim0, Stim1).
          - Convert normalized centroid positions to millimeter space.
          - Calculate motion (based on pixel changes) and speed (from positional changes).
        """
        
        # Initialize a new DataFrame for transformed data
        transform_df = pd.DataFrame()
        
        # Copy the FrameIndex column
        transform_df['FrameIndex'] = tracked_df['FrameIndex']
        
        # Copy and clean the VisualStim column
        transform_df['VisualStim'] = tracked_df['VisualStim']
        BSF.fill_zeros(transform_df, 'VisualStim', EConfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'VisualStim', EConfig.NOISE_TOLERANCE)
        
        # Copy and clean the Stim0 column
        transform_df['Stim0'] = tracked_df['Stim0']
        BSF.fill_zeros(transform_df, 'Stim0', EConfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'Stim0', EConfig.NOISE_TOLERANCE)
        
        # Conpy and clean the Stim1 column
        transform_df['Stim1'] = tracked_df['Stim1']
        BSF.fill_zeros(transform_df, 'Stim1', EConfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'Stim1', EConfig.NOISE_TOLERANCE)
        
        # Convert normalized pixel positions to millimeter space
        transform_df['Position_X'] = tracked_df['NormalizedCentroidX'] * EConfig.ARENA_WIDTH_MM
        transform_df['Position_Y'] = (tracked_df['NormalizedCentroidY'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
        # Binarize motion based on pixel change
        transform_df['Motion'] = np.where(tracked_df['PixelChange'] > 0, 1, 0)
        
        # Calculate speed based on positional changes
        transform_df['Speed'] = BSF.calculate_speed(transform_df['Position_X'], transform_df['Position_Y'], FRAME_SPAN_SEC).round(2)


        #%%% CELL 05e - POSE DATA TRANSFORMATIONS
        """
        If pose scoring is enabled, transform the pose data for further analysis.
        
        Steps:
          - Determine view orientation and associated coordinates.
          - Convert pose data to millimeter space.
          - Calculate the fly's orientation based on thorax and view positions.
          - Validate that the NaN proportion in view data does not exceed the set tolerance.
        """
        
        if EConfig.POSE_SCORING:
            # Vectorized view determination: identify rows with valid head, thorax, and abdomen positions.
            valid_mask = (
                pose_df['Head.Position.X'].notna() &
                pose_df['Thorax.Position.X'].notna() &
                pose_df['Abdomen.Position.X'].notna()
            )
        
            # Initialize view columns with default values ('Vertical' and NaN positions)
            pose_df['View'] = 'Vertical'
            pose_df['View_X'] = np.nan
            pose_df['View_Y'] = np.nan
        
            # For valid rows, select the view based on maximum confidence values
            confidences = pose_df.loc[valid_mask, ['Left.Confidence', 'Right.Confidence', 'Top.Confidence']]
            selected_view = confidences.idxmax(axis=1).str.replace('.Confidence', '', regex=False)
            pose_df.loc[valid_mask, 'View'] = selected_view
        
            # Map each view to its corresponding position columns
            col_x_map = {'Left': 'Left.Position.X', 'Right': 'Right.Position.X', 'Top': 'Top.Position.X'}
            col_y_map = {'Left': 'Left.Position.Y', 'Right': 'Right.Position.Y', 'Top': 'Top.Position.Y'}
        
            for view in ['Left', 'Right', 'Top']:
                view_mask = valid_mask & (pose_df['View'] == view)
                pose_df.loc[view_mask, 'View_X'] = pose_df.loc[view_mask, col_x_map[view]]
                pose_df.loc[view_mask, 'View_Y'] = pose_df.loc[view_mask, col_y_map[view]]
        
            # Add view information to the transformed DataFrame.
            transform_df['View'] = pose_df['View']
        
            # Check for excessive NaNs in view data.
            view_nan_count = transform_df['View'].isna().sum()
            if view_nan_count > (NUMBER_FRAMES * EConfig.POSE_TRACKING_TOLERANCE):
                error_file = filename_tracked.replace('tracked.csv', CHECKPOINT_ERRORS['VIEW_NAN_EXCEEDED']['file_end'])
                tracked_df.to_csv(os.path.join(PATH, 'Error', error_file), header=True)
                view_nan_exceeded += 1
                continue  # Skip to the next file
        
            # Convert pose coordinates to millimeter space and add to the transformed DataFrame.
            transform_df['View_X'] = pose_df['View_X'] * EConfig.ARENA_WIDTH_MM
            transform_df['View_Y'] = (pose_df['View_Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            transform_df['Head_X'] = pose_df['Head.Position.X'] * EConfig.ARENA_WIDTH_MM
            transform_df['Head_Y'] = (pose_df['Head.Position.Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            transform_df['Thorax_X'] = pose_df['Thorax.Position.X'] * EConfig.ARENA_WIDTH_MM
            transform_df['Thorax_Y'] = (pose_df['Thorax.Position.Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            transform_df['Abdomen_X'] = pose_df['Abdomen.Position.X'] * EConfig.ARENA_WIDTH_MM
            transform_df['Abdomen_Y'] = (pose_df['Abdomen.Position.Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            transform_df['LeftWing_X'] = pose_df['LeftWing.Position.X'] * EConfig.ARENA_WIDTH_MM
            transform_df['LeftWing_Y'] = (pose_df['LeftWing.Position.Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            transform_df['RightWing_X'] = pose_df['RightWing.Position.X'] * EConfig.ARENA_WIDTH_MM
            transform_df['RightWing_Y'] = (pose_df['RightWing.Position.Y'] * EConfig.ARENA_HEIGHT_MM) * -1 + EConfig.ARENA_HEIGHT_MM
        
            # Calculate the orientation using thorax and view positions.
            transform_df['Orientation'] = BSF.calculate_orientation(
                transform_df['Thorax_X'], transform_df['Thorax_Y'],
                transform_df['View_X'], transform_df['View_Y']
            )

    
        #%%% CELL 05f - LAYER 1 - MOVEMENT THRESHOLD
        """
        Classify and denoise behaviors at Layer 1 based on speed and motion thresholds.
        Steps:
          - Define Layer 1 behavior columns.
          - Classify behaviors using defined speed and motion thresholds.
          - Categorize each frame into a single behavior using np.select.
          - Check for excessive unassigned behaviors and log an error if the number exceeds tolerance.
        """
        
        # Define Layer 1 behavior columns.
        LAYER1_COLUMNS = ['layer1_jump', 'layer1_walk', 'layer1_stationary', 'layer1_freeze', 'layer1_none']
        
        # Classify behaviors based on speed/motion thresholds.
        transform_df['layer1_jump'] = np.where(transform_df['Speed'] >= EConfig.HIGH_SPEED, 1, 0)
        transform_df['layer1_walk'] = np.where(
            (transform_df['Speed'] >= EConfig.LOW_SPEED) & (transform_df['Speed'] < EConfig.HIGH_SPEED), 1, 0
        )
        transform_df['layer1_stationary'] = np.where(
            (transform_df['Speed'] < EConfig.LOW_SPEED) & (transform_df['Motion'] > 0), 1, 0
        )
        transform_df['layer1_freeze'] = np.where(transform_df['Motion'] == 0, 1, 0)
        
        # Categorize each frame into a single behavior.
        conditions = [
            transform_df['layer1_jump'] == 1,
            transform_df['layer1_walk'] == 1,
            transform_df['layer1_stationary'] == 1,
            transform_df['layer1_freeze'] == 1,
        ]
        choices = ['Layer1_Jump', 'Layer1_Walk', 'Layer1_Stationary', 'Layer1_Freeze']
        transform_df['Layer1'] = np.select(conditions, choices, default=None)
        
        # Check if too many behaviors remain unassigned.
        total_unassigned_behaviors = transform_df['Layer1'].isna().sum()
        if total_unassigned_behaviors > (NUMBER_FRAMES * EConfig.LAYER1_TOLERANCE):
            error_file = filename_tracked.replace('tracked.csv', CHECKPOINT_ERRORS['UNASSIGNED_BEHAVIOR']['file_end'])
            tracked_df.to_csv(os.path.join(PATH, 'Error', error_file), header=True)
            unassigned_behavior += 1
            continue

    
        #%%% CELL 05g - LAYER 2 - APPLYING SMALL SMOOTHING
        """
        This section handles the classification and denoising of behaviors at Layer 2.
        It applies a small smoothing window to the behaviors classified in Layer 1, further refines the classification, and categorizes the behaviors. 
        Finally, it checks if there was sufficient exploration (walking behavior) during the baseline period.
        
        Steps:
          - Initialize columns for Layer 2 behaviors.
          - Apply a centered running average to smooth the Layer 1 classifications.
          - Compute the row-wise maximum of the smoothed values to determine the dominant behavior.
          - Set binary flags for each behavior, with jump taking precedence.
          - Retain a single behavior per frame using hierarchical classification.
          - Categorize each frame's behavior into a single label.
          - Verify sufficient exploration during the baseline period.
        """
        
        LAYER2_COLUMNS = ['layer2_jump', 'layer2_walk', 'layer2_stationary', 'layer2_freeze', 'layer2_none']
        
        # Initialize Layer 2 columns to zero.
        for behavior in LAYER2_COLUMNS:
            transform_df[behavior] = 0
        
        # Apply smoothing to Layer 1 classifications.
        layer2_avg_columns = ['layer2_jump_avg', 'layer2_walk_avg', 'layer2_stationary_avg', 'layer2_freeze_avg']
        transform_df = BSF.calculate_center_running_average(transform_df, LAYER1_COLUMNS, layer2_avg_columns, LAYER2_AVG_WINDOW)
        
        # Compute row-wise maximum and the corresponding averaged behavior.
        temp = transform_df[layer2_avg_columns].fillna(-np.inf)
        max_values2 = temp.max(axis=1)
        max_behavior2 = temp.idxmax(axis=1)
        # Set rows where all values were -np.inf or maximum value is <= 0 to None.
        max_behavior2[max_values2 == -np.inf] = None
        max_behavior2[max_values2 <= 0] = None
    
        # Set binary flags with jump taking precedence.
        transform_df['layer2_jump'] = (transform_df['layer2_jump_avg'] > 0).astype(int)
        mask = transform_df['layer2_jump'] == 0
        transform_df.loc[mask, 'layer2_walk'] = (max_behavior2[mask] == 'layer2_walk_avg').astype(int)
        transform_df.loc[mask, 'layer2_stationary'] = (max_behavior2[mask] == 'layer2_stationary_avg').astype(int)
        transform_df.loc[mask, 'layer2_freeze'] = (max_behavior2[mask] == 'layer2_freeze_avg').astype(int)
        
        # Assign 'none' flag.
        transform_df['layer2_none'] = np.where(transform_df[LAYER2_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0)
        
        # Retain single behavior per frame.
        transform_df = BSF.hierarchical_classifier(transform_df, LAYER2_COLUMNS)
        
        # Layer 2 categorization using vectorized mapping.
        conditions2 = [
            transform_df['layer2_jump'] == 1,
            transform_df['layer2_walk'] == 1,
            transform_df['layer2_stationary'] == 1,
            transform_df['layer2_freeze'] == 1,
        ]
        choices2 = ['Layer2_Jump', 'Layer2_Walk', 'Layer2_Stationary', 'Layer2_Freeze']
        transform_df['Layer2'] = np.select(conditions2, choices2, default=None)
        
        # CHECKPOINT - TOO LITTLE EXPLORATION DURING BASELINE
        baseline_start = max(0, first_stim - BASELINE_FRAMES)
        baseline_end = first_stim
        walk_count_baseline = transform_df.loc[baseline_start:baseline_end, 'Layer2'].eq('Layer2_Walk').sum()
        
        if walk_count_baseline < EConfig.BASELINE_EXPLORATION * BASELINE_FRAMES:
            new_file = filename_tracked.replace('tracked.csv', CHECKPOINT_ERRORS['NO_EXPLORATION']['file_end'])
            transform_df.to_csv(os.path.join(PATH, 'Error', new_file), header=True)
            no_exploration += 1
            continue

    
        #%%% CELL 05j - RESISTANT BEHAVIORS
        """
        This section handles the classification of resistant behaviors.
        Resistant behaviors are those that persist during a startle window following stimuli.
        
        Steps:
          - Define the startle window based on the stimulus indices.
          - Classify resistant behaviors (Walk, Stationary, Freeze) based on their persistence during the startle window.
          - Apply the classification for resistant behaviors.
          - Determine frames with no resistant behavior.
          - Categorize each frame's resistant behavior into a single label.
        """
        
        # Define startle window by setting frames around each stimulus onset.
        transform_df['Startle_window'] = 0
        for onset in stim_indices:
            start = max(0, onset - EConfig.FRAME_RATE)
            end = min(len(transform_df) - 1, onset + (EConfig.FRAME_RATE * 2))
            transform_df.loc[start:end, 'Startle_window'] = 1
        
        # Define resistant behavior columns.
        RESISTANT_COLUMNS = ['resistant_walk', 'resistant_stationary', 'resistant_freeze', 'resistant_none']
        
        # Classify resistant behaviors using the helper function from BSF.
        BSF.classify_resistant_behaviors(transform_df, RESISTANT_COLUMNS, EConfig.FRAME_RATE*3)
        
        # Determine frames with no resistant behavior.
        transform_df['resistant_none'] = np.where(transform_df[RESISTANT_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0)
        
        # Categorize resistant behaviors.
        transform_df['Resistant'] = pd.Series(dtype='object')
        for i in range(len(transform_df)):
            if transform_df.loc[i, 'resistant_walk'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Walk'
            elif transform_df.loc[i, 'resistant_stationary'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Stationary'
            elif transform_df.loc[i, 'resistant_freeze'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Freeze'

        #%%% CELL 05n - SIMPLIFIED FINAL BEHAVIOR
        """
        This section creates a simplified final `Behavior` column using Layer2 classification,
        with special‐case overrides for any resistant behavior (walk, stationary, or freeze).
        
        Steps:
          - Initialize the Behavior column as object dtype.
          - For each frame:
              - If Layer2_Jump    → Behavior = 'Jump'
              - If Layer2_Walk    → Behavior = 'Resistant_Walk' if resistant_walk==1 else 'Walk'
              - If Layer2_Stationary → Behavior = 'Resistant_Stationary' if resistant_stationary==1 else 'Stationary'
              - If Layer2_Freeze  → Behavior = 'Resistant_Freeze' if resistant_freeze==1 else 'Freeze'
        """
        
        # Initialize the Behavior column
        transform_df['Behavior'] = pd.Series(dtype='object')
        
        # Frame-wise classification
        for i in range(len(transform_df)):
            layer2 = transform_df.loc[i, 'Layer2']
            res   = transform_df.loc[i, 'Resistant']
        
            if layer2 == 'Layer2_Jump':
                transform_df.loc[i, 'Behavior'] = 'Jump'
        
            elif layer2 == 'Layer2_Walk':
                if res == 'Resistant_Walk':
                    transform_df.loc[i, 'Behavior'] = 'Resistant_Walk'
                else:
                    transform_df.loc[i, 'Behavior'] = 'Walk'
        
            elif layer2 == 'Layer2_Stationary':
                if res == 'Resistant_Stationary':
                    transform_df.loc[i, 'Behavior'] = 'Resistant_Stationary'
                else:
                    transform_df.loc[i, 'Behavior'] = 'Stationary'
        
            elif layer2 == 'Layer2_Freeze':
                if res == 'Resistant_Freeze':
                    transform_df.loc[i, 'Behavior'] = 'Resistant_Freeze'
                else:
                    transform_df.loc[i, 'Behavior'] = 'Freeze'



        #%%% CELL 05m - SAVE SCORED FILE
        """
        This section handles saving the scored data to a CSV file.
        Depending on whether pose scoring is enabled, different sets of columns are saved.
        The output is aligned based on the stimulus timing, and the file is saved in the appropriate directory.
        """
        
        # Always output the regular scored file from the transformed DataFrame
        output_df = transform_df[EConfig.SCORED_COLUMNS]
        # Align the output DataFrame to include only the frames around the stimulus
        aligned_output_df = output_df.iloc[int(first_stim - BASELINE_FRAMES):int(first_stim + STIMULATION_FRAMES + RECOVERY_FRAMES), :].reset_index(drop=True)
        
        scored_file = filename_tracked.replace('tracked.csv', 'scored.csv')
        scored_folder = os.path.join(PATH, 'Scored')
        os.makedirs(scored_folder, exist_ok=True)
        aligned_output_df.to_csv(os.path.join(scored_folder, scored_file), header=True, index=False)
        
        # Additionally output the pose-scored file if POSE_SCORING is enabled
        if EConfig.POSE_SCORING:
            output_pose_df = transform_df[EConfig.SCORED_POSE_COLUMNS]
            aligned_output_pose_df = output_pose_df.iloc[int(first_stim - BASELINE_FRAMES):int(first_stim + STIMULATION_FRAMES + RECOVERY_FRAMES), :].reset_index(drop=True)
        
            scored_pose_file = filename_tracked.replace('tracked.csv', 'scored_pose.csv')
            scored_pose_folder = os.path.join(PATH, 'ScoredPose')
            os.makedirs(scored_pose_folder, exist_ok=True)
            aligned_output_pose_df.to_csv(os.path.join(scored_pose_folder, scored_pose_file), header=True, index=False)
        
        scored_files += 1

    
    #%% CELL 06 - PRINT SUMMARY
    
    summary = {
        'Total files found': total_files,
        'Files scored': scored_files,
        'Files skipped': skipped_count,
        'Total errors': total_files - scored_files - skipped_count,
        'Wrong loom count': wrong_loom_count,
        'Lost centroid position': lost_centroid_position,
        'Pose mismatch': pose_mismatch,
        'Missing pose file': missing_pose_file,
        'Excessive view NaNs': view_nan_exceeded,
        'Unassigned behavior': unassigned_behavior,
        'No exploration': no_exploration
    }
    
    print("\nClassification Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main_pipeline()