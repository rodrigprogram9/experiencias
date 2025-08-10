#%%% CELL 00 - MODULE OVERVIEW
"""
BehaviorScoringMain.py

Purpose
This module runs the end-to-end scoring for Drosophila behavior data. It loads
tracked (and optional pose) CSVs, validates key assumptions with checkpoints,
transforms coordinates to millimetres, computes kinematics, applies layered
classification, marks resistant behaviors, and writes scored outputs aligned to
the first stimulus. It prints a concise header, per-file progress, and a final
summary for quick QA across runs.

Steps
- Import libraries and compute derived values.
- Scan inputs and skip files already processed.
- Load tracked data, clean alignment, and validate stimulus stats.
- Build features, compute speed, and (optionally) process pose.
- Classify Layer 1 and Layer 2 with vectorized selections.
- Mark startle windows and classify resistant behaviors (full overlap).
- Assemble final labels, align to the experiment window, and save.
- Print a summary report at the end.

Output
- Scored CSVs in Scored/ and, if enabled, ScoredPose/.
- Error CSVs in ScoredError/ for failed checkpoints.
- Console header/progress/summary text for each run.
"""

#%%% CELL 01 - IMPORTS AND ENTRY POINT
"""
Purpose
Expose a single entry function used by the runner or notebook. Keep imports
local so the module stays light when inspected and to avoid side effects.

Steps
- Define behavior_scoring_main(PATHconfig, EXPconfig, BSconfig, BSF).
- Import numpy, os, pandas, time inside the function.
"""

def behavior_scoring_main(PATHconfig, EXPconfig, BSconfig, BSF):
    import numpy as np
    import os
    import pandas as pd
    import time
    from collections import deque, Counter
    from pathlib import Path


    #%%% CELL 02 - COMPUTE DERIVED VALUES
    """
    Purpose
    Compute frame-based parameters once and reuse them across the run. All
    *_frames come from ExperimentConfig as ints; frame_span_sec is float.

    Steps
    - Read frame span and experiment length (ints precomputed upstream).
    - Convert smoothing windows from seconds to frames.
    """
    # Frame span (sec) and total frames (int)
    FRAME_SPAN_SEC = EXPconfig.frame_span_sec            # float seconds per frame
    NUMBER_FRAMES = EXPconfig.EXPERIMENTAL_PERIODS["Experiment"]["duration_frames"]

    # Layer 2 smoothing window (sec → frames). int cast is expected here.
    LAYER2_AVG_WINDOW = int(BSconfig.LAYER2_AVG_WINDOW_SEC * EXPconfig.FRAME_RATE)

    #%%% CELL 03 - DEFINE COUNTERS
    """
    Purpose
    Initialize per-error counters and the count of scored files.

    Steps
    - Create counters for each checkpoint type and a scored_files counter.
    """
    error_reading_file = 0
    wrong_stim_count = 0
    wrong_stim_duration = 0
    lost_centroid_position = 0
    pose_mismatch = 0
    missing_pose_file = 0
    unassigned_behavior = 0
    no_exploration = 0
    view_nan_exceeded = 0
    output_len_short = 0
    scored_files = 0

    #%%% CELL 04 – PREP PATHS AND OUTPUT FOLDERS
    """
    Purpose
    Resolve input/output roots as Path objects and ensure destination folders
    exist. Outputs are flat (no per-group subfolders) to simplify skip logic.
    
    Steps
    - Wrap incoming paths as Path(...).
    - Create Scored/ and ScoredError/; add ScoredPose/ when enabled.
    """
    
    # INPUT ROOTS (AS PATHS)
    post_root      = Path(PATHconfig.pPostProcessing)
    tracked_root   = Path(PATHconfig.pTracked)
    pose_root      = Path(PATHconfig.pPose)
    
    # OUTPUT ROOTS (AS PATHS)
    scored_root       = Path(PATHconfig.pScored)
    scored_pose_root  = Path(PATHconfig.pScoredPose)
    error_root        = Path(PATHconfig.pScoredError)
    
    # ENSURE DESTINATIONS
    scored_root.mkdir(parents=True, exist_ok=True)      # mkdir -p
    error_root.mkdir(parents=True, exist_ok=True)
    if EXPconfig.POSE_SCORING:
        scored_pose_root.mkdir(parents=True, exist_ok=True)

    #%%% CELL 05 – PLAN WORK AND PRINT HEADER
    """
    Purpose
    Build the worklist from convenience globs, skip files already processed,
    and print a header that summarizes this run. Initialize timing markers
    used by the per-file loop for ETA and delta reporting.
    
    Steps
    - Materialize tracked files from PATHconfig.gTracked.
    - Ask Functions if each file is already processed.
    - Print a header with totals and skip counts; init timers.
    """
    
    # MATERIALIZE GLOB → LIST[Path]
    tracked_files = sorted(PATHconfig.gTracked)  # gTracked is a generator
    total_files = len(tracked_files)
    
    # PREFILTER USING FLAT ROOTS (NO GROUP FOLDERS)
    processed_counters = {"scored": 0, "error": 0}
    skipped_count, to_process = 0, []
    
    for tracked_path in tracked_files:
        # Functions API expects a filename string; use .name to keep compatibility
        already = BSF.is_file_already_processed(
            tracked_path.name, EXPconfig.POSE_SCORING, processed_counters, PATHconfig
        )
        if already:
            skipped_count += 1
        else:
            to_process.append(tracked_path)  # keep Path for later steps
    
    header_scored = processed_counters.get("scored", 0)
    header_errors = processed_counters.get("error", 0)
    
    # PRINT RUN HEADER
    print(BSF.report_header(
        PATHconfig.pExperimentalRoot, EXPconfig.POSE_SCORING, total_files, len(to_process),
        skipped_count, header_scored, header_errors
    ))
    
    # INIT TIMERS FOR PROGRESS/ETA
    start_time = time.time()  # wall clock for total timing
    prev_time = start_time    # time of previous loop iteration


    #%%% CELL 05a – PER-FILE LOOP & PROGRESS (rolling timing from completed files)
    """
    Purpose
    Iterate over files to process and print progress using a rolling average of
    completed files (window=5). The average s/file and ETA are computed from the
    durations of the most-recent completed files only, which keeps the numbers
    stable and realistic.
    
    Steps
    - Maintain a deque of completed-file durations (maxlen=5).
    - At the start of each iteration, append the *previous* file’s duration.
    - Compute avg s/file and ETA from that deque and the remaining session files.
    - Print the standardized progress line (filename is *not* printed here).
    """
    
    
    # ROLLING WINDOW OF COMPLETED FILE DURATIONS
    durations = deque(maxlen=5)
    session_total = len(to_process)
    session_error_types = Counter()
    session_errors = 0
    
    # Track when the previous file started; used to compute its duration
    last_file_start = None
    
    for idx, tracked_path in enumerate(to_process, start=1):
        # If we have a previous file start, close it now and record its duration.
        # This makes durations contain *completed* files only.
        now = time.time()
        if last_file_start is not None:
            durations.append(now - last_file_start)
    
        # Compute timing metrics from completed files so far.
        # Remaining is based on this session: files left after (idx-1) completed.
        remaining = session_total - (idx - 1)
        avg_s, eta = BSF.compute_timing_metrics(durations, remaining_files=remaining)
    
        # PROGRESS (NO FILENAME HERE)
        print(BSF.report_scoring_line(idx, session_total, avg_s, eta))
    
        # Mark the start of this file; it will be closed at the *next* iteration
        last_file_start = now
    
        # FILENAME (STRING) FOR COMPATIBILITY WITH HELPERS
        filename_tracked = tracked_path.name
    


        #%%% CELL 05b – LOAD TRACKED DATA & VALIDATE STIMULUS
        """
        Purpose
        Load tracked data from a Path, clean alignment pulses, validate stimulus
        count/duration, set the first stimulus index, and check centroid coverage.
        
        Steps
        - Read the tracked CSV via Path.
        - Clean the alignment column with binary helpers.
        - Detect onsets and validate expected count.
        - Validate durations using integer frames from ExperimentConfig.
        - Set first_stim and verify centroid NaN tolerance (percent report).
        """
        
        # FULL PATH TO CURRENT FILE
        tracked_file_path = tracked_path  # Path object from the loop
        
        # READ TRACKED CSV
        try:
            tracked_df = pd.read_csv(tracked_file_path)
        except Exception:
            #CHECKPOINT: Error reading tracked file
            error_key = "ERROR_READING_FILE"
            error_reading_file, err_text = BSF.checkpoint_fail(
                pd.DataFrame(), filename_tracked, error_key,
                error_reading_file, PATHconfig.pScoredError
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue  # skip to next file on read error
        
        # CLEAN ALIGNMENT (SMOOTH SPURIOUS TOGGLES)
        BSF.fill_zeros(tracked_df, EXPconfig.ALIGNMENT_COL, BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(tracked_df, EXPconfig.ALIGNMENT_COL, BSconfig.NOISE_TOLERANCE)
        
        # ONSETS: diff()>0 FINDS 0→1 TRANSITIONS
        stim_indices = tracked_df.index[tracked_df[EXPconfig.ALIGNMENT_COL].diff() > 0].tolist()
        if (not stim_indices) or (len(stim_indices) != EXPconfig.EXPECTED_STIMULUS):
            #CHECKPOINT: Wrong stimulus count
            error_key = "WRONG_STIMULUS_COUNT"
            details = (f"Found {len(stim_indices)} | "
                       f"Expected {EXPconfig.EXPECTED_STIMULUS}")
            wrong_stim_count, err_text = BSF.checkpoint_fail(
                tracked_df, filename_tracked, error_key,
                wrong_stim_count, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue
        
        # DURATION CHECK USING INTEGER FRAMES FROM EXPERIMENTCONFIG
        expected_duration_frames = EXPconfig.stimulus_duration_frames  # int frames
        durations = BSF.bout_duration(tracked_df, EXPconfig.ALIGNMENT_COL)
        min_dur, max_dur = min(durations), max(durations)
        if any(abs(d - expected_duration_frames) > BSconfig.NOISE_TOLERANCE for d in durations):
            #CHECKPOINT: Wrong stimulus duration
            error_key = "WRONG_STIMULUS_DURATION"
            details = (f"Min {min_dur}f, Max {max_dur}f | "
                       f"Expected {expected_duration_frames}f")
            wrong_stim_duration, err_text = BSF.checkpoint_fail(
                tracked_df, filename_tracked, error_key,
                wrong_stim_duration, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue
        
        # FIRST STIMULUS INDEX (ALIGNMENT ANCHOR)
        first_stim = stim_indices[0]
        
        # CENTROID NAN TOLERANCE (PERCENT OF FILE LENGTH)
        total_frames = len(tracked_df)
        pos_nan_count = tracked_df["NormalizedCentroidX"].isna().sum()
        pct_nans = int(round(100 * (pos_nan_count / max(1, total_frames))))
        allowed_pct = int(round(100 * BSconfig.NAN_TOLERANCE))
        if pos_nan_count > (total_frames * BSconfig.NAN_TOLERANCE):
            #CHECKPOINT: Too many centroid NaNs
            error_key = "LOST_CENTROID_POSITION"
            details = f"{pct_nans}% | < {allowed_pct}% allowed"
            lost_centroid_position, err_text = BSF.checkpoint_fail(
                tracked_df, filename_tracked, error_key,
                lost_centroid_position, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue

        #%%% CELL 05c - TRACKED DATA → FEATURES
        """
        Purpose
        Build the transform DataFrame in millimetres, derive motion flags, and
        compute speed. Keep rounding to one place in Main.

        Steps
        - Copy stimulus columns (VisualStim, Stim0, Stim1) and clean them.
        - Convert normalized positions to mm with Y flipped so 0 is top.
        - Create Motion from pixel-change and Speed in mm/s (rounded once).
        """
        # Init transform frame
        transform_df = pd.DataFrame()

        # Frame index preserves original indexing for debug
        transform_df["FrameIndex"] = tracked_df["FrameIndex"]

        # VisualStim (kept as real column name)
        transform_df["VisualStim"] = tracked_df["VisualStim"]
        BSF.fill_zeros(transform_df, "VisualStim", BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, "VisualStim", BSconfig.NOISE_TOLERANCE)

        # Stim0 (kept as real column name)
        transform_df["Stim0"] = tracked_df["Stim0"]
        BSF.fill_zeros(transform_df, "Stim0", BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, "Stim0", BSconfig.NOISE_TOLERANCE)

        # Stim1 (kept as real column name)
        transform_df["Stim1"] = tracked_df["Stim1"]
        BSF.fill_zeros(transform_df, "Stim1", BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, "Stim1", BSconfig.NOISE_TOLERANCE)

        # Positions in millimetres; Y flipped so top of arena is positive
        transform_df["Position_X"] = tracked_df["NormalizedCentroidX"] * EXPconfig.ARENA_WIDTH_MM
        transform_df["Position_Y"] = (
            tracked_df["NormalizedCentroidY"] * EXPconfig.ARENA_HEIGHT_MM
        ) * -1 + EXPconfig.ARENA_HEIGHT_MM  # invert to set top=+ and bottom=0

        # Motion from pixel-change; any positive change implies movement
        transform_df["Motion"] = (tracked_df["PixelChange"] > 0).astype(int)

        # Speed in mm/s (Functions returns floats; rounding only here)
        transform_df["Speed"] = BSF.calculate_speed(
            transform_df["Position_X"], transform_df["Position_Y"], FRAME_SPAN_SEC
        ).round(2)  # one rounding place only

        #%%% CELL 05d – POSE DATA (OPTIONAL)
        """
        Purpose
        Resolve the pose file via Path math, select a per-frame view by confidence,
        apply a vertical fallback, validate NaN rate, convert to millimetres, and
        compute orientation. Any failure triggers a checkpoint with details.
        
        Steps
        - Build pose path from the tracked Path.
        - Validate presence and length match (report counts in details).
        - Choose view (max confidence) and apply vertical fallback.
        - Validate NaN rate (percent) and compute orientation.
        """
        
        if EXPconfig.POSE_SCORING:
            # POSE FILE PATH (BASED ON TRACKED NAME)
            filename_pose = tracked_path.name.replace("tracked.csv", "pose.csv")
            pose_path = Path(PATHconfig.pPose) / filename_pose
        
            if not pose_path.exists():
                #CHECKPOINT: Missing pose file (no details)
                error_key = "MISSING_POSE_FILE"
                missing_pose_file, err_text = BSF.checkpoint_fail(
                    tracked_df, filename_tracked, error_key,
                    missing_pose_file, PATHconfig.pScoredError
                )
                session_errors += 1
                session_error_types[error_key] += 1
                print(BSF.report_error_line(err_text))
                print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
                continue
        
            pose_df = pd.read_csv(pose_path, sep=",")
        
            # SOURCE FILES ARE OFF BY ONE ROW (HEADER DIFFERENCE)
            tracked_len = len(tracked_df)
            pose_len = len(pose_df) - 1
            if tracked_len != pose_len:
                #CHECKPOINT: Pose length mismatch
                error_key = "POSE_MISMATCH"
                details = f"Tracked {tracked_len} | Pose {pose_len}"
                pose_mismatch, err_text = BSF.checkpoint_fail(
                    tracked_df, filename_tracked, error_key,
                    pose_mismatch, PATHconfig.pScoredError, details=details
                )
                session_errors += 1
                session_error_types[error_key] += 1
                print(BSF.report_error_line(err_text))
                print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
                continue
        
            # IF HEAD/THORAX/ABDOMEN EXIST, PICK VIEW WITH HIGHEST CONFIDENCE
            valid_mask = pose_df[["Head.Position.X", "Thorax.Position.X",
                                  "Abdomen.Position.X"]].notna().all(axis=1)
            pose_df["Selected_View"] = "Vertical"  # default until proven
            pose_df.loc[valid_mask, "Selected_View"] = (
                pose_df.loc[valid_mask, ["Left.Confidence", "Right.Confidence",
                                         "Top.Confidence", "Bottom.Confidence"]]
                .idxmax(axis=1).str.replace(".Confidence", "", regex=False)
            )
        
            # COPY COORDS FOR THE SELECTED VIEW INTO View_X / View_Y
            for v in ["Left", "Right", "Top", "Bottom"]:
                m = pose_df["Selected_View"] == v
                pose_df.loc[m, "View_X"] = pose_df.loc[m, f"{v}.Position.X"]
                pose_df.loc[m, "View_Y"] = pose_df.loc[m, f"{v}.Position.Y"]
        
            # BOTTOM → TOP TO UNIFY CEILING/FLOOR AMBIGUITY
            pose_df["View"] = pose_df["Selected_View"].replace({"Bottom": "Top"})
        
            # VERTICAL FALLBACK: USE HEAD IF PRESENT, ELSE ABDOMEN
            mvert = pose_df["View"] == "Vertical"
            pose_df.loc[mvert, "View_X"] = pose_df.loc[mvert, "Head.Position.X"].fillna(
                pose_df.loc[mvert, "Abdomen.Position.X"]
            )
            pose_df.loc[mvert, "View_Y"] = pose_df.loc[mvert, "Head.Position.Y"].fillna(
                pose_df.loc[mvert, "Abdomen.Position.Y"]
            )
        
            # KEEP HUMAN-READABLE VIEW LABEL
            transform_df["View"] = pose_df["View"]
        
            #CHECKPOINT: TOO MANY NANS IN POSE VIEW COORDS (PERCENT)
            view_nan = pose_df["View_X"].isna().sum()
            total_frames = len(tracked_df)
            pct_view_nan = int(round(100 * (view_nan / max(1, total_frames))))
            allowed_pct = int(round(100 * BSconfig.POSE_TRACKING_TOLERANCE))
            if view_nan > (total_frames * BSconfig.POSE_TRACKING_TOLERANCE):
                error_key = "VIEW_NAN_EXCEEDED"
                details = f"{pct_view_nan}% | < {allowed_pct}% allowed"
                view_nan_exceeded, err_text = BSF.checkpoint_fail(
                    tracked_df, filename_tracked, error_key,
                    view_nan_exceeded, PATHconfig.pScoredError, details=details
                )
                session_errors += 1
                session_error_types[error_key] += 1
                print(BSF.report_error_line(err_text))
                print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
                continue
        
            # VIEW IN MILLIMETRES; Y FLIPPED TO MATCH TRACKED CONVENTION
            transform_df["View_X"] = pose_df["View_X"] * EXPconfig.ARENA_WIDTH_MM
            transform_df["View_Y"] = (
                pose_df["View_Y"] * EXPconfig.ARENA_HEIGHT_MM
            ) * -1 + EXPconfig.ARENA_HEIGHT_MM
        
            # BODY PARTS IN MILLIMETRES FOR OPTIONAL DOWNSTREAM ANALYSIS
            for part in ["Head", "Thorax", "Abdomen", "LeftWing", "RightWing"]:
                transform_df[f"{part}_X"] = pose_df[f"{part}.Position.X"] * EXPconfig.ARENA_WIDTH_MM
                transform_df[f"{part}_Y"] = (
                    pose_df[f"{part}.Position.Y"] * EXPconfig.ARENA_HEIGHT_MM
                ) * -1 + EXPconfig.ARENA_HEIGHT_MM
        
            # ORIENTATION (THORAX → VIEW) IN DEGREES, 0 = NORTH
            transform_df["Orientation"] = BSF.calculate_orientation(
                transform_df["Thorax_X"], transform_df["Thorax_Y"],
                transform_df["View_X"],   transform_df["View_Y"]
            )


        #%%% CELL 05f – LAYER 1 – THRESHOLDS
        """
        Purpose
        Apply speed/motion thresholds to produce Layer 1 one-hots and a label. Use
        vectorized selection for clarity and performance, then verify the share of
        unassigned frames against the configured tolerance (reported in percent).
        
        Steps
        - Create one-hots for jump/walk/stationary/freeze.
        - Build 'Layer1' label via np.select.
        - Check unassigned proportion against LAYER1_TOLERANCE (percent detail).
        """
        
        # One-hot columns for Layer 1
        LAYER1_COLUMNS = ["layer1_jump", "layer1_walk",
                          "layer1_stationary", "layer1_freeze", "layer1_none"]
        
        # Jump when speed crosses the high threshold
        transform_df["layer1_jump"] = (transform_df["Speed"] >= BSconfig.HIGH_SPEED).astype(int)
        
        # Walk is between low and high thresholds
        transform_df["layer1_walk"] = (
            (transform_df["Speed"] >= BSconfig.LOW_SPEED) &
            (transform_df["Speed"] < BSconfig.HIGH_SPEED)
        ).astype(int)
        
        # Stationary has motion but low speed
        transform_df["layer1_stationary"] = (
            (transform_df["Speed"] < BSconfig.LOW_SPEED) &
            (transform_df["Motion"] > 0)
        ).astype(int)
        
        # Freeze has no motion at all
        transform_df["layer1_freeze"] = (transform_df["Motion"] == 0).astype(int)
        
        # Vectorized label; default None when no category matches
        conditions = [
            transform_df["layer1_jump"] == 1,
            transform_df["layer1_walk"] == 1,
            transform_df["layer1_stationary"] == 1,
            transform_df["layer1_freeze"] == 1,
        ]
        choices = ["Layer1_Jump", "Layer1_Walk", "Layer1_Stationary", "Layer1_Freeze"]
        transform_df["Layer1"] = np.select(conditions, choices, default=None)
        
        # CHECKPOINT: Too many unassigned Layer 1 frames (percent)
        total_frames = len(tracked_df)  # safe denom for percent
        total_unassigned = transform_df["Layer1"].isna().sum()  # frames with no label
        pct_unassigned = int(round(100 * (total_unassigned / max(1, total_frames))))
        allowed_pct = int(round(100 * BSconfig.LAYER1_TOLERANCE))
        
        if total_unassigned > (total_frames * BSconfig.LAYER1_TOLERANCE):
            error_key = "UNASSIGNED_BEHAVIOR"
            details = f"{pct_unassigned}% | < {allowed_pct}% allowed"
            unassigned_behavior, err_text = BSF.checkpoint_fail(
                tracked_df, filename_tracked, error_key,
                unassigned_behavior, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))  # ├────  ERROR: ...
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue


        #%%% CELL 05g – LAYER 2 – APPLYING SMALL SMOOTHING
        """
        Purpose
        Smooth Layer 1 one-hots, pick a dominant behavior, enforce hierarchy, and
        label Layer 2. Verify sufficient baseline exploration with percent details.
        
        Steps
        - Initialize layer 2 columns and apply centered smoothing.
        - Vectorize dominant behavior selection and flags.
        - Enforce hierarchy and set string label.
        - Check baseline walking fraction (percent vs allowed).
        """
        
        LAYER2_COLUMNS = ["layer2_jump", "layer2_walk",
                          "layer2_stationary", "layer2_freeze", "layer2_none"]
        for behavior in LAYER2_COLUMNS:
            transform_df[behavior] = 0
        
        # CENTERED SMOOTHING (WINDOW = LAYER2_AVG_WINDOW)
        layer2_avg_columns = ["layer2_jump_avg", "layer2_walk_avg",
                              "layer2_stationary_avg", "layer2_freeze_avg"]
        transform_df = BSF.calculate_center_running_average(
            transform_df, LAYER1_COLUMNS, layer2_avg_columns, LAYER2_AVG_WINDOW
        )
        
        # DOMINANT AVERAGED BEHAVIOR (ROW-WISE)
        temp = transform_df[layer2_avg_columns].fillna(-np.inf)
        max_values2 = temp.max(axis=1)
        max_behavior2 = temp.idxmax(axis=1)
        max_behavior2[max_values2 == -np.inf] = None  # no positive evidence → None
        max_behavior2[max_values2 <= 0] = None        # all nonpositive → None
        
        # BINARY FLAGS WITH JUMP PRECEDENCE
        transform_df["layer2_jump"] = (transform_df["layer2_jump_avg"] > 0).astype(int)
        mask = transform_df["layer2_jump"] == 0  # set others when jump is absent
        transform_df.loc[mask, "layer2_walk"] = (max_behavior2[mask] == "layer2_walk_avg").astype(int)
        transform_df.loc[mask, "layer2_stationary"] = (
            (max_behavior2[mask] == "layer2_stationary_avg").astype(int)
        )
        transform_df.loc[mask, "layer2_freeze"] = (max_behavior2[mask] == "layer2_freeze_avg").astype(int)
        
        # NONE FLAG (NO POSITIVES)
        transform_df["layer2_none"] = np.where(transform_df[LAYER2_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0)
        
        # HIERARCHICAL SELECTION (ONE BEHAVIOR PER FRAME)
        transform_df = BSF.hierarchical_classifier(transform_df, LAYER2_COLUMNS)
        
        # LAYER 2 LABEL (VECTORIZED)
        conditions2 = [transform_df["layer2_jump"] == 1, transform_df["layer2_walk"] == 1,
                       transform_df["layer2_stationary"] == 1, transform_df["layer2_freeze"] == 1]
        choices2 = ["Layer2_Jump", "Layer2_Walk", "Layer2_Stationary", "Layer2_Freeze"]
        transform_df["Layer2"] = np.select(conditions2, choices2, default=None)
        
        # CHECKPOINT: TOO LITTLE EXPLORATION DURING BASELINE (PERCENT)
        baseline_frames = EXPconfig.EXPERIMENTAL_PERIODS["Baseline"]["duration_frames"]
        baseline_start = max(0, first_stim - baseline_frames)
        baseline_end = first_stim
        walk_count_baseline = transform_df.loc[baseline_start:baseline_end, "Layer2"].eq("Layer2_Walk").sum()
        
        min_pct = int(round(100 * BSconfig.BASELINE_EXPLORATION))
        walk_pct = int(round(100 * (walk_count_baseline / max(1, baseline_frames))))
        if walk_count_baseline < (BSconfig.BASELINE_EXPLORATION * baseline_frames):
            error_key = "NO_EXPLORATION"
            details = f"Walk {walk_pct}% | > {min_pct}% allowed"
            no_exploration, err_text = BSF.checkpoint_fail(
                transform_df, filename_tracked, error_key,
                no_exploration, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue

        #%%% CELL 05j - RESISTANT BEHAVIORS
        """
        Purpose
        Mark startle windows and classify resistant behaviors that fully cover a
        window. The threshold equals the actual window length, derived from the
        same (start, end) used to mark the window on the timeline.

        Steps
        - Set Startle_window around each stimulus (−1s to +2s from onset).
        - Derive window length in frames from that construction.
        - Classify resistant one-hots using the full-overlap rule.
        - Build a concise 'Resistant' label by priority.
        """
        # Startle_window marks [-1s, +2s] around each onset
        transform_df["Startle_window"] = 0
        for onset in stim_indices:
            start = max(0, onset - EXPconfig.FRAME_RATE)  # 1s before onset
            end = min(len(transform_df) - 1, onset + (EXPconfig.FRAME_RATE * 2))  # 2s after
            transform_df.loc[start:end, "Startle_window"] = 1

        # Use the same math to get the true window length (requires full overlap)
        first_start = max(0, stim_indices[0] - EXPconfig.FRAME_RATE)
        first_end = min(len(transform_df) - 1, stim_indices[0] + (EXPconfig.FRAME_RATE * 2))
        startle_len = (first_end - first_start + 1)  # exact frames in a single window

        # Resistant columns; *_none set later for completeness
        RESISTANT_COLUMNS = ["resistant_walk", "resistant_stationary",
                             "resistant_freeze", "resistant_none"]

        # Full-overlap classification uses the derived length
        BSF.classify_resistant_behaviors(
            transform_df, RESISTANT_COLUMNS, STARTLE_WINDOW_LEN_FRAMES=startle_len
        )

        # None flag when no resistant category is active
        transform_df["resistant_none"] = np.where(
            transform_df[RESISTANT_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0
        )

        # Vectorized resistant label; priority walk > stationary > freeze
        r_conditions = [
            transform_df["resistant_walk"] == 1,
            transform_df["resistant_stationary"] == 1,
            transform_df["resistant_freeze"] == 1
        ]
        r_choices = ["Resistant_Walk", "Resistant_Stationary", "Resistant_Freeze"]
        transform_df["Resistant"] = np.select(r_conditions, r_choices, default=None)

        #%%% CELL 05n - FINAL BEHAVIOR LABEL
        """
        Purpose
        Create the final 'Behavior' label from Layer2 with a resistant override
        for freeze frames. This is done with vectorized masking.

        Steps
        - Map Layer2 labels to canonical names.
        - Override Freeze → Resistant_Freeze when flagged as resistant.
        """
        # Map Layer 2 to base behavior names
        base_map = {"Layer2_Jump": "Jump", "Layer2_Walk": "Walk",
                    "Layer2_Stationary": "Stationary", "Layer2_Freeze": "Freeze"}
        behavior_base = transform_df["Layer2"].map(base_map)  # may contain NaN

        # Override only when Layer2 says Freeze and resistant_freeze is active
        mask_res_freeze = (
            (behavior_base == "Freeze") &
            (transform_df["Resistant"] == "Resistant_Freeze")
        )
        transform_df["Behavior"] = np.where(mask_res_freeze, "Resistant_Freeze", behavior_base)

        #%%% CELL 05m – ALIGN AND SAVE
        """
        Purpose
        Align outputs around the first stimulus and write atomically. Indices rely
        on integer frames from ExperimentConfig; no ad-hoc casting is needed.
        
        Steps
        - Slice from baseline to the end of the experiment window.
        - Save Scored/ and ScoredPose/ (when enabled) using atomic writes.
        - Verify aligned length matches total frames (report details).
        """
        
        # OUTPUT SCHEMA; ENSURE CONFIG INCLUDES ANY FIELDS YOU RELY ON
        output_df = transform_df[BSconfig.SCORED_COLUMNS]
        
        # COMPUTE START/END USING INTS FROM EXPERIMENTCONFIG
        baseline_frames = EXPconfig.EXPERIMENTAL_PERIODS["Baseline"]["duration_frames"]
        total_frames = EXPconfig.EXPERIMENTAL_PERIODS["Experiment"]["duration_frames"]
        start_idx = first_stim - baseline_frames               # frames before first stimulus
        end_idx = first_stim + (total_frames - baseline_frames)  # exclusive end
        
        # SLICE AND REINDEX; THIS BECOMES THE SCORED FILE
        aligned_output_df = output_df.iloc[start_idx:end_idx, :].reset_index(drop=True)
        
        # CHECKPOINT: ALIGNED OUTPUT SHORTER THAN EXPECTED (DETAILS)
        if len(aligned_output_df) != total_frames:
            error_key = "OUTPUT_LEN_SHORT"
            details = f"Aligned {len(aligned_output_df)} | Expected {total_frames}"
            output_len_short, err_text = BSF.checkpoint_fail(
                aligned_output_df, filename_tracked, error_key,
                output_len_short, PATHconfig.pScoredError, details=details
            )
            session_errors += 1
            session_error_types[error_key] += 1
            print(BSF.report_error_line(err_text))
            print(BSF.report_error_filename(filename_tracked.replace("_tracked.csv", "")))
            continue
        
        # SAVE SCORED (PATH-LIKE DESTINATIONS)
        scored_file = filename_tracked.replace("tracked.csv", "scored.csv")
        BSF.write_csv_atomic(
            aligned_output_df, scored_root / scored_file, header=True, index=False
        )
        
        # SAVE SCORED POSE WHEN ENABLED
        if EXPconfig.POSE_SCORING:
            output_pose_df = transform_df[BSconfig.SCORED_POSE_COLUMNS]
            aligned_output_pose_df = output_pose_df.iloc[start_idx:end_idx, :].reset_index(drop=True)
            scored_pose_file = filename_tracked.replace("tracked.csv", "scored_pose.csv")
            BSF.write_csv_atomic(
                aligned_output_pose_df, scored_pose_root / scored_pose_file,
                header=True, index=False
            )
        
        scored_files += 1  # one more successful output



    #%%% CELL 06 – SUMMARY (SESSION vs GLOBAL)
    """
    Purpose
    Summarize the run with a two-column table that shows SESSION and GLOBAL
    error counts and percents, plus a per-error-type breakdown. Precede the
    table with aligned kv lines for elapsed time and file counts.
    
    Steps
    - Compute elapsed time in HHhMM format.
    - Print TIME SCORING, FILES FOUND, FILES SCORED.
    - Scan global stats under the same experiment root.
    - Print the dual SESSION|GLOBAL table using reporting helpers.
    """
    
    # ELAPSED TIME (HHhMM)
    elapsed_seconds_total = max(0.0, time.time() - start_time)
    elapsed_h = int(elapsed_seconds_total // 3600)
    elapsed_m = int((elapsed_seconds_total % 3600) // 60)
    elapsed_hhmm = f"{elapsed_h:02d}h{elapsed_m:02d}"
    
    # SESSION TOTALS AND TYPES (accumulated in the loop)
    files_scored_session = scored_files  # incremented in 05m
    session_total = session_total        # set in 05a
    session_per_type = session_error_types  # Counter from 05a
    
    # GLOBAL STATS UNDER CURRENT EXPERIMENT ROOT
    global_stats = BSF.scan_global_stats(PATHconfig)
    
    # PRINT DUAL SUMMARY WITH PRELUDE (time + counts)
    print(BSF.report_final_summary_dual(
        session_total=session_total,
        session_per_type=session_per_type,
        global_stats=global_stats,
        elapsed_hhmm=elapsed_hhmm,
        files_found=total_files,
        files_scored_session=files_scored_session) +
        BSF.done_duck())


if __name__ == "__main__":
    # This module is intended to be called by a runner or notebook. Import and call:
    # behavior_scoring_main(PATHconfig, EXPconfig, BSconfig, BSF)
    print("Import and call behavior_scoring_main(PATHconfig, EXPconfig, BSconfig, BSF).")
