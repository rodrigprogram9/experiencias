#%%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringConfig.py

Purpose
This module centralizes thresholds, smoothing windows, and tolerances used by
the behavior-scoring pipeline. It is imported by Functions/Main to ensure all
classification and validation rules stay consistent across runs.

Steps
- Declare thresholds and smoothing windows.
- Define tolerances for QA checkpoints.
- List scored output columns (pose and non-pose variants).
- Export the public configuration names for easy imports.

Output
- Constants for thresholds, windows, tolerances.
- SCORED_COLUMNS and SCORED_POSE_COLUMNS for saving outputs.
"""

#%%% CELL 01 – SCORING PARAMETERS
"""
Purpose
Define thresholds, windows, and tolerances that govern behavior classification
and checkpoint rules. Values are tuned for typical arena recordings.

Steps
- Set jump/walk thresholds (mm/s) and Layer 2 smoothing window (sec).
- Set binary noise tolerances and allowed NaN/unassigned fractions.
- Keep comments short and clear for quick edits later.
"""

# Thresholds (mm/s)
HIGH_SPEED = 75  # jump detection threshold
LOW_SPEED  = 4   # walk vs stationary threshold

# Smoothing windows (seconds)
LAYER2_AVG_WINDOW_SEC = 0.1  # small smoothing window for layer 2

# Tolerances
NOISE_TOLERANCE         = 2       # frames to smooth binary signals
NAN_TOLERANCE           = 0.0001  # max fraction of NaNs in centroid data
POSE_TRACKING_TOLERANCE = 0.05    # max fraction of missing pose frames
LAYER1_TOLERANCE        = 0.05    # max fraction of unassigned layer-1 frames
BASELINE_EXPLORATION    = 0.2     # min fraction walking during baseline

#%%% CELL 02 – OUTPUT COLUMNS
"""
Purpose
List the columns written to scored CSVs. Include resistant flags and, when
desired, the startle window so downstream tools can visualize it.

Steps
- Keep real column names as they appear in data frames.
- Add 'Resistant' and 'Startle_window' to the saved schema.
- Provide a pose variant with orientation and body-part coordinates.
"""

# Non-pose scored columns
SCORED_COLUMNS = [
    "FrameIndex", "VisualStim", "Stim0", "Stim1",
    "Position_X", "Position_Y", "Speed", "Motion",
    "Layer1", "Layer2", "Resistant", "Behavior",
]

# Pose-scored columns (appended when POSE_SCORING is True)
SCORED_POSE_COLUMNS = [
    "FrameIndex", "Orientation", "View", "View_X", "View_Y",
    "Head_X", "Head_Y", "Thorax_X", "Thorax_Y", "Abdomen_X", "Abdomen_Y",
    "LeftWing_X", "LeftWing_Y", "RightWing_X", "RightWing_Y",
]

#%%% CELL 03 – EXPORTS
"""
Purpose
Expose the public configuration names so notebooks and modules can star-import
safely without pulling private helpers.

Steps
- Declare __all__ with the constants defined above.
"""

__all__ = [
    "HIGH_SPEED", "LOW_SPEED",
    "LAYER2_AVG_WINDOW_SEC",
    "NOISE_TOLERANCE", "NAN_TOLERANCE", "POSE_TRACKING_TOLERANCE",
    "LAYER1_TOLERANCE", "BASELINE_EXPLORATION",
    "SCORED_COLUMNS", "SCORED_POSE_COLUMNS",
]
