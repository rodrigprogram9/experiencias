#%% CELL 00 – MODULE OVERVIEW
"""
CreateDataFramesConfig.py

Purpose
=======
Define default placeholders, caps, and zoom-window offsets used by the CreateDataFrames pipeline.
Imported by CreateDataFramesFunctions and Main to parameterize DataFrame construction and zoom views.

"""

#%% CELL 01 – DATAFRAME DEFAULTS

# Placeholders & sentinel values for missing or special numeric data
DEFAULT_NAN_VALUE      = -999999  # placeholder for missing numeric entries
DEFAULT_DISCRETE_JUMP  = -20000   # sentinel for discrete jump in SpeedMotion
DEFAULT_NO_MOTION      = -10000   # sentinel for no-motion in SpeedMotion

# Caps and thresholds for capped metrics
MAX_WALK_SPEED         = 25       # cap for walking speed in SpeedMotion

#%% CELL 02 – ZOOM WINDOW OFFSETS

# Zoom window offsets around stimulus onset (seconds)
ZOOM_WINDOW_BEFORE_SEC = 1        # seconds before each stimulus onset
ZOOM_WINDOW_AFTER_SEC  = 11       # seconds after each stimulus onset

#%% CELL 03 – EXPORTS
__all__ = [
    "DEFAULT_NAN_VALUE",
    "DEFAULT_DISCRETE_JUMP",
    "DEFAULT_NO_MOTION",
    "MAX_WALK_SPEED",
    "ZOOM_WINDOW_BEFORE_SEC",
    "ZOOM_WINDOW_AFTER_SEC",
]