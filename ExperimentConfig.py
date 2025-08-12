#%%% CELL 00 – MODULE OVERVIEW
"""
ExperimentConfig.py

Purpose
This module defines canonical experiment constants and converts second-based
timing into integer frame counts. It acts as the single source of truth for
stimulus, timing, arena, and grouping settings used across the pipeline.

Steps
- Declare base settings and metadata.
- Define period specs in seconds.
- Compute integer frame counts and ranges for each period.
- Add an 'Experiment' aggregate and export public names.

Output
- Constants for stimuli, timing, arena, and metadata.
- EXPERIMENTAL_PERIODS with duration_frames (int) and range_* fields.
- stimulus_duration_frames (int), frame_span_sec (float),
  total_duration_frames (int).
"""

#%%% CELL 01 – GENERAL EXPERIMENT VARIABLES
"""
Purpose
Define base configuration for alignment, stimuli, timing, arena size, filename
schema, and experimental groups. Keep 'Loom' as the protocol label.

Steps
- Set stimulus and timing values.
- Define periods in seconds.
- Provide filename schema and experimental groups.
"""

# General switches and alignment
POSE_SCORING = True  # include pose-derived metrics from SLEAP
ALIGNMENT_COL = "VisualStim"  # column with stimulus pulses (0→1 edges)

# Stimulus expectations
STIMULUS_NUMBER = 20                 # expected onsets per run
STIMULUS_DURATION_SEC = 0.5          # stimulus length (sec)
EXPECTED_STIMULUS = STIMULUS_NUMBER + 3  # extra events (e.g., lights-off)

# Timing and arena
FRAME_RATE = 60       # frames per second
ARENA_WIDTH_MM = 30   # arena width (millimetres)
ARENA_HEIGHT_MM = 30  # arena height (millimetres)

# Periods (sec)
EXPERIMENTAL_PERIODS = {
    "Baseline": {"duration_sec": 300},
    "Stimulation": {"duration_sec": 300},
    "Recovery": {"duration_sec": 300},
}

# Filename and grouping metadata
FILENAME_STRUCTURE = ["Experimenter", "Genotype", "Protocol", "Sex", "Age",
                      "Setup", "Camera", "Date", "FlyID", "Extension"]

GROUP_IDENTIFIER = "Protocol"  # metadata field used for grouping runs

# Experimental groups (keep 'Loom' as protocol label)
EXPERIMENTAL_GROUPS = {
    "Control": {
        "label": "Control",                # group name
        "idValue": "20Control_3BlackOut",  # identifier in filename metadata
        "color": "#645769",                # plot color
    },
    "Loom": {
        "label": "Loom",
        "idValue": "20Loom_3BlackOut",
        "color": "#E35B29",
    },
}

#%%% CELL 02 – DERIVED TIMING VARIABLES
"""
Purpose
Convert second-based inputs into frame-based durations and ranges, and summarize
totals. All *_frames values are computed once as integers and reused downstream.

Steps
- Compute frame_span_sec (float) and stimulus_duration_frames (int).
- For each period, compute duration_frames and range_frames (ints).
- Compute totals and add an 'Experiment' aggregate period.
"""

# Derived units
frame_span_sec = 1 / FRAME_RATE  # duration of a single frame (sec, float)
stimulus_duration_frames = int(STIMULUS_DURATION_SEC * FRAME_RATE)  # frames (int)

# Running clocks
current_time_sec = 0.0   # running clock in seconds (float)
current_time_frames = 0  # running clock in frames (int)

# Compute frame-based fields per period
for name, info in EXPERIMENTAL_PERIODS.items():
    dur_sec = info["duration_sec"]                  # seconds (float)
    dur_frames = int(dur_sec * FRAME_RATE)          # frames (int)
    info["duration_frames"] = dur_frames
    info["range_sec"] = (current_time_sec, current_time_sec + dur_sec)
    info["range_frames"] = (current_time_frames, current_time_frames + dur_frames)
    current_time_sec += dur_sec
    current_time_frames += dur_frames

# Totals
total_duration_sec = current_time_sec
total_duration_frames = current_time_frames  # int frames

# Aggregate experiment window
EXPERIMENTAL_PERIODS["Experiment"] = {
    "label": "Experiment",
    "duration_sec": total_duration_sec,
    "duration_frames": total_duration_frames,
    "range_sec": (0.0, total_duration_sec),
    "range_frames": (0, total_duration_frames),
}

# Read-only advisory: do not mutate EXPERIMENTAL_PERIODS after import

#%%% CELL 03 – EXPORTS
"""
Purpose
Expose configuration symbols for downstream modules.

Steps
- List public names in __all__ for 'from ExperimentConfig import *'.
"""

__all__ = ["POSE_SCORING",
           "ALIGNMENT_COL", "STIMULUS_NUMBER", "STIMULUS_DURATION_SEC",
           "EXPECTED_STIMULUS",
           "FRAME_RATE", "ARENA_WIDTH_MM", "ARENA_HEIGHT_MM",
           "frame_span_sec", "stimulus_duration_frames",
           "total_duration_frames", "EXPERIMENTAL_PERIODS",
           "FILENAME_STRUCTURE", "GROUP_IDENTIFIER", "EXPERIMENTAL_GROUPS"]
