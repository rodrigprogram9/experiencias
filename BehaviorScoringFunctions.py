#%%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringFunctions.py

Purpose
This module provides utilities for the Drosophila behavior pipeline. It loads and
checks files, transforms signals, computes kinematics, classifies behaviors, and
writes safe outputs. All helpers are kept small and composable for clarity.

Steps
- Declare error registry and checkpoint helpers.
- Provide skip-logic for already-processed files.
- Offer binary clean-up utilities and bout duration tools.
- Compute speed and orientation; pick best view per frame.
- Smooth signals, enforce hierarchy, and classify behaviors.
- Mark resistant bouts with full startle-window overlap.
- Write CSVs atomically and format reporting blocks.

Output
- Error registry (CHECKPOINT_ERRORS) and formatting helpers.
- Data transforms, classifiers, and atomic write function.
- Public helpers used by BehaviorScoringMain.py.
"""

#%%% CELL 01 – IMPORTS
"""
Purpose
Import required libraries. Keep the surface minimal and standard.

Steps
- Import os, numpy, pandas.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

#%%% CELL 02 – ERROR DEFINITIONS & CHECKPOINT
"""
Purpose
Define error keys/messages and provide a checkpoint handler that writes an
error CSV atomically. The handler returns the updated counter and a formatted
error line; Main controls how the line is printed (tree layout).

Steps
- Declare CHECKPOINT_ERRORS with file suffixes and messages.
- Build error output path with Path and write atomically.
- Return (counter+1, 'ERROR: ...' or 'ERROR: ... (details)').
"""

CHECKPOINT_ERRORS = {
    "ERROR_READING_FILE": {
        "message": "Error reading tracked file.",
        "file_end": "error_reading.csv",
    },
    "WRONG_STIMULUS_COUNT": {
        "message": "Wrong stimulus count detected.",
        "file_end": "wrong_stim_count.csv",
    },
    "WRONG_STIMULUS_DURATION": {
        "message": "Wrong stimulus duration detected.",
        "file_end": "wrong_stim_duration.csv",
    },
    "LOST_CENTROID_POSITION": {
        "message": "Too many centroid NaNs detected.",
        "file_end": "many_nans.csv",
    },
    "POSE_MISMATCH": {
        "message": "Mismatch between tracked and pose data lengths.",
        "file_end": "no_match_pose.csv",
    },
    "MISSING_POSE_FILE": {
        "message": "Pose file is missing.",
        "file_end": "missing_pose.csv",
    },
    "VIEW_NAN_EXCEEDED": {
        "message": "Too many NaNs in view data.",
        "file_end": "view_nan_exceeded.csv",
    },
    "UNASSIGNED_BEHAVIOR": {
        "message": "Too many unassigned behaviors detected.",
        "file_end": "unassigned_behaviors.csv",
    },
    "NO_EXPLORATION": {
        "message": "Insufficient exploration during baseline period.",
        "file_end": "too_little_exploration.csv",
    },
    "OUTPUT_LEN_SHORT": {
        "message": "Tracked file length is shorter than expected.",
        "file_end": "tracked_len_short.csv",
    },
}


def checkpoint_fail(df,
                    filename_tracked: str,
                    error_key: str,
                    error_counter: int,
                    error_dir,
                    details: str | None = None) -> tuple[int, str]:
    """
    Handle a failed checkpoint: write df with the error suffix and return
    the updated counter and a formatted error line.

    returns
    - (new_counter, "ERROR: <message>" or "ERROR: <message> (<details>)")
    """
    info = CHECKPOINT_ERRORS[error_key]
    error_file = filename_tracked.replace("tracked.csv", info["file_end"])
    error_path = Path(error_dir) / error_file  # build with Path
    write_csv_atomic(df, error_path, header=True, index=False)

    base = f"ERROR: {info['message']}"
    line = f"{base} ({details})" if details else base
    return error_counter + 1, line

#%%% CELL 03 – FILE STATUS CHECK
"""
Purpose
Check whether a tracked file is already processed or errored using flat-root
folders. Paths are handled with pathlib.Path for clarity and safety.

Steps
- Resolve output roots from PATHconfig as Path objects.
- Map tracked name to expected scored/error files.
- Update counters when a match is found and return True/False.
"""


def is_file_already_processed(filename_tracked,
                              pose_scoring,
                              processed_counters,
                              PATHconfig) -> bool:
    """
    Determine if a tracked file was already scored or labeled as error.

    parameters
    - filename_tracked: tracked filename as string
    - pose_scoring: bool flag for pose scoring destination
    - processed_counters: dict with 'scored' and 'error' counters
    - PATHconfig: config with pScored, pScoredPose, pScoredError

    returns
    - True if a scored file or an error file already exists; else False
    """
    scored_root = Path(PATHconfig.pScoredPose) if pose_scoring else Path(PATHconfig.pScored)
    error_root = Path(PATHconfig.pScoredError)

    # optional safety: if destinations are missing, treat as not processed
    if not scored_root.exists() or not error_root.exists():
        return False

    # scored name mapping
    scored_name = filename_tracked.replace(
        "tracked.csv", "scored_pose.csv" if pose_scoring else "scored.csv"
    )
    scored_path = scored_root / scored_name
    if scored_path.exists():
        processed_counters["scored"] += 1
        return True

    # any matching error file suffices (prefix match on base)
    base = filename_tracked.replace("tracked.csv", "")
    try:
        for err in error_root.iterdir():
            if err.is_file() and err.name.startswith(base):
                processed_counters["error"] += 1
                return True
    except Exception:
        pass  # transient FS errors; assume not processed

    return False

#%%% CELL 04 – BINARY CLEANERS & BOUT UTIL
"""
Purpose
Provide simple binary cleaners and a utility to compute bout durations.

Steps
- Implement fill_zeros and clean_ones (to be replaced by morphology later).
- Implement bout_duration to return lengths of 1-runs in frames.
"""

def fill_zeros(df, column, max_length):
    """
    Fill gaps in a binary column by setting isolated zeros to one when they sit
    inside short gaps. Future version will use windowed morphology (dilation).
    """
    x = df[column].to_numpy()
    n = len(x)

    # scan forward and fill single zeros inside short gaps
    for i in range(n - max_length - 1):
        if x[i] == 1 and x[i + 1] == 0:
            if x[i + 1:i + max_length + 1].sum() > 0:  # any 1 shortly after
                x[i + 1] = 1
    df[column] = x


def clean_ones(df, column, min_length):
    """
    Remove short 1-runs below a length threshold. Future version will use
    windowed morphology (opening) for robustness.
    """
    x = df[column].to_numpy()
    n = len(x)

    # scan forward and zero out spikes shorter than min_length
    for i in range(n - min_length - 1):
        if x[i] == 0 and x[i + 1] == 1:
            if x[i + 1:i + 1 + min_length + 1].sum() < 3:  # short run → drop
                x[i + 1] = 0
    df[column] = x


def bout_duration(df, column):
    """
    Return the frame lengths of each continuous 1-bout in a binary column.
    """
    x = df[column].to_numpy()
    durations, count = [], 0

    for val in x:
        if val == 1:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0

    if count > 0:  # capture an open bout at the end
        durations.append(count)
    return durations

#%%% CELL 05 – KINEMATICS & VIEW/ORIENTATION
"""
Purpose
Compute speed, orientation, and select a per-frame best view for pose use.

Steps
- Calculate speed in mm/s as floats (rounding done at the call site).
- Compute orientation from A→B, normalized to [0, 360) with 0° = North.
- Determine view from confidences or use vertical fallback logic.
"""

import numpy as np
import pandas as pd

def calculate_speed(column_x, column_y, frame_span_sec):
    """
    Return speed in mm/s as float; rounding is applied at the call site.

    Parameters
    - column_x, column_y: coordinate Series in millimetres.
    - frame_span_sec: duration of a single frame in seconds.
    """
    dx = column_x.diff()
    dy = column_y.diff()
    distance = np.sqrt(dx ** 2 + dy ** 2)  # per-frame displacement in mm
    speed = distance / frame_span_sec      # mm/s as float
    return speed.astype(float)

def calculate_orientation(pointA_x, pointA_y, pointB_x, pointB_y):
    """
    Compute orientation from A→B in degrees, normalized to [0, 360) with 0° = North.

    Robustness:
    - Local NumPy import so notebook-level 'np' shadowing can’t break it.
    - Convert inputs to NumPy arrays to bypass pandas __array_ufunc__ dispatch.
    - Preserve index if inputs are pandas Series.
    """
    import numpy as _np

    idx = getattr(pointA_x, "index", None)

    ax = _np.asarray(pointA_x, dtype=float)
    ay = _np.asarray(pointA_y, dtype=float)
    bx = _np.asarray(pointB_x, dtype=float)
    by = _np.asarray(pointB_y, dtype=float)

    dx = bx - ax
    dy = ay - by  # invert y so 0° points "up"/North

    angle = _np.arctan2(dy, dx)
    deg = _np.degrees(angle)
    deg = (deg + 360) % 360    # wrap to [0,360)
    deg = (deg + 90) % 360     # rotate so 0 = North
    deg = _np.round(deg, 2)

    if idx is not None and deg.shape == (len(idx),):
        import pandas as _pd
        return _pd.Series(deg, index=idx, name="Orientation")
    return deg

def determine_view(row):
    """
    Choose a view per frame using confidences. When all three body parts are
    present, pick the view (Left/Right/Top/Bottom) with highest confidence.
    For vertical cases, prefer Head if present, else Abdomen; otherwise NaN.
    Returns: (view_label, x, y) where x/y are the chosen coordinates (or NaN).
    """
    # full key positions available → pick by confidence
    if (pd.notna(row.get("Head.Position.X")) and
        pd.notna(row.get("Thorax.Position.X")) and
        pd.notna(row.get("Abdomen.Position.X"))):
        confidences = {
            "Left": row.get("Left.Confidence", 0),
            "Right": row.get("Right.Confidence", 0),
            "Top": row.get("Top.Confidence", 0),
            "Bottom": row.get("Bottom.Confidence", 0),
        }
        selected = max(confidences, key=confidences.get)
        vx = row.get(f"{selected}.Position.X", np.nan)
        vy = row.get(f"{selected}.Position.Y", np.nan)
        return selected, vx, vy

    # vertical fallback using head or abdomen
    if pd.notna(row.get("Head.Position.X")) or pd.notna(row.get("Abdomen.Position.X")):
        if pd.notna(row.get("Head.Position.X")):  # prefer head when present
            return "Vertical", row.get("Head.Position.X", np.nan), row.get("Head.Position.Y", np.nan)
        return "Vertical", row.get("Abdomen.Position.X", np.nan), row.get("Abdomen.Position.Y", np.nan)

    # no valid coordinates
    return np.nan, np.nan, np.nan



#%%% CELL 06 – SMOOTHING & HIERARCHY
"""
Purpose
Smooth binary traces with a centered running average and enforce exclusivity.

Steps
- Calculate centered running means for given columns.
- Keep at most one positive behavior per frame via hierarchy.
"""

def calculate_center_running_average(df, cols, output_cols, window_size):
    """
    Add centered running means for each column in cols into output_cols.
    """
    for col, out in zip(cols, output_cols):
        df[out] = df[col].rolling(window=(window_size + 1), center=True).mean()
    return df


def hierarchical_classifier(df, columns):
    """
    Keep only the first positive flag per frame across the given columns.
    """
    arr = df[columns].to_numpy(copy=True)
    cumsum = np.cumsum(arr, axis=1)  # row-wise cumulative positives
    arr[cumsum > 1] = 0              # zero out after the first positive
    df[columns] = arr
    return df

#%%% CELL 07 – CLASSIFIERS
"""
Purpose
Provide vectorized selection for layer labels and mark resistant bouts.

Steps
- Choose dominant label by row-wise argmax with a >0 guard.
- Mark resistant bouts when they fully cover the startle window.
"""

def classify_layer_behaviors(df, average_columns):
    """
    Vectorized pick of the column with the maximum averaged value per row.
    Returns a list of column names or NaN where the max value is ≤ 0.
    """
    vals = df[average_columns].to_numpy()
    idx = np.argmax(vals, axis=1)
    max_vals = vals[np.arange(vals.shape[0]), idx]
    out = np.array(average_columns, dtype=object)[idx]
    out[max_vals <= 0] = np.nan  # no positive evidence → NaN
    return out.tolist()


def classify_resistant_behaviors(df, RESISTANT_COLUMNS, STARTLE_WINDOW_LEN_FRAMES):
    """
    Mark resistant bouts that fully overlap a single startle window.

    Notes
    - Requires the bout to fully cover one startle window (full-overlap rule).
    """
    for col in RESISTANT_COLUMNS:
        base = col.replace("resistant_", "")             # walk / stationary / freeze
        layer2 = f"layer2_{base}"
        df[col] = 0

        # find onsets and offsets in the layer-2 trace
        on = df[df[layer2].diff() == 1].index
        off = df[df[layer2].diff() == -1].index
        if len(off) < len(on):                           # open bout at file end
            off = np.hstack((off, len(df)))

        # flag any bout with full startle-window overlap
        for a, b in zip(on, off):
            overlap = df.loc[a:b, "Startle_window"].sum() >= STARTLE_WINDOW_LEN_FRAMES
            if overlap:                                  # full overlap → resistant
                df.loc[a:b - 1, col] = 1
    return df

#%%% CELL 08 – ATOMIC WRITES
"""
Purpose
Write CSVs atomically using a temporary file and an atomic replace. Accept
Path-like destinations and convert once inside the function.

Steps
- Normalize final_path to Path.
- Write to <final>.tmp, fsync, then os.replace to the final path.
- Use str(...) only at the os.replace boundary.
"""


def write_csv_atomic(df, final_path, **to_csv_kwargs) -> None:
    """
    Write a CSV atomically via <final_path>.tmp and os.replace.

    notes
    - downstream code should ignore any *.tmp files during syncing
    """
    final_path = Path(final_path)  # normalize to Path
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    df.to_csv(tmp_path, **to_csv_kwargs)  # write tmp
    # ensure bytes hit disk before replace
    with open(tmp_path, "rb") as _f:
        os.fsync(_f.fileno())

    os.replace(str(tmp_path), str(final_path))  # atomic rename to final


#%%% CELL 09 – REPORTING & GLOBAL STATS
"""
Purpose
Provide helpers to format:
- Section 2 header (exact 75/72 alignment).
- Section 3 pinned progress (two-line bar + metrics) + counts (SCORED / ERROR).
- Section 3 error lines (short labels + 72-col dash-fill).
- Section 4 dual SESSION|GLOBAL summary with fixed-width numeric cells.

Global formatting rules:
- Banner width: 75 (centered, ALL CAPS)
- Content width: 72 (each content line starts at 2 spaces and ends at col 72)
- KV rule: dash-fill starts 2 spaces after the longest label in the block and
  ends 2 spaces before the value; internal separators have 2 spaces around.
- Timestamps = HH:MM:SS; computed durations use lettered style.
- Center truncation for overlong values to fit width 72 (except filename line).
"""

from pathlib import Path
from collections import Counter
import re
import numpy as np

# -------------------------
# Shared formatting constants
# -------------------------
BANNER_WIDTH   = 75
CONTENT_WIDTH  = 72
INDENT         = "  "   # two spaces
VALUE_SEP      = "  "   # two spaces around grouped values and '---'

# Progress bar geometry (fallback only)
BAR_TOTAL = 25
BAR_LEFT  = "["
BAR_RIGHT = "]"
BAR_FILL  = "#"
BAR_EMPTY = "."

# We only want the pinned display, no tree-style per-file prints:
PRINT_TREE_ERRORS = False

# -------------------------
# Low-level helpers
# -------------------------
def _banner_75(title: str) -> str:
    t = title.strip().upper()
    pad = max(BANNER_WIDTH - len(t) - 2, 0)
    left = pad // 2
    right = pad - left
    return "=" * left + " " + t + " " + "=" * right

def _truncate_center(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep_left = (max_len - 3) // 2
    keep_right = max_len - 3 - keep_left
    return s[:keep_left] + "..." + s[-keep_right:]

def _truncate_left(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[-max_len:]
    return "..." + s[-(max_len - 3):]

def _kv_line_72(label: str, value: str, longest_label: int) -> str:
    label = str(label)
    value = str(value)

    left = INDENT + label
    gap = (max(longest_label - len(label), 0) + 2)
    left += " " * gap

    max_value_len = CONTENT_WIDTH - len(left) - 2
    v = _truncate_center(value, max_value_len)

    dash_len = CONTENT_WIDTH - len(left) - 2 - len(v)
    if dash_len < 0:
        v = _truncate_center(v, max_value_len + dash_len)
        dash_len = max(CONTENT_WIDTH - len(left) - 2 - len(v), 0)

    return left + ("-" * dash_len) + "  " + v

def _dash_value_line_72_from_start(value: str) -> str:
    v = str(value)
    max_dashes = CONTENT_WIDTH - len(INDENT) - 2 - len(v)
    if max_dashes < 0:
        v = _truncate_center(v, len(v) + max_dashes)
        max_dashes = CONTENT_WIDTH - len(INDENT) - 2 - len(v)
    return INDENT + ("-" * max_dashes) + "  " + v

def _fmt_duration_lettered(seconds: float) -> str:
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        if sec >= 30:
            m = (m + 1) % 60
            if m == 0:
                h += 1
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{sec:02d}s"

def _fmt_eta_modular(seconds: float) -> str:
    return _fmt_duration_lettered(seconds)

def _progress_bar(idx: int, total: int) -> str:
    total = max(1, int(total))
    idx = max(0, min(int(idx), total))
    inner = BAR_TOTAL - 2
    filled = int(round(inner * (idx / total)))
    filled = max(0, min(filled, inner))
    return f"{BAR_LEFT}{BAR_FILL*filled}{BAR_EMPTY*(inner - filled)}{BAR_RIGHT}"

def _fmt_s_per_file(sec: float | None) -> str | None:
    if sec is None or sec <= 0:
        return None
    s = float(sec)
    if s >= 60.0:
        m = int(s // 60)
        rem = int(round(s - m * 60))
        if rem == 60:
            m += 1
            rem = 0
        return f"{m}m{rem:02d}s/file"
    txt = f"{s:.2f}".rstrip("0").rstrip(".")
    return f"{txt}s/file"

def _metrics_line_72_aligned(file_i: int, file_n: int, sec_per_file: float | None,
                             eta_seconds: float | None, right_start_width: int) -> str:
    i = max(0, int(file_i))
    n = max(1, int(file_n))
    i = min(i, n)

    parts = [f"file {i}/{n}"]
    sfile = _fmt_s_per_file(sec_per_file)
    if sfile:
        parts.append(sfile)
    if eta_seconds is not None:
        parts.append(f"{_fmt_eta_modular(float(eta_seconds))} eta")

    payload = ("  ---  ").join(parts)
    prefix = INDENT + (" " * right_start_width)

    dash_len = max(0, CONTENT_WIDTH - len(prefix) - 2 - len(payload))
    line = prefix + ("-" * dash_len) + "  " + payload
    if len(line) < CONTENT_WIDTH:
        line += " " * (CONTENT_WIDTH - len(line))
    elif len(line) > CONTENT_WIDTH:
        line = line[:CONTENT_WIDTH]
    return line

def _kv_line_72_at_col(label: str, value: str, right_start_width: int) -> str:
    lbl = str(label)
    if len(lbl) > right_start_width:
        lbl = _truncate_center(lbl, right_start_width)
    else:
        lbl = lbl + (" " * (right_start_width - len(lbl)))

    left = INDENT + lbl
    v = str(value)

    max_dashes = CONTENT_WIDTH - len(left) - 2 - len(v)
    if max_dashes < 0:
        v = _truncate_center(v, len(v) + max_dashes)
        max_dashes = CONTENT_WIDTH - len(left) - 2 - len(v)

    return left + ("-" * max_dashes) + "  " + v

def _flex_left_value_line_72(free_left: str, right_value: str) -> str:
    left = INDENT + free_left
    v = str(right_value)
    min_dashes = 4
    base_len = len(left) + 2 + 2 + len(v)
    dash_len = CONTENT_WIDTH - base_len
    if dash_len < min_dashes:
        max_left_total = CONTENT_WIDTH - (2 + 2 + len(v) + min_dashes)
        max_free_left = max(0, max_left_total - len(INDENT))
        free_left = _truncate_center(free_left, max_free_left)
        left = INDENT + free_left
        base_len = len(left) + 2 + 2 + len(v)
        dash_len = max(min_dashes, CONTENT_WIDTH - base_len)
    return left + "  " + ("-" * dash_len) + "  " + v

# -------------------------
# Orientation & smoothing helpers (reused by reporting flows)
# -------------------------
def calculate_center_running_average(df, cols, output_cols, window_size):
    for col, out in zip(cols, output_cols):
        df[out] = df[col].rolling(window=(window_size + 1), center=True).mean()
    return df

def hierarchical_classifier(df, columns):
    arr = df[columns].to_numpy(copy=True)
    cumsum = np.cumsum(arr, axis=1)
    arr[cumsum > 1] = 0
    df[columns] = arr
    return df

# -------------------------
# Header (Section 2)
# -------------------------
def report_header(root: str, pose_flag: bool,
                  total_found: int, to_score: int, skipped: int,
                  already_scored: int, already_errors: int) -> str:
    out = []
    out.append("")
    out.append(_banner_75("SCORING SESSION"))
    out.append("")
    out.append("")

    labels_top = ["PROCESSING", "POSE SCORING"]
    L1 = max(len(s) for s in labels_top)
    out.append(_kv_line_72("PROCESSING", _truncate_center(root, 9999), L1))
    out.append(_kv_line_72("POSE SCORING", str(pose_flag), L1))
    out.append("")

    labels_bot = ["FILES FOUND", "TO SCORE", "SKIPPING"]
    L2 = max(len(s) for s in labels_bot)
    out.append(_kv_line_72("FILES FOUND", f"{total_found}", L2))
    out.append(_kv_line_72("TO SCORE",    f"{to_score}",    L2))
    out.append(_kv_line_72("SKIPPING",    f"{skipped}",     L2))

    summary_val = f"scored: {already_scored}{VALUE_SEP}---{VALUE_SEP}errors: {already_errors}"
    out.append(_kv_line_72("", summary_val, L2))

    return "\n".join(out) + "\n\n"

# -------------------------
# Errors mapping + “last:” snippet
# -------------------------
# NOTE: Do not redefine CHECKPOINT_ERRORS here; use the authoritative mapping from CELL 02.

_SUMMARY_LABELS = {
    "ERROR_READING_FILE":     "error reading file",
    "WRONG_STIMULUS_COUNT":   "wrong stim count",
    "WRONG_STIMULUS_DURATION":"wrong stim duration",
    "LOST_CENTROID_POSITION": "many centroid NaNs",
    "MISSING_POSE_FILE":      "missing pose file",
    "POSE_MISMATCH":          "tracked/pose mismatch",
    "VIEW_NAN_EXCEEDED":      "many sleap view NaNs",
    "UNASSIGNED_BEHAVIOR":    "many unassigned behavior",
    "NO_EXPLORATION":         "low baseline exploration",
    "OUTPUT_LEN_SHORT":       "short output length",
}
_SUMMARY_ORDER = [
    "ERROR_READING_FILE",
    "WRONG_STIMULUS_COUNT",
    "WRONG_STIMULUS_DURATION",
    "LOST_CENTROID_POSITION",
    "MISSING_POSE_FILE",
    "POSE_MISMATCH",
    "VIEW_NAN_EXCEEDED",
    "UNASSIGNED_BEHAVIOR",
    "NO_EXPLORATION",
    "OUTPUT_LEN_SHORT",
]

def _label_for_summary(error_key: str) -> str:
    return _SUMMARY_LABELS.get(error_key, CHECKPOINT_ERRORS[error_key]["message"].lower())

_LAST_ERROR_LABEL = "–"
_LAST_ERROR_VALUE = ""

def get_last_error_label() -> str:
    if _LAST_ERROR_VALUE:
        return f"{_LAST_ERROR_LABEL} {_LAST_ERROR_VALUE}"
    return _LAST_ERROR_LABEL

def _percent_from_text(txt: str) -> str | None:
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', txt)
    if m:
        val = float(m.group(1))
        return f"{int(round(val))}%"
    return None

def _delta_from_two_numbers(txt: str) -> str | None:
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', txt)
    if len(nums) >= 2:
        a = float(nums[0]); b = float(nums[1])
        d = int(round(abs(b - a)))
        return f"Δ{d}"
    return None

def _detail_snippet(error_key: str, err_text: str) -> tuple[str, str]:
    short = _label_for_summary(error_key)
    val = ""
    if error_key == "WRONG_STIMULUS_COUNT":
        d = _delta_from_two_numbers(err_text); val = f"({d})" if d else ""
    elif error_key == "WRONG_STIMULUS_DURATION":
        d = _delta_from_two_numbers(err_text); val = f"({d})" if d else ""
    elif error_key == "LOST_CENTROID_POSITION":
        p = _percent_from_text(err_text);      val = f"({p})" if p else ""
    elif error_key == "POSE_MISMATCH":
        d = _delta_from_two_numbers(err_text); val = f"({d})" if d else ""
    elif error_key in ("VIEW_NAN_EXCEEDED", "UNASSIGNED_BEHAVIOR", "NO_EXPLORATION"):
        p = _percent_from_text(err_text);      val = f"({p})" if p else ""
    elif error_key == "OUTPUT_LEN_SHORT":
        d = _delta_from_two_numbers(err_text); val = f"({d})" if d else ""
    return short, val

# Per-file error lines (tree-style). Suppressed when PRINT_TREE_ERRORS=False.
def report_error_line(error_key: str, err_text: str) -> str:
    global _LAST_ERROR_LABEL, _LAST_ERROR_VALUE
    short, val = _detail_snippet(error_key, err_text)
    _LAST_ERROR_LABEL = short
    _LAST_ERROR_VALUE = val
    if not PRINT_TREE_ERRORS:
        return ""  # nothing to print (prevents blank lines when guarded by caller)
    label = f"└ ERROR: {short}"
    detail = err_text[err_text.find("(")+1:err_text.rfind(")")] if "(" in err_text and ")" in err_text else err_text
    if error_key == "NO_EXPLORATION" and detail:
        detail = detail.replace("Walk ", "").replace("| >", "(>")
        if not detail.endswith(")"):
            detail += ")"
    return _kv_line_72(label, detail, longest_label=len(label))

def report_error_filename(basename: str) -> str:
    if not PRINT_TREE_ERRORS:
        return ""  # nothing to print
    return f"{INDENT}  └ {basename}"  # no trailing \n (tqdm.write adds it)

# -------------------------
# Summary table (strict 6-column layout)
# -------------------------
_COL1 = 2
_COL2 = 8
_COL3 = 33
_COL4 = 13
_COL5 = 3
_COL6 = 13

def _center_width(s: str, w: int = 11) -> str:
    s = str(s)
    if len(s) > w:
        s = s[:w]
    left = (w - len(s)) // 2
    right = w - len(s) - left
    return (" " * left) + s + (" " * right)

def _errors_table_header(_dash_col_ignored: int = 0) -> str:
    col1 = INDENT
    col2 = "ERRORS".ljust(_COL2)
    col3 = ("-" * 31) + "  "
    col4 = "|" + _center_width("SESSION") + "|"
    col5 = "-" * _COL5
    col6 = "|" + _center_width("GLOBAL") + "|"
    return col1 + col2 + col3 + col4 + col5 + col6

def _errors_total_row(_dash_col_ignored: int, sess_total_str: str, glob_total_str: str) -> str:
    col1 = INDENT
    col2 = "TOTAL".ljust(_COL2)
    col3 = ("-" * 31) + "  "
    col4 = "|" + _center_width(sess_total_str) + "|"
    col5 = "-" * _COL5
    col6 = "|" + _center_width(glob_total_str) + "|"
    return col1 + col2 + col3 + col4 + col5 + col6

def _build_col3_detail(label: str) -> str:
    base = str(label)
    max_label = _COL3 - 2 - 2 - 1
    if len(base) > max_label:
        base = _truncate_center(base, max_label)
    dash_len = _COL3 - len(base) - 2 - 2
    if dash_len < 1:
        dash_len = 1
    return f"{base}  {'-'*dash_len}  "

def _errors_detail_row(stub_label: str, _dash_col_ignored: int, sess_cnt: int, glob_cnt: int) -> str:
    col1 = INDENT
    col2 = "-----".ljust(_COL2)
    col3 = _build_col3_detail(stub_label)
    col4 = "|" + _center_width(str(sess_cnt)) + "|"
    col5 = "-" * _COL5
    col6 = "|" + _center_width(str(glob_cnt)) + "|"
    return col1 + col2 + col3 + col4 + col5 + col6

def _pct_int(numer: int, denom: int) -> int:
    if denom <= 0:
        return 0
    return int(round(100 * numer / denom))

def scan_global_stats(PATHconfig) -> dict:
    p_scored = Path(PATHconfig.pScored)
    p_pose   = Path(PATHconfig.pScoredPose)
    p_error  = Path(PATHconfig.pScoredError)

    total_scored = sum(1 for _ in p_scored.rglob("*.csv")) if p_scored.exists() else 0
    total_pose   = sum(1 for _ in p_pose.rglob("*.csv"))   if p_pose.exists()   else 0
    total_error  = sum(1 for _ in p_error.rglob("*.csv"))  if p_error.exists()  else 0

    per_type = Counter()
    if p_error.exists():
        suffix2key = {v["file_end"]: k for k, v in CHECKPOINT_ERRORS.items()}
        for f in p_error.rglob("*.csv"):
            for suffix, key in suffix2key.items():
                if f.name.endswith(suffix):
                    per_type[key] += 1
                    break

    return {"total": total_scored + total_pose + total_error,
            "errors": total_error,
            "per_type": per_type}

def report_final_summary_dual(*,
                              files_found: int,
                              files_processed_session: int,
                              files_scored_session: int,
                              session_per_type: Counter,
                              global_stats: dict) -> str:
    out = []
    out.append("")
    out.append("")
    out.append(_banner_75("SESSION SUMMARY"))
    out.append("")

    labels = ["FILES FOUND", "FILES PROCESSED", "FILES SCORED"]
    L = max(len(s) for s in labels)
    out.append(_kv_line_72("FILES FOUND",      f"{files_found}",              L))
    out.append(_kv_line_72("FILES PROCESSED",  f"{files_processed_session}",  L))
    out.append(_kv_line_72("FILES SCORED",     f"{files_scored_session}",     L))
    out.append("")

    out.append(_errors_table_header(0))

    session_errors = sum(session_per_type.values())
    s_pct = _pct_int(session_errors, files_processed_session)
    g_errors = global_stats.get("errors", 0)
    g_total  = files_found
    g_pct = _pct_int(g_errors, g_total)
    out.append(_errors_total_row(0, f"{session_errors} ({s_pct}%)", f"{g_errors} ({g_pct}%)"))

    global_per = global_stats.get("per_type", Counter())
    for key in _SUMMARY_ORDER:
        label = _label_for_summary(key)
        s_cnt = session_per_type.get(key, 0)
        g_cnt = global_per.get(key, 0)
        out.append(_errors_detail_row(label, 0, s_cnt, g_cnt))

    return "\n".join(out)

# -------------------------
# Pinned scoring display
# -------------------------
class PinnedScoringDisplay:
    """
    Updates in place via IPython display:
      1) "  SCORING  [#####........]"
      2) "            --------------  file i/n  ---  XXs/file  ---  MMmSSs eta"
      3) "  SCORED   ----------------------------------------------  X"
      4) "  ERROR    last: <label> (value)  ------------------------  Y"  (only if Y>0)
    """
    def __init__(self, total_session: int):
        self.total = int(max(0, total_session))
        self.ok = 0
        self.err = 0
        self.last_snippet = "–"
        self._right_start_width = len("SCORING  ")
        self._can_display = False
        try:
            from IPython.display import display, Markdown
            self._Markdown = Markdown
            l1 = self._build_bar_line(done=0)
            l2 = _metrics_line_72_aligned(0, self.total, None, None, self._right_start_width)
            lines = [l1, l2, "", self._kv_line_scored(), self._maybe_line_error()]
            lines = [ln for ln in lines if ln != ""]
            self._handle = display(Markdown("```text\n" + "\n".join(lines) + "\n```"), display_id=True)
            self._can_display = True
        except Exception:
            self._handle = None

    def _build_bar_line(self, done: int) -> str:
        label = f"{INDENT}SCORING  "
        bar_width = CONTENT_WIDTH - len(label)
        inner = max(0, bar_width - 2)
        frac = (done / self.total) if self.total > 0 else 0.0
        filled = int(round(max(0.0, min(1.0, frac)) * inner))
        line = label + "[" + ("#" * filled) + ("." * (inner - filled)) + "]"
        if len(line) < CONTENT_WIDTH:
            line += " " * (CONTENT_WIDTH - len(line))
        elif len(line) > CONTENT_WIDTH:
            line = line[:CONTENT_WIDTH]
        return line

    def _kv_line_scored(self) -> str:
        return _kv_line_72_at_col("SCORED", f"{self.ok}", self._right_start_width)

    def _maybe_line_error(self) -> str:
        if self.err <= 0:
            return ""
        snippet = f"ERROR    last: {self.last_snippet}"
        return _flex_left_value_line_72(snippet, f"{self.err}")

    def _render(self, done, ema_seconds, eta_seconds):
        if not self._can_display or self._handle is None:
            return
        l1 = self._build_bar_line(done)
        l2 = _metrics_line_72_aligned(done, self.total, ema_seconds, eta_seconds, self._right_start_width)
        lines = [l1, l2, "", self._kv_line_scored(), self._maybe_line_error()]
        lines = [ln for ln in lines if ln != ""]
        self._handle.update(self._Markdown("```text\n" + "\n".join(lines) + "\n```"))

    # Accept positional or keyword styles from Main
    def update_success(self, done=None, ema_seconds=None, eta_seconds=None, **kwargs):
        if done is None:
            done = kwargs.get("done_count", kwargs.get("done", 0))
        if ema_seconds is None:
            ema_seconds = kwargs.get("ema_seconds", kwargs.get("ema", None))
        if eta_seconds is None:
            eta_seconds = kwargs.get("eta_seconds", kwargs.get("eta", None))
        self.ok += 1
        self._render(int(done or 0), ema_seconds, eta_seconds)

    def update_error(self, done=None, last_snippet=None, ema_seconds=None, eta_seconds=None, **kwargs):
        if done is None:
            done = kwargs.get("done_count", kwargs.get("done", 0))
        if last_snippet is None:
            last_snippet = (
                kwargs.get("last_label",
                kwargs.get("last_snippet",
                kwargs.get("last",
                kwargs.get("snippet", None))))
            )
        if ema_seconds is None:
            ema_seconds = kwargs.get("ema_seconds", kwargs.get("ema", None))
        if eta_seconds is None:
            eta_seconds = kwargs.get("eta_seconds", kwargs.get("eta", None))
        self.err += 1
        if last_snippet:
            self.last_snippet = last_snippet
        self._render(int(done or 0), ema_seconds, eta_seconds)

    def last_error(self, short_label: str, value_parenthesized: str):
        self.last_snippet = short_label + (f" {value_parenthesized}" if value_parenthesized else "")

    def close(self):
        return
