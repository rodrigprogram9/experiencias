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
Compute speed and orientation and select a per-frame best view for pose use.

Steps
- Calculate speed in mm/s as floats (rounding done at the call site).
- Determine view from confidences or use vertical fallback logic.
- Compute orientation in degrees with 0 at North.
"""

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


def determine_view(row):
    """
    Choose a view per frame using confidences. When all three body parts are
    present, pick the view (Left/Right/Top) with highest confidence. For
    vertical cases, prefer Head if present, else Abdomen; otherwise return NaN.
    """
    # full key positions available → pick by confidence
    if (pd.notna(row.get("Head.Position.X")) and
        pd.notna(row.get("Thorax.Position.X")) and
        pd.notna(row.get("Abdomen.Position.X"))):
        confidences = {
            "Left": row.get("Left.Confidence", 0),
            "Right": row.get("Right.Confidence", 0),
            "Top": row.get("Top.Confidence", 0)
        }
        selected = max(confidences, key=confidences.get)
        vx = row.get(f"{selected}.Position.X", np.nan)
        vy = row.get(f"{selected}.Position.Y", np.nan)
        return selected, vx, vy

    # vertical fallback using head or abdomen
    if pd.notna(row.get("Head.Position.X")) or pd.notna(row.get("Abdomen.Position.X")):
        if pd.notna(row.get("Head.Position.X")):       # prefer head when present
            return "Vertical", row.get("Head.Position.X", np.nan), row.get("Head.Position.Y", np.nan)
        return "Vertical", row.get("Abdomen.Position.X", np.nan), row.get("Abdomen.Position.Y", np.nan)

    # no valid coordinates
    return np.nan, np.nan, np.nan


def calculate_orientation(pointA_x, pointA_y, pointB_x, pointB_y):
    """
    Compute orientation from A→B in degrees, normalized to [0, 360) with 0 = North.
    """
    dx = pointB_x - pointA_x
    dy = pointA_y - pointB_y  # invert y to set 0 = North
    angle = np.arctan2(dy, dx)
    deg = np.degrees(angle)
    deg = (deg + 360) % 360
    deg = (deg + 90) % 360  # shift so 0 corresponds to North
    return np.round(deg, 2)

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
Provide helpers to format the run header with left/right alignment, progress
lines (no filenames), error filename lines, and a dual SESSION|GLOBAL summary.
Also scan the current experiment to compute global totals and per-error stats.

Steps
- report_header(...) → pre-run summary with aligned label/value columns.
- report_scoring_line(...) → "SCORING: file i/N (S.s/file – HHhMM eta)".
- report_error_line(...) / report_error_filename(...).
- scan_global_stats(...) → totals and per-error counts on disk.
- report_final_summary_dual(...) → compact labels + aligned table.
"""

from pathlib import Path
from collections import Counter

def compute_timing_metrics(prev_durations, remaining_files: int) -> tuple[float, str]:
    """
    Compute rolling avg seconds/file and ETA from a deque of completed file
    durations. The deque should contain the durations of the most recent
    completed files (e.g., maxlen=5). ETA is based on this average and the
    number of remaining files in THIS session.

    returns
    - (avg_seconds_per_file, eta_str as 'HHhMM')
    """
    if len(prev_durations) == 0:
        avg_s = 0.0
    else:
        avg_s = sum(prev_durations) / len(prev_durations)

    eta_seconds = max(0.0, avg_s * max(0, remaining_files))
    hh = int(eta_seconds // 3600)
    mm = int((eta_seconds % 3600) // 60)
    eta_str = f"{hh:02d}h{mm:02d}"
    return avg_s, eta_str


# Layout controls
_SPACER = "   "     # 3-space spacer between label/bar/value
_BASE_BAR = 50      # baseline dash-bar length for header rows

def _end_col(pad_to: int, max_val_width: int, base_bar: int = _BASE_BAR) -> int:
    """
    Compute the absolute column index where VALUES should END, so that all
    header values are right-aligned. Columns are measured as visible chars.
    """
    # label.ljust(pad_to) + spacer + bar + spacer + value
    # We want: end_of_value = (pad_to + len(spacer) + base_bar + len(spacer) + max_val_width)
    return pad_to + len(_SPACER) + base_bar + len(_SPACER) + max_val_width

def _kv_line_aligned(label: str, value: str, pad_to: int, end_column: int) -> str:
    """
    Return an aligned header line:
      label.ljust(pad_to) + spacer + dashes + spacer + value
    The dash count is chosen so the value's right edge lands at end_column.
    """
    left = label.ljust(pad_to)
    bar_start = pad_to + len(_SPACER)         # where dashes begin
    value_end = end_column                    # fixed right edge for value
    value_start = max(bar_start, value_end - len(value))
    dash_len = max(0, value_start - bar_start)
    return f"{left}{_SPACER}{'-' * dash_len}{_SPACER}{value}"

def report_header(experimental_root,
                  pose_scoring: bool,
                  total_found: int,
                  to_score: int,
                  skipped: int,
                  already_scored: int,
                  already_errors: int) -> str:
    """
    Return a standardized multi-line header. Left labels align to a common
    width; right values align to a fixed column. The final dash-only line
    uses a blank label and the same right edge for perfect alignment.
    """
    root = str(experimental_root)
    pose_flag = "TRUE" if pose_scoring else "FALSE"

    # Determine label pad and right edge from the numeric rows
    labels = ["FILES FOUND", "TO SCORE", "SKIPPING"]
    pad_to = max(len(s) for s in labels)      # left label column width
    max_width = max(len(str(total_found)), len(str(to_score)), len(str(skipped)))
    endcol = _end_col(pad_to, max_width, _BASE_BAR)

    lines = []
    lines.append(f"PROCESSING: {root}")
    lines.append(f"POSE:    {pose_flag}\n")

    # Aligned header rows (values right-aligned)
    lines.append(_kv_line_aligned("FILES FOUND", str(total_found), pad_to, endcol))
    lines.append(_kv_line_aligned("TO SCORE",    str(to_score),    pad_to, endcol))
    lines.append(_kv_line_aligned("SKIPPING",    str(skipped),     pad_to, endcol))

    # Final dash-only row (blank label), same right alignment
    summary_value = f"scored: {already_scored}   ---   errors: {already_errors}"
    lines.append(_kv_line_aligned(" ", summary_value, pad_to, endcol) + "\n")
    return "\n".join(lines)

def report_scoring_line(idx: int,
                        total_session: int,
                        sec_per_file: float,
                        eta_str: str) -> str:
    """Return a single scoring progress line (no filename)."""
    return (f"SCORING: file {idx}/{total_session} "
            f"({sec_per_file:.2f} s/file – {eta_str} eta)")

def report_error_line(err_text: str) -> str:
    """Return the error line with tree prefix."""
    return f"├────  {err_text}"

def report_error_filename(basename: str) -> str:
    """Return the filename line with tree prefix."""
    return f"└────  {basename}"

def _error_suffix_map() -> dict[str, str]:
    """Map file_end → error_key for fast classification."""
    return {v["file_end"]: k for k, v in CHECKPOINT_ERRORS.items()}

def scan_global_stats(PATHconfig) -> dict:
    """
    Scan the current experiment folders to compute global totals and per-error
    counts. Scope is the current experiment (PathConfig.pExperimentalRoot).

    returns
    - {"total": int, "errors": int, "per_type": Counter({key: count, ...})}
    """
    p_scored = Path(PATHconfig.pScored)
    p_pose   = Path(PATHconfig.pScoredPose)
    p_error  = Path(PATHconfig.pScoredError)

    total_scored = sum(1 for _ in p_scored.rglob("*.csv")) if p_scored.exists() else 0
    total_pose   = sum(1 for _ in p_pose.rglob("*.csv"))   if p_pose.exists()   else 0
    total_error  = sum(1 for _ in p_error.rglob("*.csv"))  if p_error.exists()  else 0

    per_type = Counter()
    if p_error.exists():
        suffix2key = _error_suffix_map()
        for f in p_error.rglob("*.csv"):
            for suffix, key in suffix2key.items():
                if f.name.endswith(suffix):
                    per_type[key] += 1
                    break

    return {"total": total_scored + total_pose + total_error,
            "errors": total_error,
            "per_type": per_type}

def _pct_int(numer: int, denom: int) -> int:
    """Return integer percent with safe denom."""
    if denom <= 0:
        return 0
    return int(round(100 * numer / denom))

# Compact labels for the summary table
_SUMMARY_LABELS = {
    "ERROR_READING_FILE":     "error reading file",
    "WRONG_STIMULUS_COUNT":   "wrong stim count",
    "WRONG_STIMULUS_DURATION":"wrong stim duration",
    "LOST_CENTROID_POSITION": "centroid nans",
    "POSE_MISMATCH":          "pose mismatch",
    "MISSING_POSE_FILE":      "missing pose file",
    "VIEW_NAN_EXCEEDED":      "view nans",
    "UNASSIGNED_BEHAVIOR":    "unassigned frames",
    "NO_EXPLORATION":         "low exploration",
    "OUTPUT_LEN_SHORT":       "short aligned length",
}

def _label_for_summary(error_key: str) -> str:
    """Return compact label for summary listing."""
    return _SUMMARY_LABELS.get(error_key,
                               CHECKPOINT_ERRORS[error_key]["message"].lower())

def report_final_summary_dual(session_total: int,
                              session_per_type: Counter,
                              global_stats: dict,
                              elapsed_hhmm: str | None = None,
                              files_found: int | None = None,
                              files_scored_session: int | None = None) -> str:
    """
    Build a dual summary table with SESSION and GLOBAL columns, preceded by
    optional kv lines for elapsed time and file counts. Rows are aligned so
    the '|' columns match, using a fixed left bar width.

    returns
    - multi-line string ready to print
    """
    out = []

    # Optional preface: time and file counts (use the same alignment routine)
    if elapsed_hhmm is not None:
        pad_to = len("TIME SCORING")
        endcol = _end_col(pad_to, max_val_width=len(elapsed_hhmm))
        out.append(_kv_line_aligned("\n\nTIME SCORING", elapsed_hhmm, pad_to, endcol))
        out.append("")

    if (files_found is not None) and (files_scored_session is not None):
        labels = ["FILES FOUND", "FILES SCORED"]
        pad_to = max(len(s) for s in labels)
        max_width = max(len(str(files_found)), len(str(files_scored_session)))
        endcol = _end_col(pad_to, max_width)
        out.append(_kv_line_aligned("FILES FOUND",  str(files_found),         pad_to, endcol))
        out.append(_kv_line_aligned("FILES SCORED", str(files_scored_session), pad_to, endcol))

    # Header and totals
    out.append("---------------------------------|  SESSION  |---|  GLOBAL  |")

    session_errors = sum(session_per_type.values())
    global_total   = global_stats["total"]
    global_errors  = global_stats["errors"]
    global_per     = global_stats["per_type"]

    s_pct = _pct_int(session_errors, session_total)
    g_pct = _pct_int(global_errors, global_total)

    out.append(f"ERRORS   ------------------------|  {session_errors} ({s_pct}%)  "
               f"|---|  {global_errors} ({g_pct}%)  |")

    # Fixed left bar width so the '|' column aligns with the header
    LEFT_BAR_WIDTH = len("ERRORS   ------------------------")  # = 33

    def _row(label: str, s_cnt: int, g_cnt: int) -> str:
        stub = f"---   {label}   "                   # keep this spacing
        dash = "-" * max(0, LEFT_BAR_WIDTH - len(stub))
        # numbers right-aligned to width=5 with 2-space margins
        return f"{stub}{dash}|  {s_cnt:>5}  |---|  {g_cnt:>5}  |"

    # Sorted by session count desc, then label
    all_keys = set(session_per_type) | set(global_per)
    for key in sorted(all_keys, key=lambda k: (-session_per_type.get(k, 0),
                                               _label_for_summary(k))):
        label = _label_for_summary(key)
        s_cnt = session_per_type.get(key, 0)
        g_cnt = global_per.get(key, 0)
        out.append(_row(label, s_cnt, g_cnt))

    return "\n".join(out)





#Keep the done duck. It celebrates the end of a run. Whitespace art is sacred.
def done_duck(i=15):return f"""\n\n\n{' '*(i+9)}__(·)<    ,\n{' '*(i+6)}O  \\_) )   c|_|\n{' '*i}{'~'*27}"""
