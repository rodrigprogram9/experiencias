#%%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringColabConfig.py

Purpose
This module stages the behavior-scoring pipeline for Google Colab. It mirrors
the Drive folder tree into /content for fast I/O, selectively copies inputs
that still need work, creates placeholders so skip logic remains valid, and
syncs fresh outputs back to Drive. It can also warm up throughput and run a
quiet background sync during long runs.

Steps
- Load canonical paths from PathConfig and validate Drive inputs.
- Create local mirrors under /content with an identical structure.
- Detect already-processed files and copy only remaining inputs.
- Create placeholders for skipped inputs and existing outputs.
- Rebase a PathConfig-like namespace to the local mirrors.
- Sync new outputs back to Drive and optionally run background sync.

Output
- Data classes for Drive and local paths.
- StageSummary without scoredpose (tracked, pose, scored, error_items).
- Public functions to stage, rebase paths, and sync results.
"""

from __future__ import annotations

#%%% CELL 01 – IMPORTS
"""
Purpose
Import standard libraries and typing helpers required for staging and syncing.

Steps
- Import subprocess, shutil, time, dataclasses, pathlib, types, typing, threading.
"""

import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple, List, Set

# Background sync
import threading

#%%% CELL 02 – DATA CLASSES
"""
Purpose
Provide containers for Drive and local paths plus a concise staging summary.

Steps
- Define DrivePaths and LocalPaths with canonical members.
- Define StageSummary with counts (no scoredpose field).
"""

@dataclass(frozen=True)
class DrivePaths:
    """Canonical locations on Google Drive (from PathConfig)."""
    root: Path
    tracked: Path
    pose: Path
    scored: Path
    scoredpose: Path
    error: Path
    codes: Path


@dataclass(frozen=True)
class LocalPaths:
    """Local mirrors under /content for fast I/O during the run."""
    root: Path
    tracked: Path
    pose: Path
    scored: Path
    scoredpose: Path
    error: Path


@dataclass(frozen=True)
class StageSummary:
    """Counts of files staged into local mirrors (inputs + existing outputs)."""
    tracked: int
    pose: int
    scored: int
    error_items: int

#%%% CELL 03 – PUBLIC API
"""
Purpose
Expose functions to load configs, validate inputs, create local mirrors, and
rebase PathConfig-like objects for use inside Colab.

Steps
- Implement load_configs, validate_inputs, and local_mirrors.
"""

def load_configs(PathConfig) -> Tuple[Path, DrivePaths]:
    """Return (drive_root, drive_paths) using the canonical PathConfig."""
    drive_root = Path(PathConfig.pExperimentalRoot)
    drive_paths = DrivePaths(
        root=drive_root,
        tracked=Path(PathConfig.pTracked),
        pose=Path(PathConfig.pPose),
        scored=Path(PathConfig.pScored),
        scoredpose=Path(PathConfig.pScoredPose),
        error=Path(PathConfig.pScoredError),
        codes=Path(PathConfig.pCodes),
    )
    return drive_root, drive_paths


def _require_csv_folder(folder: Path, message: str) -> None:
    if not folder.exists():
        raise RuntimeError(f"{message}: {folder} (folder not found)")
    if not any(folder.rglob("*.csv")):
        raise RuntimeError(f"{message}: {folder} (no CSV files found)")
        
def validate_inputs(
    drive_paths: DrivePaths, pose_scoring: Optional[bool] = None, *, verbose: bool = False
) -> bool:
    """
    Validate required inputs on Drive and auto-detect pose_scoring when None.
    Returns final pose_scoring (auto-detected if None).
    """
    _require_csv_folder(drive_paths.tracked, "Tracked inputs not found or empty")

    # Auto-detect: pose scoring only if pose folder exists and has CSVs
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    if pose_scoring:
        _require_csv_folder(
            drive_paths.pose,
            "POSE_SCORING=True but Pose inputs not found or empty. "
            "Set POSE_SCORING=False or add Pose CSVs.",
        )

    if verbose:
        print(f"Validation OK. pose_scoring={pose_scoring}")

    return pose_scoring


def local_mirrors(
    drive_root: Path, drive_paths: DrivePaths, local_root: Optional[Path] = None, *, verbose: bool = False
) -> LocalPaths:
    """
    Create local mirrors under /content with an identical substructure.
    """
    # Namespace per experiment under /content/exp_runs/<experiment_name>
    base = local_root or (Path("/content/exp_runs") / drive_root.name)

    def mirror(p: Path) -> Path:
        return base / p.relative_to(drive_root)

    local = LocalPaths(
        root=base,
        tracked=mirror(drive_paths.tracked),
        pose=mirror(drive_paths.pose),
        scored=mirror(drive_paths.scored),
        scoredpose=mirror(drive_paths.scoredpose),
        error=mirror(drive_paths.error),
    )

    # Create all required dirs; idempotent behavior
    for d in (local.tracked, local.pose, local.scored, local.scoredpose, local.error):
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Virtual paths created")
    return local

#%%% CELL 04 – PROCESSED DETECTION & SELECTIVE COPY
"""
Purpose
Provide helpers to list CSVs, detect processed items on Drive, and copy only
the files that still need work.

Steps
- List CSVs in a tree.
- Map tracked→scored names and detect any matching error file.
- Copy a selected list using rsync or a Python fallback.
"""

def _list_csvs(src: Path) -> List[Path]:
    if not src.exists():
        return []
    return [p for p in src.rglob("*.csv") if p.is_file()]


def _scored_name_for(tracked_name: str, pose_scoring: bool) -> str:
    if tracked_name.endswith("tracked.csv"):
        return tracked_name.replace(
            "tracked.csv", "scored_pose.csv" if pose_scoring else "scored.csv"
        )
    return tracked_name.replace(
        ".csv", "_scored_pose.csv" if pose_scoring else "_scored.csv"
    )


def _has_matching_error(tracked_name: str, error_dir: Path) -> bool:
    base = tracked_name.replace("tracked.csv", "")
    if not error_dir.exists():
        return False
    try:
        for f in error_dir.iterdir():
            if f.is_file() and f.name.startswith(base):
                return True  # any *_error.csv matches
    except Exception:
        pass  # ignore transient read errors and continue
    return False


def _is_processed_on_drive(tracked_rel: str, drive_paths: DrivePaths, pose_scoring: bool) -> bool:
    scored_name = _scored_name_for(tracked_rel, pose_scoring)
    scored_hit = (drive_paths.scoredpose if pose_scoring else drive_paths.scored) / scored_name
    if scored_hit.exists():
        return True
    if _has_matching_error(tracked_rel, drive_paths.error):
        return True
    return False


def _copy_selected(src: Path, dst: Path, rel_files: List[str]) -> int:
    """Copy only the files listed in rel_files; return count successfully copied."""
    if not rel_files:
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    # Prefer rsync when available; it handles trees efficiently
    if shutil.which("rsync"):
        files_list = dst / ".rsync_files.txt"
        try:
            files_list.write_text("\n".join(rel_files) + "\n")
            cmd = [
                "rsync", "-a", "--no-compress", "--prune-empty-dirs",
                "--files-from", str(files_list), str(src) + "/", str(dst) + "/"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or "rsync failed")  # bubble up
        except Exception:
            # Fallback: copy one-by-one; robust but slower
            for rel in rel_files:
                s = src / rel
                d = dst / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                if s.exists():
                    shutil.copy2(s, d)
        finally:
            try:
                files_list.unlink(missing_ok=True)
            except Exception:
                pass
        return len(rel_files)

    # Pure-Python fallback if rsync is not present
    copied = 0
    for rel in rel_files:
        s = src / rel
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.copy2(s, d)
            copied += 1
    return copied


#%%% CELL 04A – FORMAT CONSTANTS & HELPERS
"""
Purpose
Provide shared formatting helpers for consistent console output:
- Banners at 75 chars
- Content rows at 72 chars with 2-space indent
- Dash-fill that starts 2 spaces after the longest label in the block and
  ends 2 spaces before the value
- Lettered duration formatting (MMmSSs or HHhMMm)
- Center truncation for long values
"""

# Shared constants & rules (for implementation)
BANNER_WIDTH = 75
CONTENT_WIDTH = 72
INDENT = "  "         # two spaces
VALUE_SEP = "  "      # two spaces before/after value groups & '---'

# Progress bar constants (not used in this cell, here for global consistency)
BAR_TOTAL = 28        # includes brackets
BAR_LEFT = "["
BAR_RIGHT = "]"
BAR_FILL = "#"
BAR_EMPTY = "."

def _banner(title: str) -> str:
    t = title.strip().upper()
    pad = max(BANNER_WIDTH - len(t) - 2, 0)
    left = pad // 2
    right = pad - left
    return "=" * left + " " + t + " " + "=" * right

def _truncate_left(s: str, max_len: int) -> str:
    """Truncate string from the left, keeping the rightmost characters."""
    s = str(s)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[-max_len:]
    return "..." + s[-(max_len - 3):]

def _fmt_duration_lettered(seconds: float) -> str:
    """
    Return lettered duration:
      - < 1h  -> 'MMmSSs'
      - >=1h  -> 'HHhMMm'
    Seconds are rounded to nearest second; minutes are rounded from the seconds.
    """
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        # Round minutes from seconds (>=30s -> +1m)
        if sec >= 30:
            m = (m + 1) % 60
            if m == 0:
                h += 1
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{sec:02d}s"

def _kv_line(label: str, value: str, longest_label: int) -> str:
    """
    Build a single content row (72 chars total):
      '␣␣' + label + gap + dashes + '␣␣' + value
    where
      gap = (longest_label - len(label)) + 2  # 2 spaces after longest label
    and the dash-fill ends 2 spaces before the value.
    If value overflows, it's center-truncated to fit exactly 72 cols.
    """
    label = str(label)
    value = str(value)

    # Left part: indent + label + gap to align dash start
    left = INDENT + label
    gap = (max(longest_label - len(label), 0) + 2)
    left += " " * gap

    # Compute max space for value (2 spaces reserved before value)
    max_value_len = CONTENT_WIDTH - len(left) - 2
    v = _truncate_left(value, max_value_len)

    # Dash-fill length so the final line ends exactly at CONTENT_WIDTH
    dash_len = CONTENT_WIDTH - len(left) - 2 - len(v)
    if dash_len < 0:
        # (Shouldn't happen because we truncated v, but guard anyway)
        v = _truncate_left(v, max_value_len + dash_len)
        dash_len = max(CONTENT_WIDTH - len(left) - 2 - len(v), 0)

    return left + ("-" * dash_len) + "  " + v



#%%% CELL 05 – PLACEHOLDERS & WARMUP
"""
Purpose
Create local placeholders so skip logic works without copying all outputs and
measure Drive→/content throughput to estimate staging time.

Steps
- Touch zero-byte placeholders for existing outputs on Drive.
- Read a short sample of real inputs to estimate MB/s.
"""

def _mirror_placeholders(src: Path, dst: Path, patterns: Optional[Iterable[str]]) -> int:
    """Create zero-byte placeholders in dst for files that exist in src."""
    if not src.exists():
        return 0

    if patterns is None:
        files = [p for p in src.rglob("*") if p.is_file()]
    else:
        files = []
        for pat in patterns:
            files.extend(src.rglob(pat))
        files = [p for p in files if p.is_file()]

    n = 0
    for f in files:
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.touch()  # zero-byte placeholder
            n += 1
    return n


def _warmup_measure_speed_mbps(
    drive_paths: DrivePaths,
    to_copy_tracked: List[str],
    to_copy_pose: List[str],
    *,
    min_seconds: float = 5.0,
    min_megabytes: float = 100.0,
    max_seconds: float = 10.0,
    chunk_size: int = 8 * 1024 * 1024,  # 8 MB
) -> float | None:
    """
    Measure Drive→/content throughput by reading the actual to-copy files for a
    short warm-up. Data is discarded; no writes occur. Returns MB/s or None.
    """
    import itertools

    # Build absolute paths on Drive for the to-copy inputs
    candidates: List[Path] = []
    for rel in to_copy_tracked:
        p = drive_paths.tracked / rel
        if p.exists() and p.is_file():
            candidates.append(p)
    for rel in to_copy_pose:
        p = drive_paths.pose / rel
        if p.exists() and p.is_file():
            candidates.append(p)

    if not candidates:
        return None

    target_bytes = int(min_megabytes * 1024 * 1024)
    start = time.perf_counter()
    read_bytes = 0

    # Cycle files until thresholds or the max time is reached
    for file_path in itertools.cycle(candidates):
        if time.perf_counter() - start >= max_seconds:
            break
        try:
            with open(file_path, "rb") as f:
                while True:
                    if (time.perf_counter() - start) >= max_seconds:
                        break
                    if ((time.perf_counter() - start) >= min_seconds and
                            read_bytes >= target_bytes):
                        break
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break  # end of file
                    read_bytes += len(chunk)
        except Exception:
            continue  # skip unreadable files and continue

        # Early stop once we hit both thresholds
        if ((time.perf_counter() - start) >= min_seconds and
                read_bytes >= target_bytes):
            break

    elapsed = max(time.perf_counter() - start, 1e-6)
    if read_bytes <= 0:
        return None
    return (read_bytes / (1024 * 1024)) / elapsed

#%%% CELL 06 – STAGING (DISABLED: RUNS DIRECTLY ON DRIVE)
"""
Purpose
Disable staging entirely. The pipeline now reads inputs directly from Drive.

Output
Return a StageSummary of zeros so calls won't break if this is invoked.
"""

from dataclasses import dataclass
from typing import Optional

def stage_to_local(
    drive_paths: DrivePaths,
    local_paths: LocalPaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = False,
    prior_speed_mbps: Optional[float] = None,
) -> StageSummary:
    """No-op: staging removed. Reads/writes happen directly on Drive."""
    return StageSummary(tracked=0, pose=0, scored=0, error_items=0)


#%%% CELL 07B – MIXED PATHCONFIG (INPUTS ON DRIVE, OUTPUTS LOCAL)
"""
Purpose
Build a PathConfig-like namespace that reads inputs from Drive but writes
outputs to /content local mirrors (for batched sync).
"""

from types import SimpleNamespace
from pathlib import Path

def make_mixed_pathconfig(PathConfig, drive_paths: DrivePaths, local_paths: LocalPaths):
    base = dict(vars(PathConfig))

    def P(x):
        return x if isinstance(x, Path) else Path(x)

    # Inputs from Drive
    base["pExperimentalRoot"] = P(drive_paths.root)
    base["pTracked"]          = P(drive_paths.tracked)
    base["pPose"]             = P(drive_paths.pose)

    # Outputs to local
    base["pScored"]       = P(local_paths.scored)
    base["pScoredPose"]   = P(local_paths.scoredpose)
    base["pScoredError"]  = P(local_paths.error)

    # Codes stay on Drive
    if hasattr(PathConfig, "pCodes"):
        base["pCodes"] = getattr(PathConfig, "pCodes")

    # Convenience globs
    try:
        p_tracked     = P(base["pTracked"])
        p_pose        = P(base["pPose"])
        p_scored      = P(base["pScored"])
        p_scored_pose = P(base["pScoredPose"])
        p_error       = P(base["pScoredError"])
    except KeyError:
        pass
    else:
        base["gTracked"]    = p_tracked.rglob("*tracked.csv")
        base["gPose"]       = p_pose.rglob("*pose.csv")
        base["gScored"]     = p_scored.rglob("*.csv")
        base["gScoredPose"] = p_scored_pose.rglob("*.csv")
        base["gError"]      = p_error.rglob("*.csv")

    return SimpleNamespace(**base)


#%%% CELL 08 – FINISH PRINTER (NO SYNC)
"""
Purpose
Print the final 75/72-formatted endcap without subrows or syncing.
Matches your requested output format exactly.
"""
# Keep the done duck. It celebrates the end of a run.
def done_duck(i=24): return f"""\n\n\n{' '*(i+9)}__(·)<    ,\n{' '*(i+6)}O  \\_) )   c|_|\n{' '*i}{'~'*27}"""

def print_finish(dest_path: str, scoring_seconds: float) -> None:
    print(done_duck())
    print(_banner("SCORING AND SAVING COMPLETE"))
    print()

    L_saved = len("SAVED IN DRIVE")
    print(_kv_line("SAVED IN DRIVE", str(dest_path), L_saved))
    print()

    session_str = _fmt_duration_lettered(max(0.0, float(scoring_seconds or 0.0)))
    print(_kv_line("SESSION TIME", session_str, L_saved))
    print()
    print("=" * 75)


#%%% CELL 09 – BACKGROUND SYNC
"""
Purpose
Keep batch saving capability.
"""

import threading
import time
import shutil

_sync_thread = None
_stop_sync = False

def start_background_sync(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    batch_size: int = 30
) -> None:
    """Start a background thread that syncs new scored files in batches."""
    global _sync_thread, _stop_sync
    _stop_sync = False

    def _sync_loop():
        last_count = 0
        while not _stop_sync:
            time.sleep(5)
            scored_files = list(local_paths.scored.rglob("*.csv"))
            if len(scored_files) - last_count >= batch_size:
                sync_outputs_back(local_paths, drive_paths, pose_scoring, verbose=False)
                last_count = len(scored_files)

    _sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    _sync_thread.start()

def stop_background_sync() -> None:
    """Stop the background sync thread."""
    global _stop_sync, _sync_thread
    _stop_sync = True
    if _sync_thread is not None:
        _sync_thread.join()
        _sync_thread = None


#%%% CELL 10 – SYNC OUTPUTS BACK TO DRIVE
"""
Purpose
Push any remaining local outputs to Drive at the end of a run.
"""

        
def sync_outputs_back(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    verbose: bool = True
) -> None:
    """Sync local scored files back to Drive."""
    if verbose:
        print(_banner("SAVING OUTPUTS BACK TO DRIVE"))
        print()

    # Sync scored
    if local_paths.scored.exists():
        _rsync(local_paths.scored, drive_paths.scored)
    if pose_scoring and local_paths.scoredpose.exists():
        _rsync(local_paths.scoredpose, drive_paths.scoredpose)
    if local_paths.error.exists():
        _rsync(local_paths.error, drive_paths.error)

def _rsync(src: Path, dst: Path):
    """Helper to copy directory tree."""
    dst.mkdir(parents=True, exist_ok=True)
    for file in src.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(src)
            dst_path = dst / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst_path)


#%%% CELL 11 – EXPORTS
"""
Purpose
Expose public names needed for no-staging + batched-saving workflow.
"""

__all__ = [
    "DrivePaths",
    "LocalPaths",
    "StageSummary",
    "load_configs",
    "validate_inputs",
    "local_mirrors",
    "make_mixed_pathconfig",
    "sync_outputs_back",
    "start_background_sync",
    "stop_background_sync",
    "_banner",
    "_kv_line",
    "_fmt_duration_lettered",
    "done_duck",
    "print_finish",
]
