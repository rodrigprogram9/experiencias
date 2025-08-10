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

#%%% CELL 06 – STAGING (ETA, PLACEHOLDERS, INPUT COPY)
"""
Purpose
Stage inputs into local mirrors, print an ETA based on a brief warm-up, create
placeholders for existing outputs, and return a concise summary.

Steps
- Detect which inputs still need copying and estimate time.
- Copy only pending inputs; touch placeholders for skipped items.
- Mirror placeholders for existing outputs (Scored, Error).
- Return StageSummary (no scoredpose entry).
"""

def stage_to_local(
    drive_paths: DrivePaths,
    local_paths: LocalPaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = True
) -> StageSummary:
    """Stage inputs selectively and mirror existing outputs as placeholders."""
    # Auto-detect pose_scoring if not given
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    print("================= LOADING FILES FROM DRIVE =================")

    # Gather all candidate inputs
    tracked_files = _list_csvs(drive_paths.tracked)
    total_tracked = len(tracked_files)

    pose_files = _list_csvs(drive_paths.pose) if pose_scoring else []
    total_pose = len(pose_files) if pose_scoring else 0

    # Build worklist for tracked files: only unprocessed inputs
    rel_tracked = [str(p.relative_to(drive_paths.tracked)) for p in tracked_files]
    to_copy_tracked: List[str] = []
    skipped_already_processed: List[str] = []

    for rel in rel_tracked:
        if _is_processed_on_drive(rel, drive_paths, pose_scoring):
            skipped_already_processed.append(rel)  # will touch placeholders
        else:
            to_copy_tracked.append(rel)

    # Pose files to copy only for tracked files we will process
    to_copy_pose: List[str] = []
    if pose_scoring and drive_paths.pose.exists():
        for rel in to_copy_tracked:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            if (drive_paths.pose / pose_rel).exists():
                to_copy_pose.append(pose_rel)

    # Estimate ETA for the files we will actually copy
    est_files_to_copy = len(to_copy_tracked) + len(to_copy_pose)
    total_input_files = total_tracked + total_pose

    total_bytes = 0
    for rel in to_copy_tracked:
        f = drive_paths.tracked / rel
        try:
            total_bytes += f.stat().st_size
        except OSError:
            pass  # some files may be transient on the mount
    for rel in to_copy_pose:
        f = drive_paths.pose / rel
        try:
            total_bytes += f.stat().st_size
        except OSError:
            pass

    # Adaptive warm-up over the actual to-copy list: 5–10 s and/or ≥100 MB
    speed_mbps = _warmup_measure_speed_mbps(
        drive_paths,
        to_copy_tracked=to_copy_tracked,
        to_copy_pose=to_copy_pose,
        min_seconds=5.0,
        min_megabytes=100.0,
        max_seconds=10.0,
    ) or 12.0  # conservative fallback if measurement failed

    if est_files_to_copy > 0 and total_bytes > 0:
        est_seconds = (total_bytes / (1024 * 1024)) / max(speed_mbps, 0.1)
        print(
            f"\n   Estimated: ~{_fmt_seconds(est_seconds)} at {speed_mbps:.1f} MB/s "
            f"for {est_files_to_copy}/{total_input_files} files"
        )
    else:
        print("\n   Estimated: No new input files to copy")

    # Perform staging
    t0 = time.perf_counter()

    # Inputs (copy only unprocessed)
    n_tracked = _copy_selected(drive_paths.tracked, local_paths.tracked, to_copy_tracked)
    n_pose = 0
    if pose_scoring and to_copy_pose:
        n_pose = _copy_selected(drive_paths.pose, local_paths.pose, to_copy_pose)

    # Inputs (placeholders for already-processed files)
    for rel in skipped_already_processed:
        p = local_paths.tracked / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()  # zero-byte placeholder

    # Pose placeholders for skipped tracked items (if a pose file exists)
    if pose_scoring and drive_paths.pose.exists():
        for rel in skipped_already_processed:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            pose_src = drive_paths.pose / pose_rel
            if pose_src.exists():
                q = local_paths.pose / pose_rel
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    q.touch()

    # Existing outputs (placeholders only)
    n_scored = _mirror_placeholders(drive_paths.scored, local_paths.scored, patterns=("*.csv",))
    if pose_scoring:
        _ = _mirror_placeholders(drive_paths.scoredpose, local_paths.scoredpose, patterns=("*.csv",))
    n_error = _mirror_placeholders(drive_paths.error, local_paths.error, patterns=None)

    if verbose:
        t1 = time.perf_counter()
        print(f"\n   Inputs        : Tracked={n_tracked}"
              f"{', Pose=' + str(n_pose) if pose_scoring else ''}")
        print(f"   Existing outs : Scored={n_scored}, ScoredError={n_error}")
        print(f"   Staging time  : {_fmt_seconds(t1 - t0)}")
        print("\n======================= READY TO RUN =======================")

    return StageSummary(tracked=n_tracked, pose=n_pose, scored=n_scored, error_items=n_error)

#%%% CELL 07 – LOCAL PATHCONFIG REBASE
"""
Purpose
Return a PathConfig-like object rebased under the local root using Path
objects (not strings) and inject convenience globs computed from those Paths.
This lets the pipeline run unchanged while targeting local mirrors. Globs are
generators; iterate them after rebasing.

Steps
- Convert Drive-based paths to local Path objects.
- Preserve non-path attributes and callables.
- Inject gTracked/gPose/gScored/gScoredPose/gError using .rglob().
"""

def make_local_pathconfig(PathConfig, local_paths: LocalPaths):
    """
    Rebase every Drive path under the local root and return a namespace with
    Path objects plus fresh convenience globs. Globs are generators; iterate
    them after calling this function.
    """
    drive_root = Path(PathConfig.pExperimentalRoot)
    local_root = Path(local_paths.root)

    rebased = {}

    for name, value in vars(PathConfig).items():
        if name.startswith("__"):
            continue  # skip dunder attributes
        if callable(value):
            rebased[name] = value
            continue
        try:
            p = Path(value)
            rel = p.relative_to(drive_root)
            rebased[name] = local_root / rel  # keep as Path, not str
        except Exception:
            rebased[name] = value  # keep non-path values verbatim

    # Keep Codes path pointing to Drive for clarity in notebooks
    if hasattr(PathConfig, "pCodes"):
        rebased["pCodes"] = PathConfig.pCodes

    # Inject convenience globs rebased to local mirrors
    try:
        p_tracked = (rebased["pTracked"] if isinstance(rebased["pTracked"], Path)
                     else Path(rebased["pTracked"]))
        p_pose = (rebased["pPose"] if isinstance(rebased["pPose"], Path)
                  else Path(rebased["pPose"]))
        p_scored = (rebased["pScored"] if isinstance(rebased["pScored"], Path)
                    else Path(rebased["pScored"]))
        p_scoredpose = (rebased["pScoredPose"] if isinstance(rebased["pScoredPose"], Path)
                        else Path(rebased["pScoredPose"]))
        p_error = (rebased["pScoredError"] if isinstance(rebased["pScoredError"], Path)
                   else Path(rebased["pScoredError"]))
    except KeyError:
        # if any required path is missing, skip globs silently
        pass
    else:
        # note: these are generators; they are evaluated when iterated
        rebased["gTracked"] = p_tracked.rglob("*tracked.csv")
        rebased["gPose"] = p_pose.rglob("*pose.csv")
        rebased["gScored"] = p_scored.rglob("*.csv")
        rebased["gScoredPose"] = p_scoredpose.rglob("*.csv")
        rebased["gError"] = p_error.rglob("*.csv")

    return SimpleNamespace(**rebased)


#%%% CELL 08 – SYNC OUTPUTS BACK TO DRIVE
"""
Purpose
Copy outputs from local → Drive in bulk, skipping placeholders and avoiding
overwrites when running in upload mode.

Steps
- Sync Scored and Error outputs; include ScoredPose if enabled.
- Skip *.tmp and zero-byte placeholders.
"""

def sync_outputs_back(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = False
) -> None:
    """
    Copy outputs from local → Drive in bulk (resilient, skip placeholders).
    """
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    t0 = time.perf_counter()

    _copy_tree(local_paths.scored, drive_paths.scored, patterns=("*.csv",), upload_mode=True)
    _copy_tree(local_paths.error, drive_paths.error, patterns=None, upload_mode=True)
    if pose_scoring:
        _copy_tree(local_paths.scoredpose, drive_paths.scoredpose, patterns=("*.csv",), upload_mode=True)

    if verbose:
        t1 = time.perf_counter()
        print("\n\n================ SCORING AND SAVING COMPLETE ================")
        print("\n              Synced outputs back to Drive.")
        print(f"                 Sync time     : {_fmt_seconds(t1 - t0)}")

#%%% CELL 09 – BACKGROUND SYNC (SILENT, FINAL FILES ONLY)
"""
Purpose
Provide a silent background thread that syncs after exactly N new final files
appear. This reduces manual sync invocations during long runs.

Steps
- Track seen outputs and compare with the current set every few seconds.
- Sync after an exact batch of new files is detected.
- Allow explicit start/stop around a run.
"""

_bg_state = {
    "thread": None,
    "stop": threading.Event(),
    "seen": set(),        # set[str] of files already accounted for
    "batch_size": 30,
}


def _final_csvs_in_dir(p: Path) -> Set[str]:
    if not p.exists():
        return set()
    return {str(f.resolve()) for f in p.rglob("*.csv") if not f.name.endswith(".tmp")}


def _final_outputs_set(local_paths: LocalPaths, pose_scoring: bool) -> Set[str]:
    s = _final_csvs_in_dir(local_paths.scored) | _final_csvs_in_dir(local_paths.error)
    if pose_scoring:
        s |= _final_csvs_in_dir(local_paths.scoredpose)
    return s


def start_background_sync(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    batch_size: int = 30
) -> None:
    """Start a silent background sync that triggers after EXACTLY batch_size new files."""
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    if _bg_state["thread"] and _bg_state["thread"].is_alive():
        return  # already running

    _bg_state["stop"].clear()
    _bg_state["batch_size"] = int(batch_size)
    _bg_state["seen"] = _final_outputs_set(local_paths, pose_scoring)

    def _worker():
        while not _bg_state["stop"].is_set():
            time.sleep(5)
            try:
                current = _final_outputs_set(local_paths, pose_scoring)
                new_files = current - _bg_state["seen"]
                if len(new_files) >= _bg_state["batch_size"]:
                    try:
                        sync_outputs_back(local_paths, drive_paths, pose_scoring, verbose=False)
                    finally:
                        _bg_state["seen"] = current
            except Exception:
                pass  # stay quiet and keep trying

    t = threading.Thread(target=_worker, daemon=True)
    _bg_state["thread"] = t
    t.start()


def stop_background_sync() -> None:
    """Stop the background sync thread."""
    if _bg_state["thread"]:
        _bg_state["stop"].set()
        _bg_state["thread"].join(timeout=15)
        _bg_state["thread"] = None

#%%% CELL 10 – INTERNAL HELPERS (RESILIENT COPY)
"""
Purpose
Implement resilient copying using rsync when available and a Python fallback
when needed. Handle Drive mount drops with a remount attempt.

Steps
- Validate CSV folders, copy trees with patterns, and remount on failure.
- Provide rsync wrapper and simple formatting helpers.
"""

def _require_csv_folder(folder: Path, message: str) -> None:
    if not folder.exists():
        raise RuntimeError(f"{message}: {folder} (folder not found)")
    if not any(folder.rglob("*.csv")):
        raise RuntimeError(f"{message}: {folder} (no CSV files found)")


def _copy_tree(
    src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False
) -> int:
    """
    Copy files from src to dst; return count of files attempted to copy.

    - Prefer rsync. If the mount drops, remount once and retry.
    - Upload mode: do not overwrite Drive files; skip zero-byte placeholders.
    """
    if not src.exists():
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    # Build candidate file list
    if patterns is None:
        candidates = [p for p in src.rglob("*") if p.is_file()]
    else:
        candidates = []
        for pat in patterns:
            candidates.extend(src.rglob(pat))
        candidates = [p for p in candidates if p.is_file()]

    file_count = len(candidates)
    if file_count == 0:
        return 0

    # Prefer rsync
    if shutil.which("rsync"):
        try:
            _rsync_copy(src, dst, patterns, upload_mode=upload_mode)
            return file_count
        except _MountDropError:
            if _remount_drive():
                _rsync_copy(src, dst, patterns, upload_mode=upload_mode)
                return file_count
            raise RuntimeError(f"Drive remount failed while copying: {src} → {dst}")
        except Exception:
            pass  # fall through to Python copy

    # Python fallback with the same rules as rsync mode
    did_remount = False
    for path in candidates:
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if upload_mode:
                try:
                    if path.stat().st_size == 0:
                        continue  # skip placeholders
                except OSError:
                    continue
                if out.exists():
                    continue  # do not overwrite Drive files
            shutil.copy2(path, out)
        except OSError as oe:
            # 107 is a typical "transport endpoint" error on /content/drive
            if not did_remount and getattr(oe, "errno", None) == 107:
                did_remount = _remount_drive()
                if did_remount:
                    if upload_mode:
                        try:
                            if path.stat().st_size == 0:
                                continue
                        except OSError:
                            continue
                        if out.exists():
                            continue
                    shutil.copy2(path, out)
                    continue
            raise
    return file_count


def _rsync_copy(
    src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False
) -> None:
    """Copy using rsync and raise _MountDropError on mount-related failures."""
    cmd = ["rsync", "-a", "--no-compress", "--prune-empty-dirs"]

    # Upload mode: do not overwrite existing files; skip zero-byte placeholders
    if upload_mode:
        cmd += ["--ignore-existing", "--min-size=1"]

    if patterns is None:
        pass  # copy everything
    else:
        pats = set(patterns)
        if pats == {"*.csv"} or pats == {".csv", "*.csv"}:
            cmd += ["--include", "*/", "--include", "*.csv", "--exclude", "*"]
        else:
            # Fallback handled by caller with Python copy
            raise RuntimeError("Unsupported include pattern; falling back to Python copy.")

    dst.mkdir(parents=True, exist_ok=True)
    src_arg = str(src) + "/"
    dst_arg = str(dst) + "/"

    proc = subprocess.run(cmd + [src_arg, dst_arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "") + " " + (proc.stdout or "")
        if ("Transport endpoint is not connected" in err) or ("Input/output error" in err):
            raise _MountDropError(err)  # signal a remount condition
        raise RuntimeError(f"rsync failed ({proc.returncode}): {err.strip()}")


class _MountDropError(RuntimeError):
    """Raised when the Drive mount drops during rsync operations."""
    pass


def _remount_drive() -> bool:
    """Attempt to remount /content/drive; return True when successful."""
    try:
        subprocess.run(["fusermount", "-u", "/content/drive"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        subprocess.run(["rm", "-rf", "/content/drive"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        from google.colab import drive as colab_drive
        colab_drive.mount("/content/drive", force_remount=True)
        return True
    except Exception:
        return False


def _fmt_seconds(s: float) -> str:
    """Return HH:MM:SS or MM:SS for short durations."""
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

#%%% CELL 11 – EXPORTS
"""
Purpose
Expose public names that notebooks and runners are expected to import.

Steps
- Provide data classes and the main staging/sync functions.
"""
__all__ = [
    "DrivePaths",
    "LocalPaths",
    "StageSummary",
    "load_configs",
    "validate_inputs",
    "local_mirrors",
    "stage_to_local",
    "make_local_pathconfig",
    "sync_outputs_back",
    "start_background_sync",
    "stop_background_sync",
]

#%%% CELL 12 – EXECUTION GUARD
"""
Purpose
Prevent accidental module execution in Colab; this file is imported by the
notebook or by a runner script that calls the public functions.

Steps
- Raise a clear error if executed directly.
"""
if __name__ == "__main__":
    raise RuntimeError("Direct execution not supported – use the Run notebook.")
