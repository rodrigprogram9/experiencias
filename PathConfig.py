#%%% CELL 00 – MODULE OVERVIEW
"""
PathConfig.py

Purpose
This module defines and exports the experiment folder tree as Path objects. It
uses a single placeholder "__EXP_ROOT__" that upstream tooling replaces with
the real experiment root. All other modules import these constants to locate
data, code, and notebooks reliably.

Steps
- Import Path and declare the root placeholder.
- Build standard folder constants under the root.
- Provide convenience glob generators for common queries.
- Export individual file locations and the public symbol list.

Output
- Path constants for root, folders, and key files.
- Convenience glob generators: gTracked, gPose, gScored, gScoredPose, gError.

Structure
ExperimentalFolder/
├── Protocols/
│   ├── Codes/
│   │   ├── BehaviorScoringRun.ipynb
│   │   ├── CreateDataFramesRun.ipynb
│   │   ├── BehaviorScoring/
│   │   │   ├── BehaviorScoringFunctions.py
│   │   │   └── BehaviorScoringMain.py
│   │   ├── CreateDataFrames/
│   │   │   ├── CreateDataFramesFunctions.py
│   │   │   └── CreateDataFramesMain.py
│   │   └── Config/
│   │       ├── ExperimentConfig.py
│   │       ├── BehaviorScoringConfig.py
│   │       ├── CreateDataFramesConfig.py
│   │       ├── ParamsConfig.py
│   │       ├── ColorsConfig.py
│   │       ├── TimeConfig.py
│   │       └── PathConfig.py
│   └── Bonfly/
│       ├── Protocol/
│       └── Tracker/
├── RawData/
├── PostProcessing/
│   ├── Arenas/
│   ├── CropVideo/
│   ├── Tracked/
│   ├── Pose/
│   ├── Scored/
│   ├── ScoredPose/
│   └── ScoredError/
└── Analysis/
    ├── DataFrames/
    ├── ZoomDataFrames/
    └── Plots/
"""

#%%% CELL 01 – IMPORTS & ROOT PLACEHOLDER
"""
Purpose
Import Path and declare a root placeholder that external setup replaces.

Steps
- Import Path from pathlib.
- Define pExperimentalRoot using the "__EXP_ROOT__" placeholder.
- Provide a small helper to build subpaths under the root.
"""

from pathlib import Path

# Auto-injected upstream: replace with the real experiment root before use
pExperimentalRoot = Path("_PLACEHOLDER_EXPERIMENT_ROOT_")

def _p(sub: str) -> Path:
    """Return a path under the experimental root."""
    return pExperimentalRoot / sub  # small helper avoids repeated / chains

#%%% CELL 02 – STANDARD FOLDER CONSTANTS
"""
Purpose
Expose the canonical folder structure as Path objects for downstream modules.

Steps
- Define Protocols, Codes, and subfolders for code organization.
- Define RawData and PostProcessing folders and their subfolders.
- Define Analysis folders for derived outputs.
"""

# Protocols and codes
pProtocols        = _p("Protocols")
pCodes            = pProtocols / "Codes"
pBehaviorScoring  = pCodes / "BehaviorScoring"
pCreateDataFrames = pCodes / "CreateDataFrames"
pConfig           = pCodes / "Config"

# Raw data
pRawData          = _p("RawData")

# Post-processing
pPostProcessing   = _p("PostProcessing")
pArenas           = pPostProcessing / "Arenas"
pCropVideo        = pPostProcessing / "CropVideo"
pTracked          = pPostProcessing / "Tracked"
pPose             = pPostProcessing / "Pose"
pScored           = pPostProcessing / "Scored"
pScoredPose       = pPostProcessing / "ScoredPose"
pScoredError      = pPostProcessing / "ScoredError"

# Analysis products
pAnalysis         = _p("Analysis")
pDataFrames       = pAnalysis / "DataFrames"
pZoomDataFrames   = pAnalysis / "ZoomDataFrames"
pPlots            = pAnalysis / "Plots"

#%%% CELL 03 – CONVENIENCE GLOBS
"""
Purpose
Provide ready-to-use glob generators for common queries in notebooks/scripts.

Steps
- Expose tracked, pose, scored, scored pose, and error CSV iterators.
- These return Path generators; iterate or wrap with list() when needed.

note
- In Colab, call make_local_pathconfig(...) first to get globs rebased to
  /content. The rebased object injects fresh g* generators under local paths.
"""

# Tracked and pose inputs
gTracked    = pTracked.rglob("*tracked.csv")
gPose       = pPose.rglob("*pose.csv")

# Outputs (final CSVs; consumers should ignore any *.tmp files)
gScored     = pScored.rglob("*.csv")
gScoredPose = pScoredPose.rglob("*.csv")
gError      = pScoredError.rglob("*.csv")

#%%% CELL 04 – INDIVIDUAL FILE LOCATIONS
"""
Purpose
Expose key notebooks and scripts as Path objects for quick access/import.

Steps
- Provide notebook paths.
- Provide code module paths.
- Provide shared config module paths.
"""

# Notebooks
BSR  = pCodes / "BehaviorScoringRun.ipynb"
CDFR = pCodes / "CreateDataFramesRun.ipynb"

# BehaviorScoring code
BSF  = pBehaviorScoring / "BehaviorScoringFunctions.py"
BSM  = pBehaviorScoring / "BehaviorScoringMain.py"

# CreateDataFrames code
CDFF = pCreateDataFrames / "CreateDataFramesFunctions.py"
CDFM = pCreateDataFrames / "CreateDataFramesMain.py"

# Shared configs
EXPconfig   = pConfig / "ExperimentConfig.py"         # import as EXPconfig
BSconfig    = pConfig / "BehaviorScoringConfig.py"    # import as BSconfig
CDFconfig   = pConfig / "CreateDataFramesConfig.py"   # import as CDFconfig
PARAMconfig = pConfig / "ParamsConfig.py"             # import as PARAMconfig
COLORconfig = pConfig / "ColorsConfig.py"             # import as COLORconfig
TIMEconfig  = pConfig / "TimeConfig.py"               # import as TIMEconfig
PATHconfig  = pConfig / "PathConfig.py"               # import as PATHconfig

#%%% CELL 05 – EXPORTS
"""
Purpose
Declare the public surface so users can rely on star-imports in notebooks.

Steps
- Export root, folders, globs, and key file paths.
"""

__all__ = [
    # root
    "pExperimentalRoot",
    # folders
    "pProtocols", "pCodes", "pBehaviorScoring", "pCreateDataFrames", "pConfig",
    "pRawData",
    "pPostProcessing", "pArenas", "pCropVideo", "pTracked", "pPose",
    "pScored", "pScoredPose", "pScoredError",
    "pAnalysis", "pDataFrames", "pZoomDataFrames", "pPlots",
    # convenience globs
    "gTracked", "gPose", "gScored", "gScoredPose", "gError",
    # notebooks
    "BSR", "CDFR",
    # code modules
    "BSF", "BSM", "CDFF", "CDFM",
    # configs
    "EXPconfig", "BSconfig", "CDFconfig", "PARAMconfig",
    "COLORconfig", "TIMEconfig", "PATHconfig",
]

