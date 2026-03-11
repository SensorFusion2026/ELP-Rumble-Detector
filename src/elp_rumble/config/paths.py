"""
paths.py

Centralized path configuration for the ELP Rumble Detector.

Current path policy:
- Versioned manifests live in `src/elp_rumble/data_creation/`.
- Generated artifacts (wav clips and TFRecords) live under `data/`.
- Raw Cornell source paths are resolved only when `ENVIRONMENT=local`.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- Project root (repo root) -----------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Get environment variables from .env located at the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

ENVIRONMENT = os.getenv("ENVIRONMENT", "remote")
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")
# ---------------------------------------------------------------------
# Raw Cornell data roots (local only)
# ---------------------------------------------------------------------

if ENVIRONMENT == "local":
    if not CORNELL_DATA_ROOT:
        raise ValueError(
            "CORNELL_DATA_ROOT not found. Create a .env in the project root and add:\n"
            'CORNELL_DATA_ROOT="/path/to/your/data"\n'
            'ENVIRONMENT="local"'
        )

    RAW_ROOT = Path(CORNELL_DATA_ROOT)

    # ------------------------- PNNN -------------------------
    PNNN1_METADATA = RAW_ROOT / "Rumble" / "Training" / "pnnn"
    PNNN1_SOUNDS = RAW_ROOT / "Rumble" / "Training" / "Sounds"
    PNNN2_METADATA = RAW_ROOT / "Rumble" / "Testing" / "PNNN"
    PNNN2_SOUNDS = RAW_ROOT / "Rumble" / "Testing" / "PNNN" / "Sounds"

    # ------------------------ Dzanga ------------------------
    DZANGA_METADATA = RAW_ROOT / "Rumble" / "Testing" / "Dzanga"
    DZANGA_SOUNDS = RAW_ROOT / "Rumble" / "Testing" / "Dzanga" / "Sounds"

else:
    RAW_ROOT = None

    PNNN1_METADATA = None
    PNNN1_SOUNDS = None

    PNNN2_METADATA = None
    PNNN2_SOUNDS = None

    DZANGA_METADATA = None
    DZANGA_SOUNDS = None

# ---------------------------------------------------------------------
# Repository-managed data directories
# ---------------------------------------------------------------------

# Versioned manifests for reproducible data creation
DATA_CREATION_ROOT = PROJECT_ROOT / "src" / "elp_rumble" / "data_creation"
CLIPS_PLAN_CSV = DATA_CREATION_ROOT / "clips_plan.csv"
SPLITS_DIR = DATA_CREATION_ROOT / "splits"

# Generated artifacts (not versioned)
DATA_ROOT = PROJECT_ROOT / "data"

# Generated wav clips
WAV_CLIPS_ROOT = DATA_ROOT / "wav_clips"

# Generated TFRecords
TFRECORDS_ROOT = DATA_ROOT / "tfrecords"
TFRECORDS_AUDIO_DIR = TFRECORDS_ROOT / "tfrecords_audio"
TFRECORDS_SPECTROGRAM_DIR = TFRECORDS_ROOT / "tfrecords_spectrogram"

# Training run outputs
RUNS_DIR = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------
# Utility: create derived directories if needed
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create required repo-managed directories if they do not exist.
    Safe to call at the beginning of preprocessing/training scripts.
    """
    for p in [
        DATA_ROOT,
        WAV_CLIPS_ROOT,
        SPLITS_DIR,
        TFRECORDS_ROOT,
        TFRECORDS_AUDIO_DIR,
        TFRECORDS_SPECTROGRAM_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)