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

    # -------------------- Positive raw data --------------------
    # Train / validation (PNNN)
    POS_TRAIN_VAL1_METADATA_DIR = RAW_ROOT / "Rumble" / "Training" / "pnnn"
    POS_TRAIN_VAL1_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Training" / "Sounds"
    POS_TRAIN_VAL2_METADATA_DIR = RAW_ROOT / "Rumble" / "Testing" / "PNNN"
    POS_TRAIN_VAL2_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Testing" / "PNNN" / "Sounds"

    # Holdout testing (Dzanga) — also used for holdout-test negative planning
    # via buffered exclusion around positive rumble annotations.
    POS_HOLDOUT_TEST_METADATA_DIR = RAW_ROOT / "Rumble" / "Testing" / "Dzanga"
    POS_HOLDOUT_TEST_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Testing" / "Dzanga" / "Sounds"

    # -------------------- Negative raw data --------------------
    # Train/val negatives: 24hr PNNN WAVs with long stretches of non-gunshot
    # background noise, repurposed as rumble-detector negatives.
    NEG_SOURCE_INPUT_DIR = RAW_ROOT / "Gunshot" / "Testing" / "PNNN" / "Sounds"

else:
    RAW_ROOT = None

    POS_TRAIN_VAL1_METADATA_DIR = None
    POS_TRAIN_VAL1_SOUNDS_DIR = None

    POS_TRAIN_VAL2_METADATA_DIR = None
    POS_TRAIN_VAL2_SOUNDS_DIR = None

    POS_HOLDOUT_TEST_METADATA_DIR = None
    POS_HOLDOUT_TEST_SOUNDS_DIR = None

    NEG_SOURCE_INPUT_DIR = None

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