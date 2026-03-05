"""
paths.py

Centralized path configuration for the ELP Rumble Detector.
Adapted from legacy data_creation/data_path_config.py, but using the gunshot-style pattern:
- module-level Path constants (no class)
- loads .env
- defines PROJECT_ROOT via parents[3]
- defines RAW_ROOT from CORNELL_DATA_ROOT for local
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Get environment variables from .env
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "remote")
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")

# ---------- Project root (repo root) -----------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    # Positive Raw Data Paths - Train
    POS_TRAIN_VAL1_METADATA_DIR = RAW_ROOT / "Rumble" / "Training" / "pnnn"
    POS_TRAIN_VAL1_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Training" / "Sounds"
    POS_TRAIN_VAL2_METADATA_DIR = RAW_ROOT / "Rumble" / "Testing" / "PNNN"
    POS_TRAIN_VAL2_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Testing" / "PNNN" / "Sounds"

    # Holdout testing (Dzanga)
    POS_HOLDOUT_TEST_METADATA_DIR = RAW_ROOT / "Rumble" / "Testing" / "Dzanga"
    POS_HOLDOUT_TEST_SOUNDS_DIR = RAW_ROOT / "Rumble" / "Testing" / "Dzanga" / "Sounds"

    # -------------------- Negative raw data --------------------
    # NOTE: This is 24hr .wavs containing gunshot sounds with long stretches of 
    #       non-gunshot background noise in-between, to be used as negatives for 
    #       rumble detector.
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
# Derived project data directories (repo-managed)
# ---------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data"

# ------------- Preprocessed Clips Paths -------------
CLIPS_TRAIN_VAL_ROOT = DATA_ROOT / "clips_train_val"
CLIPS_HOLDOUT_TEST_ROOT = DATA_ROOT / "clips_holdout_test"

# Positive preprocessed clips
POS_TRAIN_VAL_CLIPS_DIR = CLIPS_TRAIN_VAL_ROOT / "pos_pnnn_clips"
POS_HOLDOUT_TEST_CLIPS_DIR = CLIPS_HOLDOUT_TEST_ROOT / "pos_dzanga_clips"

# Negative preprocessed clips
TRAIN_VAL_NEG_CLIPS_DIR = CLIPS_TRAIN_VAL_ROOT / "neg_pnnn_gunshot_clips"
HOLDOUT_TEST_NEG_CLIPS_DIR = CLIPS_HOLDOUT_TEST_ROOT / "neg_pnnn_gunshot_clips"

# ------------- TFRecords Paths -------------
TFRECORDS_ROOT = DATA_ROOT / "tfrecords"

# TFRecords audio
TFRECORDS_AUDIO_DIR = TFRECORDS_ROOT / "tfrecords_audio"

# TFRecords spectrogram
TFRECORDS_SPECTROGRAM_DIR = TFRECORDS_ROOT / "tfrecords_spectrogram"


# ---------------------------------------------------------------------
# Utility: create derived directories if needed
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create all repo-managed directories if they do not exist.
    Safe to call at the beginning of preprocessing/training scripts.
    """
    for p in [
        DATA_ROOT,
        CLIPS_TRAIN_VAL_ROOT,
        CLIPS_HOLDOUT_TEST_ROOT,
        POS_TRAIN_VAL_CLIPS_DIR,
        POS_HOLDOUT_TEST_CLIPS_DIR,
        TRAIN_VAL_NEG_CLIPS_DIR,
        HOLDOUT_TEST_NEG_CLIPS_DIR,
        TFRECORDS_ROOT,
        TFRECORDS_AUDIO_DIR,
        TFRECORDS_SPECTROGRAM_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)