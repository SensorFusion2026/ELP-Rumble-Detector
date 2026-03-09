# src/elp_rumble/data_creation/create_splits.py
# Usage: python -m elp_rumble.data_creation.create_splits

import pandas as pd
import numpy as np

from elp_rumble.config.paths import CLIPS_PLAN_CSV, SPLITS_DIR

# -----------------------
# Paths
# -----------------------
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Settings
# -----------------------
SEED = 42
RNG = np.random.default_rng(SEED)

TRAIN_FRAC = 0.8
# val = remainder (non-Dzanga only; Dzanga clips are always test)

MODEL1_CAPS = {
    "train_val":    {"pos": 60, "neg": 60},
    "holdout_test": {"pos": 30, "neg": 30},
}

MODEL2_FRAC = 0.5
MODEL3_FRAC = 1.0


# -----------------------
# Helper functions
# -----------------------
def subsample_by_wav(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    Subsample the dataset by selecting a fraction of unique source WAV files
    and keeping all clips that originate from those WAVs.

    This is used to control dataset size for different model versions (e.g.,
    Model 2 uses 50% of available WAVs), while avoiding clip-level leakage.
    """
    wavs = df["source_wav_relpath"].drop_duplicates().to_numpy()
    RNG.shuffle(wavs)

    k = max(1, int(frac * len(wavs)))
    keep = set(wavs[:k])
    return df[df["source_wav_relpath"].isin(keep)].copy()

def split_by_wav(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test splits using location-based holdout and WAV-level grouping.

    Clips marked holdout_test in clips_plan (Dzanga) -> test split.
    Clips marked train_val in clips_plan (PNNN) -> train/val by source WAV.

    All clips from the same source WAV are placed in the same split to prevent
    background and recording-condition leakage across splits.
    """
    holdout = df[df["split"] == "holdout_test"].copy()
    train_val = df[df["split"] == "train_val"].copy()

    holdout["split"] = "test"

    wavs = train_val["source_wav_relpath"].drop_duplicates().to_numpy()
    RNG.shuffle(wavs)
    n_train = int(TRAIN_FRAC * len(wavs))
    train_wavs = set(wavs[:n_train])

    train_val["split"] = "val"
    train_val.loc[train_val["source_wav_relpath"].isin(train_wavs), "split"] = "train"

    return pd.concat([holdout, train_val], ignore_index=True)

def build_model1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small feasibility dataset using absolute caps per split+label.
    """
    parts = []
    for split, caps in MODEL1_CAPS.items():
        for label, cap in caps.items():
            sub = df[(df["split"] == split) & (df["label"] == label)]
            if sub.empty:
                continue
            sub = sub.sample(frac=1.0, random_state=SEED)
            parts.append(sub.head(cap))

    if not parts:
        return df.head(0).copy()

    return pd.concat(parts, ignore_index=True)


def write_split(df: pd.DataFrame, path):
    cols = ["split", "label", "location", "source_wav_relpath", "clip_wav_relpath"]
    df = df[cols].sort_values(cols)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def summarize(df, name):
    print(f"\n{name}")
    print(df.groupby(["split", "location", "label"]).size())


def main():
    # -----------------------
    # Load clip plan
    # -----------------------
    if not CLIPS_PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing {CLIPS_PLAN_CSV}. Run create_clips_plan.py first.")

    plan = pd.read_csv(CLIPS_PLAN_CSV)

    required = {"label", "location", "split", "source_wav_relpath", "clip_wav_relpath"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"clips_plan.csv missing columns: {sorted(missing)}")

    # -----------------------
    # Build splits
    # -----------------------

    # Model 1: feasibility (small amount of data)
    model1_base = build_model1(plan)
    model1 = split_by_wav(model1_base)

    # Model 2: scalability (50% of data)
    model2_base = subsample_by_wav(plan, MODEL2_FRAC)
    model2 = split_by_wav(model2_base)

    # Model 3: performance (full data)
    model3_base = subsample_by_wav(plan, MODEL3_FRAC)
    model3 = split_by_wav(model3_base)

    # -----------------------
    # Write outputs
    # -----------------------
    write_split(model1, SPLITS_DIR / "model1.csv")
    write_split(model2, SPLITS_DIR / "model2.csv")
    write_split(model3, SPLITS_DIR / "model3.csv")

    # -----------------------
    # Print summaries
    # -----------------------
    summarize(model1, "MODEL 1 (Feasibility)")
    summarize(model2, "MODEL 2 (Scaled)")
    summarize(model3, "MODEL 3 (Performance)")


if __name__ == "__main__":
    main()
