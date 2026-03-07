# src/elp_rumble/data_creation/create_splits.py
# Usage: python -m elp_rumble.data_creation.create_splits

from pathlib import Path
import os
import pandas as pd

HERE = Path(__file__).resolve().parent
PLAN_CSV = HERE / "clips_plan.csv"
SPLITS_DIR = HERE / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("SPLIT_SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.8"))
if not (0.0 < TRAIN_FRAC < 1.0):
    raise ValueError(f"Invalid TRAIN_FRAC={TRAIN_FRAC}. Require 0 < TRAIN_FRAC < 1.")

# Feasibility model target totals (overall, not per split)
MODEL1_POS_TOTAL = 60
MODEL1_NEG_TOTAL = 60

# Train/validate split for seen subset in model1
MODEL1_TEST_POS = MODEL1_POS_TOTAL // 2  # Dzanga holdout positives
MODEL1_SEEN_POS = MODEL1_POS_TOTAL - MODEL1_TEST_POS


def _shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _assemble_split_df(
    train_pos: pd.DataFrame,
    train_neg: pd.DataFrame,
    validate_pos: pd.DataFrame,
    validate_neg: pd.DataFrame,
    test_pos: pd.DataFrame,
    test_neg: pd.DataFrame,
) -> pd.DataFrame:
    train_pos = train_pos.copy()
    train_neg = train_neg.copy()
    validate_pos = validate_pos.copy()
    validate_neg = validate_neg.copy()
    test_pos = test_pos.copy()
    test_neg = test_neg.copy()

    train_pos["split"] = "train"
    train_neg["split"] = "train"
    validate_pos["split"] = "validate"
    validate_neg["split"] = "validate"
    test_pos["split"] = "test"
    test_neg["split"] = "test"

    split_df = pd.concat(
        [train_pos, train_neg, validate_pos, validate_neg, test_pos, test_neg],
        ignore_index=True,
    )
    return split_df[
        ["split", "label", "location", "source_wav_relpath", "clip_wav_relpath"]
    ].sort_values(["split", "label", "location", "clip_wav_relpath"])


def _build_model3(
    seen_pos: pd.DataFrame,
    holdout_pos: pd.DataFrame,
    seen_neg_pool: pd.DataFrame,
    holdout_neg_pool: pd.DataFrame,
) -> pd.DataFrame:
    n_seen_pos = len(seen_pos)
    n_train_pos = int(TRAIN_FRAC * n_seen_pos)

    train_pos = seen_pos.iloc[:n_train_pos].copy()
    validate_pos = seen_pos.iloc[n_train_pos:].copy()

    needed_seen_neg = len(train_pos) + len(validate_pos)
    if len(seen_neg_pool) < needed_seen_neg:
        raise ValueError(
            f"Insufficient seen negatives ({len(seen_neg_pool)}) for required seen positives ({needed_seen_neg})."
        )

    seen_neg = seen_neg_pool.iloc[:needed_seen_neg].copy().reset_index(drop=True)
    train_neg = seen_neg.iloc[:len(train_pos)].copy()
    validate_neg = seen_neg.iloc[len(train_pos):].copy()

    if len(holdout_neg_pool) < len(holdout_pos):
        raise ValueError(
            f"Insufficient holdout negatives ({len(holdout_neg_pool)}) for holdout positives ({len(holdout_pos)})."
        )
    test_neg = holdout_neg_pool.iloc[:len(holdout_pos)].copy()

    return _assemble_split_df(train_pos, train_neg, validate_pos, validate_neg, holdout_pos, test_neg)


def _build_model1(
    seen_pos: pd.DataFrame,
    holdout_pos: pd.DataFrame,
    seen_neg_pool: pd.DataFrame,
    holdout_neg_pool: pd.DataFrame,
) -> pd.DataFrame:
    if len(seen_pos) < MODEL1_SEEN_POS:
        raise ValueError(f"Need at least {MODEL1_SEEN_POS} seen positives for model1.")
    if len(holdout_pos) < MODEL1_TEST_POS:
        raise ValueError(f"Need at least {MODEL1_TEST_POS} holdout positives for model1.")

    seen_pos_small = seen_pos.iloc[:MODEL1_SEEN_POS].copy().reset_index(drop=True)
    test_pos = holdout_pos.iloc[:MODEL1_TEST_POS].copy().reset_index(drop=True)

    n_train_pos = int(TRAIN_FRAC * MODEL1_SEEN_POS)
    train_pos = seen_pos_small.iloc[:n_train_pos].copy()
    validate_pos = seen_pos_small.iloc[n_train_pos:].copy()

    needed_seen_neg = len(train_pos) + len(validate_pos)
    if len(seen_neg_pool) < needed_seen_neg:
        raise ValueError(
            f"Insufficient seen negatives ({len(seen_neg_pool)}) for required model1 seen positives ({needed_seen_neg})."
        )
    if len(holdout_neg_pool) < len(test_pos):
        raise ValueError(
            f"Insufficient holdout negatives ({len(holdout_neg_pool)}) for required model1 holdout positives ({len(test_pos)})."
        )

    seen_neg_small = seen_neg_pool.iloc[:needed_seen_neg].copy().reset_index(drop=True)
    train_neg = seen_neg_small.iloc[:len(train_pos)].copy()
    validate_neg = seen_neg_small.iloc[len(train_pos):].copy()
    test_neg = holdout_neg_pool.iloc[:len(test_pos)].copy().reset_index(drop=True)

    output = _assemble_split_df(train_pos, train_neg, validate_pos, validate_neg, test_pos, test_neg)

    n_pos = int((output["label"] == "pos").sum())
    n_neg = int((output["label"] == "neg").sum())
    if n_pos != MODEL1_POS_TOTAL or n_neg != MODEL1_NEG_TOTAL:
        raise ValueError(
            f"Model1 count mismatch: pos={n_pos} (expected {MODEL1_POS_TOTAL}), "
            f"neg={n_neg} (expected {MODEL1_NEG_TOTAL})."
        )

    # Guard Dzanga semantics: Dzanga positives should only appear in test.
    bad_dzanga = output[
        (output["label"] == "pos")
        & (output["location"] == "dzanga")
        & (output["split"] != "test")
    ]
    if not bad_dzanga.empty:
        raise ValueError("Model1 split violation: Dzanga positives found outside test split.")

    return output


def main():
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing {PLAN_CSV}. Run create_clips_plan.py first.")

    plan = pd.read_csv(PLAN_CSV)
    required = {
        "label",
        "location",
        "split_hint",
        "source_wav_relpath",
        "clip_wav_relpath",
    }
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"clips_plan.csv missing columns: {sorted(missing)}")

    seen_pos = _shuffle(plan[(plan["label"] == "pos") & (plan["split_hint"] == "seen")].copy(), SEED)
    holdout_pos = plan[(plan["label"] == "pos") & (plan["split_hint"] == "holdout")].copy()
    seen_neg_pool = _shuffle(plan[(plan["label"] == "neg") & (plan["split_hint"] == "seen")].copy(), SEED)
    holdout_neg_pool = _shuffle(plan[(plan["label"] == "neg") & (plan["split_hint"] == "holdout")].copy(), SEED)

    if seen_pos.empty or holdout_pos.empty or seen_neg_pool.empty or holdout_neg_pool.empty:
        raise ValueError("Plan has empty required groups. Rebuild clips_plan.csv.")

    model1 = _build_model1(seen_pos, holdout_pos, seen_neg_pool, holdout_neg_pool)
    model3 = _build_model3(seen_pos, holdout_pos, seen_neg_pool, holdout_neg_pool)

    model1_csv = SPLITS_DIR / "model1.csv"
    model3_csv = SPLITS_DIR / "model3.csv"
    model1.to_csv(model1_csv, index=False)
    model3.to_csv(model3_csv, index=False)

    print(f"Wrote {model1_csv} ({len(model1)} rows)")
    print(model1.groupby(["split", "label"]).size())
    print(f"Wrote {model3_csv} ({len(model3)} rows)")
    print(model3.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
