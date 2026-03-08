# src/elp_rumble/data_creation/create_clips_plan.py
# Usage: python -m elp_rumble.data_creation.create_clips_plan

from pathlib import Path
import random
import wave
import pandas as pd

from .utils import find_wav_files, validate_dir
from elp_rumble.config.paths import (
    RAW_ROOT,
    POS_TRAIN_VAL1_METADATA_DIR,
    POS_TRAIN_VAL1_SOUNDS_DIR,
    POS_TRAIN_VAL2_METADATA_DIR,
    POS_TRAIN_VAL2_SOUNDS_DIR,
    POS_HOLDOUT_TEST_METADATA_DIR,
    POS_HOLDOUT_TEST_SOUNDS_DIR,
    NEG_SOURCE_INPUT_DIR,
    CLIPS_PLAN_CSV,
)

CLIP_LEN_S = 5.0
SEED = 42
RNG = random.Random(SEED)


def _source_relpath(path: Path) -> str:
    if RAW_ROOT is None:
        raise ValueError("RAW_ROOT is not configured. Set ENVIRONMENT=local and CORNELL_DATA_ROOT.")
    return str(path.resolve().relative_to(Path(RAW_ROOT).resolve()))


def _resolve_wav_path(sounds_dir: Path, begin_file: str, split: str) -> Path | None:
    candidate = sounds_dir / begin_file
    if candidate.exists():
        return candidate

    # Dzanga metadata may use dzan_ while disk files use dz_.
    if split == "holdout_test" and begin_file.startswith("dzan_"):
        alt = sounds_dir / begin_file.replace("dzan_", "dz_", 1)
        if alt.exists():
            return alt

    return None


def _load_positive_rows(metadata_dir: Path, sounds_dir: Path, split: str, location: str, start_idx: int):
    rows = []
    idx = start_idx

    meta_files = sorted([p for p in metadata_dir.iterdir() if p.suffix == ".txt"])
    for meta_file in meta_files:
        data = pd.read_csv(meta_file, delimiter="\t")

        if "Tag 1" in data.columns:
            data = data[~data["Tag 1"].isin(["DUMMY_NoEles", "DUMMY_noEles"])]
        if "notes" in data.columns:
            data = data[~data["notes"].astype(str).str.contains("faint|marginal|gorilla", case=False, na=False)]

        for _, clip in data.iterrows():
            begin_file = str(clip["Begin File"])
            wav_path = _resolve_wav_path(sounds_dir, begin_file, split)
            if wav_path is None:
                continue

            start_s = float(clip["File Offset (s)"])
            source_rel = _source_relpath(wav_path)
            clip_rel = f"pos/{split}/{wav_path.stem}_pos_{int(round(start_s))}_{idx}.wav"

            rows.append(
                {
                    "label": "pos",
                    "location": location,
                    "split": split,
                    "source_wav_relpath": source_rel,
                    "start_s": start_s,
                    "duration_s": float(CLIP_LEN_S),
                    "clip_wav_relpath": clip_rel,
                }
            )
            idx += 1

    return rows, idx


def _collect_negative_windows() -> list[dict]:
    neg_wavs: list[str] = []
    find_wav_files(str(NEG_SOURCE_INPUT_DIR), neg_wavs)
    if not neg_wavs:
        raise ValueError(f"No negative source wav files found in {NEG_SOURCE_INPUT_DIR}")

    windows = []
    for wav_str in neg_wavs:
        wav_path = Path(wav_str)
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            nframes = w.getnframes()

        frames_per_clip = int(CLIP_LEN_S * sr)
        if nframes < frames_per_clip:
            continue

        source_rel = _source_relpath(wav_path)
        max_start = nframes - frames_per_clip
        start_frame = 0
        while start_frame <= max_start:
            windows.append(
                {
                    "source_wav_relpath": source_rel,
                    "start_s": float(start_frame / sr),
                    "duration_s": float(CLIP_LEN_S),
                    "wav_stem": wav_path.stem,
                }
            )
            start_frame += frames_per_clip

    return windows


def main():
    if CLIPS_PLAN_CSV.exists():
        raise RuntimeError(
            f"{CLIPS_PLAN_CSV} already exists. Delete it intentionally before regenerating clips_plan.csv."
        )

    validate_dir(POS_TRAIN_VAL1_METADATA_DIR, "POS_TRAIN_VAL1_METADATA_DIR")
    validate_dir(POS_TRAIN_VAL1_SOUNDS_DIR, "POS_TRAIN_VAL1_SOUNDS_DIR")
    validate_dir(POS_TRAIN_VAL2_METADATA_DIR, "POS_TRAIN_VAL2_METADATA_DIR")
    validate_dir(POS_TRAIN_VAL2_SOUNDS_DIR, "POS_TRAIN_VAL2_SOUNDS_DIR")
    validate_dir(POS_HOLDOUT_TEST_METADATA_DIR, "POS_HOLDOUT_TEST_METADATA_DIR")
    validate_dir(POS_HOLDOUT_TEST_SOUNDS_DIR, "POS_HOLDOUT_TEST_SOUNDS_DIR")
    validate_dir(NEG_SOURCE_INPUT_DIR, "NEG_SOURCE_INPUT_DIR")

    rows = []
    idx = 1

    train_val_pos_1, idx = _load_positive_rows(
        Path(POS_TRAIN_VAL1_METADATA_DIR),
        Path(POS_TRAIN_VAL1_SOUNDS_DIR),
        split="train_val",
        location="pnnn",
        start_idx=idx,
    )
    train_val_pos_2, idx = _load_positive_rows(
        Path(POS_TRAIN_VAL2_METADATA_DIR),
        Path(POS_TRAIN_VAL2_SOUNDS_DIR),
        split="train_val",
        location="pnnn",
        start_idx=idx,
    )
    holdout_test_pos, idx = _load_positive_rows(
        Path(POS_HOLDOUT_TEST_METADATA_DIR),
        Path(POS_HOLDOUT_TEST_SOUNDS_DIR),
        split="holdout_test",
        location="dzanga",
        start_idx=idx,
    )

    rows.extend(train_val_pos_1)
    rows.extend(train_val_pos_2)
    rows.extend(holdout_test_pos)

    train_val_pos_count = len([r for r in rows if r["label"] == "pos" and r["split"] == "train_val"])
    holdout_test_pos_count = len([r for r in rows if r["label"] == "pos" and r["split"] == "holdout_test"])

    windows = _collect_negative_windows()
    RNG.shuffle(windows)

    required_neg = train_val_pos_count + holdout_test_pos_count
    if len(windows) < required_neg:
        raise ValueError(
            f"Insufficient negative windows ({len(windows)}) for required negatives ({required_neg})."
        )

    train_val_windows = windows[:train_val_pos_count]
    holdout_test_windows = windows[train_val_pos_count:train_val_pos_count + holdout_test_pos_count]

    neg_idx = 1
    for w in train_val_windows:
        rows.append(
            {
                "label": "neg",
                "location": "pnnn_neg",
                "split": "train_val",
                "source_wav_relpath": w["source_wav_relpath"],
                "start_s": w["start_s"],
                "duration_s": w["duration_s"],
                "clip_wav_relpath": f"neg/train_val/{w['wav_stem']}_neg_{int(w['start_s'] * 1000)}_{neg_idx}.wav",
            }
        )
        neg_idx += 1

    for w in holdout_test_windows:
        rows.append(
            {
                "label": "neg",
                "location": "pnnn_neg",
                "split": "holdout_test",
                "source_wav_relpath": w["source_wav_relpath"],
                "start_s": w["start_s"],
                "duration_s": w["duration_s"],
                "clip_wav_relpath": f"neg/holdout_test/{w['wav_stem']}_neg_{int(w['start_s'] * 1000)}_{neg_idx}.wav",
            }
        )
        neg_idx += 1

    plan = pd.DataFrame(rows)
    plan = plan.sort_values(["label", "split", "source_wav_relpath", "start_s"]).reset_index(drop=True)
    plan.to_csv(CLIPS_PLAN_CSV, index=False)

    print(f"Wrote plan: {CLIPS_PLAN_CSV}")
    print(f"Train/validation positives: {train_val_pos_count}")
    print(f"Holdout test positives: {holdout_test_pos_count}")
    print(f"Train/validation negatives: {len(train_val_windows)}")
    print(f"Holdout test negatives: {len(holdout_test_windows)}")
    print(f"Total planned clips: {len(plan)}")


if __name__ == "__main__":
    main()
