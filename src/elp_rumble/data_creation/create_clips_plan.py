# src/elp_rumble/data_creation/create_clips_plan.py
# Usage: python -m elp_rumble.data_creation.create_clips_plan

from pathlib import Path
import random
import wave

import numpy as np
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

# -----------------------
# Settings
# -----------------------
CLIP_LEN_S = 5.0
BUFFER_S = 10.0   # holdout negatives must be >= BUFFER_S from any positive window
SEED = 42
RNG = random.Random(SEED)
NP_RNG = np.random.default_rng(SEED)


# -----------------------
# Helper functions
# -----------------------
def _source_relpath(path: Path) -> str:
    """Return path relative to RAW_ROOT for portability."""
    if RAW_ROOT is None:
        raise ValueError("RAW_ROOT is not configured. Set ENVIRONMENT=local and CORNELL_DATA_ROOT.")
    return str(path.resolve().relative_to(Path(RAW_ROOT).resolve()))


def _resolve_wav_path(sounds_dir: Path, begin_file: str, location: str) -> Path | None:
    """Resolve a wav filename, handling Dzanga naming variants."""
    candidate = sounds_dir / begin_file
    if candidate.exists():
        return candidate

    # Dzanga metadata may use dzan_ while disk files use dz_.
    if location == "dzanga" and begin_file.startswith("dzan_"):
        alt = sounds_dir / begin_file.replace("dzan_", "dz_", 1)
        if alt.exists():
            return alt

    return None


def _plan_positives(metadata_dir: Path, sounds_dir: Path, split: str, location: str, start_idx: int):
    """Plan positive clips from a single metadata + sounds source."""
    rows = []
    idx = start_idx

    meta_files = sorted(p for p in metadata_dir.iterdir() if p.suffix == ".txt")
    for meta_file in meta_files:
        data = pd.read_csv(meta_file, delimiter="\t")

        if "Tag 1" in data.columns:
            data = data[~data["Tag 1"].isin(["DUMMY_NoEles", "DUMMY_noEles"])]
        if "notes" in data.columns:
            data = data[~data["notes"].astype(str).str.contains("faint|marginal|gorilla", case=False, na=False)]

        for _, clip in data.iterrows():
            begin_file = str(clip["Begin File"])
            wav_path = _resolve_wav_path(sounds_dir, begin_file, location)
            if wav_path is None:
                continue

            start_s = float(clip["File Offset (s)"])
            source_rel = _source_relpath(wav_path)
            clip_rel = f"pos/{split}/{wav_path.stem}_pos_{int(round(start_s))}_{idx}.wav"

            rows.append({
                "label": "pos",
                "location": location,
                "split": split,
                "source_wav_relpath": source_rel,
                "start_s": start_s,
                "duration_s": float(CLIP_LEN_S),
                "clip_wav_relpath": clip_rel,
            })
            idx += 1

    return rows, idx


def _plan_negatives(neg_source_dir: Path) -> list[dict]:
    """Collect all possible negative clip windows from the negative source directory."""
    neg_wavs: list[str] = []
    find_wav_files(str(neg_source_dir), neg_wavs)
    if not neg_wavs:
        raise ValueError(f"No negative source wav files found in {neg_source_dir}")

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
            windows.append({
                "source_wav_relpath": source_rel,
                "start_s": float(start_frame / sr),
                "duration_s": float(CLIP_LEN_S),
                "wav_stem": wav_path.stem,
            })
            start_frame += frames_per_clip

    return windows


def _plan_dzanga_holdout_negatives(
    metadata_dir: Path,
    sounds_dir: Path,
    target_count: int,
) -> list[dict]:
    """Plan holdout-test negatives from Dzanga WAVs via buffered exclusion
    around positive rumble annotations.

    Mirrors the ELP-Gunshot-Detector pattern: for each WAV with positives,
    build forbidden zones (annotation span + BUFFER_S on each side), merge
    overlapping zones, invert to allowed intervals, discretize into clip-
    length starts, then sample without replacement from the global pool.
    """
    # Build positive windows per WAV from all non-DUMMY annotations.
    # Keep faint/marginal/gorilla as forbidden (conservative).
    pos_windows_by_wav: dict[str, list[tuple[float, float]]] = {}

    meta_files = sorted(p for p in metadata_dir.iterdir() if p.suffix == ".txt")
    for meta_file in meta_files:
        data = pd.read_csv(meta_file, delimiter="\t")

        if "Tag 1" in data.columns:
            data = data[~data["Tag 1"].isin(["DUMMY_NoEles", "DUMMY_noEles"])]

        required_cols = {"Begin File", "File Offset (s)", "Begin Time (s)", "End Time (s)"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(
                f"{meta_file.name}: missing columns for buffer exclusion: {sorted(missing)}"
            )

        for _, row in data.iterrows():
            begin_file = str(row["Begin File"])
            wav_path = _resolve_wav_path(sounds_dir, begin_file, "dzanga")
            if wav_path is None:
                continue

            source_rel = _source_relpath(wav_path)
            offset = float(row["File Offset (s)"])
            duration = float(row["End Time (s)"]) - float(row["Begin Time (s)"])
            pos_windows_by_wav.setdefault(source_rel, []).append(
                (offset, offset + duration)
            )

    # For each WAV, compute allowed intervals and collect candidate starts
    all_starts: list[tuple[str, float, str]] = []

    for source_rel, windows in sorted(pos_windows_by_wav.items()):
        wav_abs = RAW_ROOT / source_rel
        if not wav_abs.exists():
            continue

        with wave.open(str(wav_abs), "rb") as w:
            sr = w.getframerate()
            total_s = w.getnframes() / sr

        # Forbidden zones: annotation span + buffer on each side
        forbidden = [
            (max(0.0, begin - BUFFER_S), min(total_s, end + BUFFER_S))
            for begin, end in windows
        ]

        # Sort and merge overlapping forbidden zones
        forbidden.sort()
        merged: list[list[float]] = []
        for start, end in forbidden:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        # Invert to allowed intervals
        allowed: list[tuple[float, float]] = []
        t = 0.0
        for start, end in merged:
            if start - t >= CLIP_LEN_S:
                allowed.append((t, start))
            t = max(t, end)
        if total_s - t >= CLIP_LEN_S:
            allowed.append((t, total_s))
        if not allowed:
            continue

        # Discretize allowed intervals into clip-length starts
        wav_stem = Path(source_rel).stem
        for a, b in allowed:
            s = a
            while s + CLIP_LEN_S <= b:
                all_starts.append((source_rel, s, wav_stem))
                s += CLIP_LEN_S

    if not all_starts:
        raise ValueError(
            "No valid Dzanga holdout negative windows found after buffer exclusion. "
            "Try reducing BUFFER_S or adding more Dzanga source WAVs."
        )

    # Sample from the global pool
    k = min(target_count, len(all_starts))
    if k < target_count:
        print(
            f"WARNING: Only {k} Dzanga holdout neg windows available "
            f"(requested {target_count}). Using all available."
        )

    indices = NP_RNG.choice(len(all_starts), size=k, replace=False)
    chosen = [all_starts[i] for i in sorted(indices)]

    rows = []
    for j, (source_rel, start_s, wav_stem) in enumerate(chosen, start=1):
        rows.append({
            "label": "neg",
            "location": "dzanga",
            "split": "holdout_test",
            "source_wav_relpath": source_rel,
            "start_s": float(start_s),
            "duration_s": float(CLIP_LEN_S),
            "clip_wav_relpath": f"neg/holdout_test/{wav_stem}_neg_{int(start_s * 1000)}_{j}.wav",
        })

    return rows


def main():
    # -----------------------
    # Safety check: clip plan overwrite
    # -----------------------
    if CLIPS_PLAN_CSV.exists():
        raise RuntimeError(
            f"\n⚠️  clips_plan.csv already exists at:\n"
            f"    {CLIPS_PLAN_CSV}\n\n"
            f"This file is the source of truth for clip generation and is shared\n"
            f"across the team. Re-creating it will require everyone to re-cut\n"
            f"WAV clips and rebuild downstream artifacts (splits, TFRecords).\n\n"
            f"If you REALLY intend to regenerate the clip plan:\n"
            f"  1) Delete this file manually\n"
            f"  2) Inform the rest of the team\n"
            f"  3) Re-run create_clips_plan.py\n"
            f"  4) Re-generate WAV clips, split files, and TFRecords\n"
        )

    validate_dir(POS_TRAIN_VAL1_METADATA_DIR, "POS_TRAIN_VAL1_METADATA_DIR")
    validate_dir(POS_TRAIN_VAL1_SOUNDS_DIR, "POS_TRAIN_VAL1_SOUNDS_DIR")
    validate_dir(POS_TRAIN_VAL2_METADATA_DIR, "POS_TRAIN_VAL2_METADATA_DIR")
    validate_dir(POS_TRAIN_VAL2_SOUNDS_DIR, "POS_TRAIN_VAL2_SOUNDS_DIR")
    validate_dir(POS_HOLDOUT_TEST_METADATA_DIR, "POS_HOLDOUT_TEST_METADATA_DIR")
    validate_dir(POS_HOLDOUT_TEST_SOUNDS_DIR, "POS_HOLDOUT_TEST_SOUNDS_DIR")
    validate_dir(NEG_SOURCE_INPUT_DIR, "NEG_SOURCE_INPUT_DIR")

    # -----------------------
    # Positive datasets
    # -----------------------
    pos_datasets = [
        ("pnnn", "train_val", POS_TRAIN_VAL1_METADATA_DIR, POS_TRAIN_VAL1_SOUNDS_DIR),
        ("pnnn", "train_val", POS_TRAIN_VAL2_METADATA_DIR, POS_TRAIN_VAL2_SOUNDS_DIR),
        ("dzanga", "holdout_test", POS_HOLDOUT_TEST_METADATA_DIR, POS_HOLDOUT_TEST_SOUNDS_DIR),
    ]

    # -----------------------
    # POS plan
    # -----------------------
    rows = []
    idx = 1

    for location, split, metadata_dir, sounds_dir in pos_datasets:
        pos_rows, idx = _plan_positives(
            Path(metadata_dir), Path(sounds_dir), split, location, idx
        )
        rows.extend(pos_rows)
        print(f"Planned {len(pos_rows)} pos clips for {location}/{split}")

    train_val_pos_count = sum(1 for r in rows if r["label"] == "pos" and r["split"] == "train_val")
    holdout_test_pos_count = sum(1 for r in rows if r["label"] == "pos" and r["split"] == "holdout_test")

    # -----------------------
    # NEG plan: train/val negatives (PNNN background)
    # -----------------------
    windows = _plan_negatives(Path(NEG_SOURCE_INPUT_DIR))
    RNG.shuffle(windows)

    if len(windows) < train_val_pos_count:
        raise ValueError(
            f"Insufficient PNNN negative windows ({len(windows)}) "
            f"for required train/val negatives ({train_val_pos_count})."
        )

    train_val_windows = windows[:train_val_pos_count]

    neg_idx = 1
    for w in train_val_windows:
        rows.append({
            "label": "neg",
            "location": "pnnn",
            "split": "train_val",
            "source_wav_relpath": w["source_wav_relpath"],
            "start_s": w["start_s"],
            "duration_s": w["duration_s"],
            "clip_wav_relpath": f"neg/train_val/{w['wav_stem']}_neg_{int(w['start_s'] * 1000)}_{neg_idx}.wav",
        })
        neg_idx += 1

    print(f"Planned {len(train_val_windows)} neg clips for pnnn/train_val")

    # -----------------------
    # NEG plan: holdout-test negatives (Dzanga buffered exclusion)
    # -----------------------
    dzanga_neg_rows = _plan_dzanga_holdout_negatives(
        Path(POS_HOLDOUT_TEST_METADATA_DIR),
        Path(POS_HOLDOUT_TEST_SOUNDS_DIR),
        holdout_test_pos_count,
    )
    rows.extend(dzanga_neg_rows)

    print(f"Planned {len(dzanga_neg_rows)} neg clips for dzanga/holdout_test")

    # -----------------------
    # Write plan CSV
    # -----------------------
    plan = pd.DataFrame(rows)
    plan = plan.sort_values(["label", "split", "source_wav_relpath", "start_s"]).reset_index(drop=True)

    CLIPS_PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(CLIPS_PLAN_CSV, index=False)

    print(f"\nWrote plan: {CLIPS_PLAN_CSV}")
    print(f"Train/val positives:              {train_val_pos_count}")
    print(f"Holdout-test positives:           {holdout_test_pos_count}")
    print(f"Train/val negatives (PNNN):       {len(train_val_windows)}")
    print(f"Holdout-test negatives (Dzanga):  {len(dzanga_neg_rows)}")
    print(f"Total planned clips:              {len(plan)}")


if __name__ == "__main__":
    main()
