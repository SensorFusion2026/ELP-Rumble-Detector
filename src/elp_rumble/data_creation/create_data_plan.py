# src/elp_rumble/data_creation/create_data_plan.py
# Usage: python -m elp_rumble.data_creation.create_data_plan
#
# Plans all clips (positive + negative candidates),
# assigns WAVs to train/val/test splits, trims negatives to 3:1 per split,
# and writes clips_plan.csv + splits/model{1,2,3}.csv.

from pathlib import Path
import wave

import numpy as np
import pandas as pd

from .utils import validate_dir
from elp_rumble.config.paths import (
    RAW_ROOT,
    PNNN1_METADATA,
    PNNN1_SOUNDS,
    PNNN2_METADATA,
    PNNN2_SOUNDS,
    DZANGA_METADATA,
    DZANGA_SOUNDS,
    CLIPS_PLAN_CSV,
    SPLITS_DIR,
)

# ── Settings ─────────────────────────────────────────────────────────
CLIP_LEN_S = 5.0
BUFFER_S = 5.0
NEG_PER_POS = 3
SEED = 42
RNG = np.random.default_rng(SEED)

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test = remainder (0.1)

MODEL1_CAPS = {"pos": 50, "neg": 150}
MODEL2_FRAC = 0.5

SPLIT_NAMES = ["train", "val", "test"]

# Temporal segmentation for Dzanga only (huge multi-day WAVs).
# PNNN WAVs are small enough for whole-WAV grouping.
DZANGA_SEGMENT_S = 8 * 3600.0  # 8-hour segments


# ── Clip planning helpers ────────────────────────────────────────────

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


def _plan_positives(
    metadata_dir: Path, sounds_dir: Path, location: str, start_idx: int,
) -> tuple[list[dict], dict[str, list[tuple[float, float]]], int]:
    """Plan positive clips from a single metadata + sounds source.

    Returns:
        rows: list of clip plan dicts
        pos_windows_by_wav: {source_relpath: [(start, end), ...]}
        next_idx: next available clip index
    """
    rows: list[dict] = []
    pos_windows_by_wav: dict[str, list[tuple[float, float]]] = {}
    idx = start_idx

    meta_files = sorted(p for p in metadata_dir.iterdir() if p.suffix == ".txt")
    for meta_file in meta_files:
        data = pd.read_csv(meta_file, delimiter="\t")

        if "Tag 1" in data.columns:
            data = data[~data["Tag 1"].isin(["DUMMY_NoEles", "DUMMY_noEles"])]

        has_begin_end = {"Begin Time (s)", "End Time (s)"}.issubset(data.columns)

        # Build forbidden windows from ALL non-DUMMY annotations (including
        # faint/marginal/gorilla) so negatives never overlap ambiguous rumbles.
        for _, clip in data.iterrows():
            begin_file = str(clip["Begin File"])
            wav_path = _resolve_wav_path(sounds_dir, begin_file, location)
            if wav_path is None:
                continue

            offset = float(clip["File Offset (s)"])
            source_rel = _source_relpath(wav_path)

            if has_begin_end:
                duration = float(clip["End Time (s)"]) - float(clip["Begin Time (s)"])
            else:
                duration = CLIP_LEN_S

            pos_windows_by_wav.setdefault(source_rel, []).append(
                (offset, offset + duration)
            )

        # Filter to confident annotations for positive clips only.
        if "notes" in data.columns:
            data = data[~data["notes"].astype(str).str.contains("faint|marginal|gorilla", case=False, na=False)]

        for _, clip in data.iterrows():
            begin_file = str(clip["Begin File"])
            wav_path = _resolve_wav_path(sounds_dir, begin_file, location)
            if wav_path is None:
                continue

            offset = float(clip["File Offset (s)"])
            source_rel = _source_relpath(wav_path)
            clip_rel = f"pos/{wav_path.stem}_pos_{int(round(offset))}_{idx}.wav"
            rows.append({
                "label": "pos",
                "location": location,
                "source_wav_relpath": source_rel,
                "start_s": offset,
                "duration_s": float(CLIP_LEN_S),
                "clip_wav_relpath": clip_rel,
            })
            idx += 1

    return rows, pos_windows_by_wav, idx


def _generate_neg_candidates(
    pos_windows_by_wav: dict[str, list[tuple[float, float]]],
    location: str,
    start_idx: int,
) -> tuple[list[dict], int]:
    """Generate ALL negative candidate clips via buffered exclusion.

    Unlike the old _plan_negatives_buffered(), this keeps every candidate
    (no per-WAV cap). Trimming to 3:1 happens later in _trim_negatives().
    """
    rows: list[dict] = []
    idx = start_idx

    for source_rel in sorted(pos_windows_by_wav):
        windows = pos_windows_by_wav[source_rel]

        wav_abs = RAW_ROOT / source_rel
        if not wav_abs.exists():
            print(f"WARNING: WAV not found, skipping negatives: {wav_abs}")
            continue

        with wave.open(str(wav_abs), "rb") as w:
            sr = w.getframerate()
            total_s = w.getnframes() / sr

        # Forbidden zones: annotation span ± buffer
        forbidden = [
            (max(0.0, begin - BUFFER_S), min(total_s, end + BUFFER_S))
            for begin, end in windows
        ]
        forbidden.sort()

        # Merge overlapping forbidden zones
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
            print(f"WARNING: No allowed intervals for negatives in {source_rel}")
            continue

        # Discretize allowed intervals into clip-length candidate starts
        wav_stem = Path(source_rel).stem
        for a, b in allowed:
            s = a
            while s + CLIP_LEN_S <= b:
                rows.append({
                    "label": "neg",
                    "location": location,
                    "source_wav_relpath": source_rel,
                    "start_s": float(s),
                    "duration_s": float(CLIP_LEN_S),
                    "clip_wav_relpath": f"neg/{wav_stem}_neg_{int(s * 1000)}_{idx}.wav",
                })
                idx += 1
                s += CLIP_LEN_S

    return rows, idx


# ── Split helpers ────────────────────────────────────────────────────

def _split_by_wav(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test splits via WAV-level grouping.

    Dzanga WAVs are huge multi-day recordings, so they are subdivided
    into temporal segments (DZANGA_SEGMENT_S) for finer-grained split
    assignment.  PNNN WAVs are small enough to keep as whole-WAV groups.

    Algorithm:
      1. Seed: one group per (location, split) — smallest groups by pos count,
         ensuring every location appears in every split.
      2. Greedy LPT by positive count: remaining groups sorted by pos count
         descending → assign to split with greatest positive deficit.
    """
    targets = {"train": TRAIN_FRAC, "val": VAL_FRAC,
               "test": 1.0 - TRAIN_FRAC - VAL_FRAC}

    # Compute group key:
    #   Dzanga  → WAV path + temporal segment (large multi-day recordings)
    #   PNNN   → WAV path only (whole-WAV grouping)
    df = df.copy()
    df["_group"] = df.apply(
        lambda r: (
            r["source_wav_relpath"] + "::" + str(int(r["start_s"] // DZANGA_SEGMENT_S))
            if r["location"] == "dzanga"
            else r["source_wav_relpath"]
        ),
        axis=1,
    )

    group_pos = df[df["label"] == "pos"].groupby("_group").size()
    group_total = df.groupby("_group").size().to_dict()
    group_location = df.groupby("_group")["location"].first().to_dict()

    total_pos = len(df[df["label"] == "pos"])
    target_pos = {s: targets[s] * total_pos for s in SPLIT_NAMES}

    group_to_split: dict[str, str] = {}
    cur_pos: dict[str, float] = {s: 0 for s in SPLIT_NAMES}

    def _pos_deficit(s: str) -> float:
        return (target_pos[s] - cur_pos[s]) / total_pos if total_pos else 0

    def _assign(g: str, s: str) -> None:
        group_to_split[g] = s
        cur_pos[s] += group_pos.get(g, 0)

    # Phase 1: Seed — one group per (location, split)
    for location in sorted(df["location"].unique()):
        loc_groups = sorted(
            [g for g in group_total if group_location[g] == location],
            key=lambda g: group_pos.get(g, 0),
        )
        if len(loc_groups) < 3:
            for i, g in enumerate(loc_groups):
                _assign(g, SPLIT_NAMES[i % 3])
        else:
            for g, s in zip(loc_groups[:3], ["test", "val", "train"]):
                _assign(g, s)

    # Phase 2: Greedy LPT by positive count
    remaining = [g for g in group_total if g not in group_to_split]
    remaining_arr = np.array(remaining)
    RNG.shuffle(remaining_arr)
    remaining = sorted(remaining_arr,
                       key=lambda g: group_pos.get(g, 0), reverse=True)

    for g in remaining:
        best = max(SPLIT_NAMES, key=_pos_deficit)
        _assign(g, best)

    df["split"] = df["_group"].map(group_to_split)
    df = df.drop(columns=["_group"])
    return df


def _trim_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Trim negatives per split to exactly NEG_PER_POS:1 ratio.

    For each split: keep all positives, randomly sample up to
    NEG_PER_POS × n_pos negatives. If a split has fewer candidates
    than the target, keep all and print a warning.
    """
    parts: list[pd.DataFrame] = []
    for split in SPLIT_NAMES:
        sub = df[df["split"] == split]
        pos = sub[sub["label"] == "pos"]
        neg = sub[sub["label"] == "neg"]

        target_neg = NEG_PER_POS * len(pos)

        if len(neg) > target_neg:
            neg = neg.sample(n=target_neg, random_state=SEED)
        elif len(neg) < target_neg:
            print(
                f"WARNING: {split} has only {len(neg)} neg candidates "
                f"(wanted {target_neg}). Keeping all."
            )

        parts.append(pos)
        parts.append(neg)

    result = pd.concat(parts, ignore_index=True)
    # Shuffle clips within each split to avoid ordering bias
    result = result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return result


def _downsample_model(
    split_df: pd.DataFrame,
    frac: float | None = None,
    caps: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Downsample an already-split dataset per (split, label).

    Exactly one of *frac* or *caps* must be provided:
      - frac: keep this fraction of each (split, label) group.
      - caps: {"pos": N, "neg": M} total caps distributed across splits
              proportionally to TRAIN_FRAC / VAL_FRAC / TEST_FRAC.

    Preserves split ratios and neg:pos balance.
    """
    if (frac is None) == (caps is None):
        raise ValueError("Exactly one of frac or caps must be provided")

    split_fracs = {"train": TRAIN_FRAC, "val": VAL_FRAC,
                   "test": 1.0 - TRAIN_FRAC - VAL_FRAC}

    parts: list[pd.DataFrame] = []
    for split in SPLIT_NAMES:
        sub = split_df[split_df["split"] == split]
        for label in ["pos", "neg"]:
            chunk = sub[sub["label"] == label]
            if chunk.empty:
                continue

            if caps is not None:
                n = max(1, round(split_fracs[split] * caps[label]))
            else:
                n = max(1, round(frac * len(chunk)))

            if len(chunk) > n:
                chunk = chunk.sample(n=n, random_state=SEED)
            parts.append(chunk)

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ── Output helpers ───────────────────────────────────────────────────

def _write_split(df: pd.DataFrame, path):
    cols = ["split", "label", "location", "source_wav_relpath", "clip_wav_relpath"]
    df = df[cols].sort_values(cols)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def _summarize(df, name):
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(df.groupby(["split", "label", "location"]).size().to_string())
    print()
    split_label = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("Per-split label totals:")
    print(split_label.to_string())
    print()
    total = len(df)
    print("Per-split stats:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_pos = len(sub[sub["label"] == "pos"])
        n_neg = len(sub[sub["label"] == "neg"])
        ratio = n_neg / n_pos if n_pos else float("inf")
        pct = 100 * len(sub) / total if total else 0
        print(f"  {split:5s}: {len(sub):6d} clips ({pct:5.1f}%)  "
              f"neg:pos = {ratio:.2f}:1  ({n_neg} neg / {n_pos} pos)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Safety check: clip plan overwrite
    if CLIPS_PLAN_CSV.exists():
        raise RuntimeError(
            f"\n\u26a0\ufe0f  clips_plan.csv already exists at:\n"
            f"    {CLIPS_PLAN_CSV}\n\n"
            f"This file is the source of truth for clip generation and is shared\n"
            f"across the team. Re-creating it will require everyone to re-cut\n"
            f"WAV clips and rebuild downstream artifacts (splits, TFRecords).\n\n"
            f"If you REALLY intend to regenerate the clip plan:\n"
            f"  1) Delete this file manually\n"
            f"  2) Inform the rest of the team\n"
            f"  3) Re-run create_data_plan.py\n"
            f"  4) Re-generate WAV clips, split files, and TFRecords\n"
        )

    # Validate directories
    datasets = [
        ("pnnn1", PNNN1_METADATA, PNNN1_SOUNDS),
        ("pnnn2", PNNN2_METADATA, PNNN2_SOUNDS),
        ("dzanga", DZANGA_METADATA, DZANGA_SOUNDS),
    ]
    for loc, meta, sounds in datasets:
        validate_dir(meta, f"{loc.upper()}_METADATA")
        validate_dir(sounds, f"{loc.upper()}_SOUNDS")

    # ── Step 1+2: Plan all positives and ALL negative candidates ─────
    all_pos: list[dict] = []
    all_neg: list[dict] = []
    pos_idx = 1
    neg_idx = 1

    for location, metadata_dir, sounds_dir in datasets:
        pos_rows, pos_windows_by_wav, pos_idx = _plan_positives(
            Path(metadata_dir), Path(sounds_dir), location, pos_idx,
        )
        all_pos.extend(pos_rows)
        print(f"Planned {len(pos_rows)} pos clips for {location}")

        neg_rows, neg_idx = _generate_neg_candidates(
            pos_windows_by_wav, location, neg_idx,
        )
        all_neg.extend(neg_rows)
        print(f"Generated {len(neg_rows)} neg candidates for {location}")

    all_candidates = pd.DataFrame(all_pos + all_neg)
    total_pos = len(all_pos)
    total_neg_candidates = len(all_neg)
    print(f"\nTotal positives:       {total_pos}")
    print(f"Total neg candidates:  {total_neg_candidates}")
    print(f"Candidate ratio:       {total_neg_candidates / total_pos:.1f}:1")

    # ── Step 3+4: Build model datasets (split + trim) ───────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    model3_split = _split_by_wav(all_candidates)
    model3 = _trim_negatives(model3_split)
    model2 = _downsample_model(model3, frac=MODEL2_FRAC)
    model1 = _downsample_model(model3, caps=MODEL1_CAPS)

    # ── Step 5: Write model CSVs ────────────────────────────────────
    _write_split(model1, SPLITS_DIR / "model1.csv")
    _write_split(model2, SPLITS_DIR / "model2.csv")
    _write_split(model3, SPLITS_DIR / "model3.csv")

    # ── Step 6: Write clips_plan.csv (model3 is the superset) ───────
    plan_cols = ["label", "location", "source_wav_relpath", "start_s",
                 "duration_s", "clip_wav_relpath"]
    # Recover start_s and duration_s from the candidate pool
    candidate_lookup = all_candidates.set_index("clip_wav_relpath")[["start_s", "duration_s"]]
    plan = model3.drop(columns=["start_s", "duration_s"], errors="ignore").copy()
    plan = plan.join(candidate_lookup, on="clip_wav_relpath")

    plan = plan[plan_cols].sort_values(
        ["label", "location", "source_wav_relpath", "start_s"]
    ).reset_index(drop=True)

    CLIPS_PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(CLIPS_PLAN_CSV, index=False)

    total_selected_neg = len(plan[plan["label"] == "neg"])
    total_selected_pos = len(plan[plan["label"] == "pos"])
    ratio = total_selected_neg / total_selected_pos if total_selected_pos else float("inf")
    print(f"\nWrote clips plan: {CLIPS_PLAN_CSV}")
    print(f"Selected positives: {total_selected_pos}")
    print(f"Selected negatives: {total_selected_neg}")
    print(f"Overall neg:pos:    {ratio:.2f}:1")
    print(f"Total clips to cut: {len(plan)}")

    # ── Summaries ───────────────────────────────────────────────────
    _summarize(model1, "MODEL 1 (Feasibility)")
    _summarize(model2, "MODEL 2 (Scaled)")
    _summarize(model3, "MODEL 3 (Performance)")


if __name__ == "__main__":
    main()
