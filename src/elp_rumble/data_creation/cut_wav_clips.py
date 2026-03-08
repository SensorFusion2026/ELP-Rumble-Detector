# src/elp_rumble/data_creation/cut_wav_clips.py
# Usage: python -m elp_rumble.data_creation.cut_wav_clips

from pathlib import Path
import wave
import numpy as np
import pandas as pd

from .utils import apply_low_pass_filter, down_sample, save_audio_to_wav
from elp_rumble.config.paths import RAW_ROOT, WAV_CLIPS_ROOT, CLIPS_PLAN_CSV

TARGET_SR = 4000
CUTOFF_HZ = 200


def _dtype_for_width(sample_width: int):
    if sample_width == 1:
        return np.int8
    if sample_width == 2:
        return np.int16
    if sample_width == 4:
        return np.int32
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def main():
    if RAW_ROOT is None:
        raise ValueError("RAW_ROOT is not configured. Set ENVIRONMENT=local and CORNELL_DATA_ROOT.")
    if not CLIPS_PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing plan file: {CLIPS_PLAN_CSV}. Run create_clips_plan.py first.")

    plan = pd.read_csv(CLIPS_PLAN_CSV)
    required = {"source_wav_relpath", "start_s", "duration_s", "clip_wav_relpath"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"clips_plan.csv missing columns: {sorted(missing)}")

    WAV_CLIPS_ROOT.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_exists = 0
    skipped_missing = 0
    skipped_short = 0
    skipped_bad_format = 0

    for _, row in plan.iterrows():
        source_rel = str(row["source_wav_relpath"])
        start_s = float(row["start_s"])
        duration_s = float(row["duration_s"])

        src_path = Path(RAW_ROOT) / source_rel
        out_path = WAV_CLIPS_ROOT / str(row["clip_wav_relpath"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            skipped_exists += 1
            continue
        if not src_path.exists():
            skipped_missing += 1
            continue

        try:
            with wave.open(str(src_path), "rb") as w:
                sr = w.getframerate()
                sample_width = w.getsampwidth()
                channels = w.getnchannels()
                start_frame = int(start_s * sr)
                n_frames = int(duration_s * sr)

                try:
                    w.setpos(start_frame)
                except wave.Error:
                    skipped_short += 1
                    continue

            frames = w.readframes(n_frames)
        except wave.Error:
            skipped_bad_format += 1
            continue

        dtype = _dtype_for_width(sample_width)
        data = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            data = data.reshape((-1, channels)).mean(axis=1)

        if len(data) < n_frames:
            skipped_short += 1
            continue

        expected_frames = int(duration_s * TARGET_SR)
        data = apply_low_pass_filter(data, sr, cutoff_hz=CUTOFF_HZ)
        data = down_sample(data, sr, TARGET_SR, expected_frames)

        if len(data) != expected_frames:
            skipped_short += 1
            continue

        save_audio_to_wav(str(out_path), data, TARGET_SR, 1, 2)
        saved += 1

    print("\n=== Done cutting clips ===")
    print(f"Saved:                {saved}")
    print(f"Skipped (exists):     {skipped_exists}")
    print(f"Skipped (missing):    {skipped_missing}")
    print(f"Skipped (short):      {skipped_short}")
    print(f"Skipped (bad format): {skipped_bad_format}")
    print(f"Output root:          {WAV_CLIPS_ROOT}")


if __name__ == "__main__":
    main()
