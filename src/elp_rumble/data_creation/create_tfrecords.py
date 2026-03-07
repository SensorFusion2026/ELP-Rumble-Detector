# src/elp_rumble/data_creation/create_tfrecords.py
# Usage:
#   python -m elp_rumble.data_creation.create_tfrecords
#
# Optional environment variables:
#   MODEL=model1|model3      (default: model3)
#   SPLIT_SEED=42            (used by create_splits.py)
#   TRAIN_FRAC=0.8           (used by create_splits.py)

from pathlib import Path
import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from elp_rumble.config.paths import WAV_CLIPS_ROOT, TFRECORDS_SPECTROGRAM_DIR

MODEL = os.getenv("MODEL", "model3").strip()
if MODEL not in {"model1", "model3"}:
    raise ValueError("Invalid MODEL. Use MODEL=model1 or MODEL=model3.")

SPLITS_CSV = Path(__file__).resolve().parent / "splits" / f"{MODEL}.csv"
OUT_DIR = TFRECORDS_SPECTROGRAM_DIR / MODEL if MODEL == "model1" else TFRECORDS_SPECTROGRAM_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"neg": 0, "pos": 1}

TARGET_SR = 4000
CLIP_LEN_S = 5.0
EXPECTED_SAMPLES = int(TARGET_SR * CLIP_LEN_S)

FRAME_LENGTH = 2000
FRAME_STEP = 32
MAX_FREQUENCY = 200


def _bytes_feature(x: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def _int64_feature(x: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x)]))


def wav_to_logspec(wav_path: str) -> tf.Tensor:
    wav_bytes = tf.io.read_file(wav_path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    sr = tf.cast(sr, tf.int32)

    if sr == 8000:
        audio = audio[::2]
        sr = 4000
    elif sr != 4000:
        raise ValueError(f"Unsupported sample rate {int(sr.numpy())} for {wav_path}")

    n = tf.shape(audio)[0]
    if n < EXPECTED_SAMPLES:
        audio = tf.pad(audio, [[0, EXPECTED_SAMPLES - n]])
    else:
        audio = audio[:EXPECTED_SAMPLES]

    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )

    freq_resolution = float(TARGET_SR) / float(FRAME_LENGTH)
    bins_to_grab = int(MAX_FREQUENCY / freq_resolution)
    stft = stft[:, 2:bins_to_grab]

    spec = tf.math.log(tf.abs(stft) + 1e-10)
    return tf.expand_dims(spec, axis=-1)


def serialize_example(spec: tf.Tensor, label: int, location: str, clip_rel: str) -> bytes:
    spec_bytes = tf.io.serialize_tensor(spec).numpy()
    features = {
        "sample": _bytes_feature(spec_bytes),
        "label": _int64_feature(label),
        "location": _bytes_feature(location.encode("utf-8")),
        "clip_wav_relpath": _bytes_feature(clip_rel.encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def compute_stats(entries: list[dict]) -> tuple[float, float]:
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    for entry in entries:
        spec = wav_to_logspec(entry["clip_abs_path"]).numpy()
        total_sum += float(np.sum(spec))
        total_sum_sq += float(np.sum(np.square(spec)))
        total_count += int(spec.size)

    if total_count == 0:
        raise ValueError("Cannot compute spectrogram normalization stats on empty split set.")

    mean = total_sum / total_count
    variance = max((total_sum_sq / total_count) - (mean * mean), 1e-12)
    std = float(np.sqrt(variance))
    return float(mean), float(std)


def write_split_tfrecord(entries: list[dict], output_path: Path, mean: float, std: float) -> int:
    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for entry in entries:
            spec = wav_to_logspec(entry["clip_abs_path"])
            spec = (spec - mean) / std
            writer.write(
                serialize_example(
                    spec,
                    entry["label_int"],
                    entry["location"],
                    entry["clip_wav_relpath"],
                )
            )
            count += 1
    return count


def main():
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(
            f"Missing split CSV: {SPLITS_CSV}. Run create_clips_plan.py, cut_wav_clips.py, and create_splits.py first."
        )

    df = pd.read_csv(SPLITS_CSV)
    required = {"split", "label", "location", "clip_wav_relpath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{SPLITS_CSV} missing columns: {sorted(missing)}")

    entries = []
    skipped_missing = 0
    for _, row in df.iterrows():
        split = str(row["split"])
        label = str(row["label"])
        if split not in {"train", "validate", "test"}:
            continue
        if label not in LABEL_MAP:
            continue

        clip_rel = str(row["clip_wav_relpath"])
        clip_abs = WAV_CLIPS_ROOT / clip_rel
        if not clip_abs.exists():
            skipped_missing += 1
            continue

        entries.append(
            {
                "split": split,
                "label": label,
                "label_int": LABEL_MAP[label],
                "location": str(row["location"]),
                "clip_wav_relpath": clip_rel,
                "clip_abs_path": str(clip_abs),
            }
        )

    if not entries:
        raise ValueError("No usable clip entries found from split CSV.")

    train_entries = [e for e in entries if e["split"] == "train"]
    validate_entries = [e for e in entries if e["split"] == "validate"]
    test_entries = [e for e in entries if e["split"] == "test"]

    mean, std = compute_stats(train_entries + validate_entries)

    counts = {
        "train": write_split_tfrecord(train_entries, OUT_DIR / "train.tfrecord", mean, std),
        "validate": write_split_tfrecord(validate_entries, OUT_DIR / "validate.tfrecord", mean, std),
        "test": write_split_tfrecord(test_entries, OUT_DIR / "test.tfrecord", mean, std),
    }

    manifest = OUT_DIR / "clip_splits.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "location", "clip_wav_relpath", "clip_abs_path"])
        for e in sorted(entries, key=lambda x: (x["split"], x["label"], x["clip_wav_relpath"])):
            writer.writerow([
                e["split"],
                e["label"],
                e["location"],
                e["clip_wav_relpath"],
                e["clip_abs_path"],
            ])

    print(f"TFRecords written to: {OUT_DIR}")
    print(f"Split CSV: {SPLITS_CSV}")
    print(f"Manifest: {manifest}")
    print(f"Counts: {counts}")
    print(f"Skipped missing clips: {skipped_missing}")
    print(f"Normalization stats: mean={mean:.6f}, std={std:.6f}")


if __name__ == "__main__":
    main()
