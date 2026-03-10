# src/elp_rumble/data_creation/create_tfrecords.py
# Usage:
#   python -m elp_rumble.data_creation.create_tfrecords
#
# This script writes BOTH:
# 1) normalized audio TFRecords (for RNN)
# 2) normalized spectrogram TFRecords (for CNN)
# for all split definitions: model1, model2, and model3.

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import compute_statistics, load_wav_file, write_tfrecords
from elp_rumble.config.paths import (
    TFRECORDS_AUDIO_DIR,
    TFRECORDS_SPECTROGRAM_DIR,
    SPLITS_DIR,
    WAV_CLIPS_ROOT,
)
MODELS = ("model1", "model2", "model3")

LABEL_MAP = {"neg": 0, "pos": 1}
SEED = 42

# Spectrogram parameters (keep aligned with model input pipeline)
FRAME_LENGTH = 2000
FRAME_STEP = 32
SAMPLE_RATE = 4000
MAX_FREQUENCY = 200


def _dataset_from_entries(entries):
    if not entries:
        raise ValueError("Split has no entries; cannot create TFRecord dataset.")

    paths = [e["clip_abs_path"] for e in entries]
    labels = [e["label_int"] for e in entries]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(
        lambda path, label: (load_wav_file(path), tf.cast(label, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _build_entries(df):
    entries = []
    skipped_missing = 0

    for _, row in df.iterrows():
        split = str(row["split"])
        label = str(row["label"])
        if split not in {"train", "val", "test"}:
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

    return entries, skipped_missing


def _stft_hann_window(audio, frame_length, frame_step, bins_to_grab):
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hann_window,
    )
    stft = stft[:, 2:bins_to_grab]
    stft = tf.math.log(tf.abs(stft) + 1e-10)
    return tf.expand_dims(stft, axis=-1)


def _apply_stft(dataset, frame_length, frame_step, sample_rate, max_frequency):
    freq_resolution = sample_rate / frame_length
    bins_to_grab = int(max_frequency / freq_resolution)
    return dataset.map(
        lambda audio, label: (_stft_hann_window(audio, frame_length, frame_step, bins_to_grab), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _compute_spec_stats(datasets):
    all_spectrograms = []
    for dataset in datasets:
        for spectrogram, _ in dataset:
            all_spectrograms.append(spectrogram.numpy())

    if not all_spectrograms:
        raise ValueError("Cannot compute spectrogram normalization stats from empty datasets.")

    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    global_mean = np.mean(all_spectrograms)
    global_std = np.std(all_spectrograms)
    return float(global_mean), float(global_std)


def main():
    tf.config.set_visible_devices([], "GPU")
    print(tf.config.list_physical_devices())

    for model in MODELS:
        _create_tfrecords_for_model(model)


def _create_tfrecords_for_model(model):
    splits_csv = SPLITS_DIR / f"{model}.csv"
    out_audio_dir = TFRECORDS_AUDIO_DIR / model
    out_spec_dir = TFRECORDS_SPECTROGRAM_DIR / model
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_spec_dir.mkdir(parents=True, exist_ok=True)

    if not splits_csv.exists():
        raise FileNotFoundError(
            f"Missing split CSV: {splits_csv}. Run create_data_plan.py and cut_wav_clips.py first."
        )

    df = pd.read_csv(splits_csv)
    required = {"split", "label", "location", "clip_wav_relpath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{splits_csv} missing columns: {sorted(missing)}")

    entries, skipped_missing = _build_entries(df)
    if not entries:
        raise ValueError("No usable clip entries found from split CSV.")

    rng = random.Random(SEED)

    train_entries = [e for e in entries if e["split"] == "train"]
    val_entries = [e for e in entries if e["split"] == "val"]
    test_entries = [e for e in entries if e["split"] == "test"]

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    rng.shuffle(test_entries)

    # -------- AUDIO TFRECORDS (RNN) --------
    train_audio = _dataset_from_entries(train_entries)
    val_audio = _dataset_from_entries(val_entries)
    test_audio = _dataset_from_entries(test_entries)

    audio_stats_dataset = train_audio.map(lambda x, _: x)
    audio_mean, audio_std = compute_statistics(audio_stats_dataset)

    train_audio = train_audio.map(
        lambda audio, label: ((audio - audio_mean) / audio_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(20000, reshuffle_each_iteration=False)

    val_audio = val_audio.map(
        lambda audio, label: ((audio - audio_mean) / audio_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_audio = test_audio.map(
        lambda audio, label: ((audio - audio_mean) / audio_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(10000, reshuffle_each_iteration=False)

    write_tfrecords(train_audio, os.path.join(out_audio_dir, "train"))
    write_tfrecords(val_audio, os.path.join(out_audio_dir, "validate"))
    write_tfrecords(test_audio, os.path.join(out_audio_dir, "test"))

    # -------- SPECTROGRAM TFRECORDS (CNN) --------
    train_spec = _apply_stft(train_audio, FRAME_LENGTH, FRAME_STEP, SAMPLE_RATE, MAX_FREQUENCY)
    val_spec = _apply_stft(val_audio, FRAME_LENGTH, FRAME_STEP, SAMPLE_RATE, MAX_FREQUENCY)
    test_spec = _apply_stft(test_audio, FRAME_LENGTH, FRAME_STEP, SAMPLE_RATE, MAX_FREQUENCY)

    spec_mean, spec_std = _compute_spec_stats([train_spec])

    train_spec = train_spec.map(
        lambda spec, label: ((spec - spec_mean) / spec_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_spec = val_spec.map(
        lambda spec, label: ((spec - spec_mean) / spec_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_spec = test_spec.map(
        lambda spec, label: ((spec - spec_mean) / spec_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    write_tfrecords(train_spec, os.path.join(out_spec_dir, "train"))
    write_tfrecords(val_spec, os.path.join(out_spec_dir, "validate"))
    write_tfrecords(test_spec, os.path.join(out_spec_dir, "test"))

    print(f"Model: {model}")
    print(f"Split CSV: {splits_csv}")
    print(f"Counts: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
    print(f"Skipped missing clips: {skipped_missing}")
    print(f"Audio TFRecords written to: {out_audio_dir}")
    print(f"Audio normalization stats: mean={audio_mean:.6f}, std={audio_std:.6f}")
    print(f"Spectrogram TFRecords written to: {out_spec_dir}")
    print(f"Spectrogram normalization stats: mean={spec_mean:.6f}, std={spec_std:.6f}")
    print("-")


if __name__ == "__main__":
    main()
