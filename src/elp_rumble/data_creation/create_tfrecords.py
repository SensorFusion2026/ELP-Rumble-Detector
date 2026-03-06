import os
import csv
import random
import tensorflow as tf
from functools import reduce
from .utils import compute_statistics, load_wav_file, write_tfrecords
from elp_rumble.config.paths import (
    POS_TRAIN_VAL_CLIPS_DIR,
    TRAIN_VAL_NEG_CLIPS_DIR,
    POS_HOLDOUT_TEST_CLIPS_DIR,
    HOLDOUT_TEST_NEG_CLIPS_DIR,
    TFRECORDS_AUDIO_DIR,
)


def _dataset_from_entries(entries):
    if not entries:
        raise ValueError("Split has no entries; cannot create TFRecord dataset.")

    paths = [e["file_path"] for e in entries]
    labels = [e["label"] for e in entries]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(
        lambda path, label: (load_wav_file(path), tf.cast(label, tf.int64)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def main():
    tf.config.set_visible_devices([], "GPU")
    print(tf.config.list_physical_devices())

    train_frac = float(os.getenv("TRAIN_FRAC", "0.8"))
    seed = int(os.getenv("SPLIT_SEED", "42"))
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"Invalid TRAIN_FRAC={train_frac}. Require 0 < TRAIN_FRAC < 1.")

    output_audio_folder = TFRECORDS_AUDIO_DIR
    os.makedirs(output_audio_folder, exist_ok=True)

    rng = random.Random(seed)

    # Positive clips define target counts for each split.
    seen_pos = tf.io.gfile.glob(os.path.join(POS_TRAIN_VAL_CLIPS_DIR, "*.wav"))
    holdout_pos = tf.io.gfile.glob(os.path.join(POS_HOLDOUT_TEST_CLIPS_DIR, "*.wav"))
    seen_neg = tf.io.gfile.glob(os.path.join(TRAIN_VAL_NEG_CLIPS_DIR, "*.wav"))
    holdout_neg = tf.io.gfile.glob(os.path.join(HOLDOUT_TEST_NEG_CLIPS_DIR, "*.wav"))

    if not seen_pos or not holdout_pos or not seen_neg or not holdout_neg:
        raise ValueError(
            "Missing clip files. Run positive/negative clip generation before create_tfrecords.py"
        )

    seen_pos = list(seen_pos)
    holdout_pos = list(holdout_pos)
    seen_neg = list(seen_neg)
    holdout_neg = list(holdout_neg)

    rng.shuffle(seen_pos)
    rng.shuffle(seen_neg)
    rng.shuffle(holdout_neg)

    n_seen_pos = len(seen_pos)
    n_train_pos = int(train_frac * n_seen_pos)
    n_validate_pos = n_seen_pos - n_train_pos

    train_pos = seen_pos[:n_train_pos]
    validate_pos = seen_pos[n_train_pos:]

    # Derive seen negative counts from seen positive counts.
    needed_seen_neg = n_train_pos + n_validate_pos
    if len(seen_neg) < needed_seen_neg:
        raise ValueError(
            f"Insufficient seen negatives ({len(seen_neg)}) for required seen positives ({needed_seen_neg})."
        )
    train_neg = seen_neg[:n_train_pos]
    validate_neg = seen_neg[n_train_pos:needed_seen_neg]

    # Holdout test uses all Dzanga positives + matched count of holdout negatives.
    n_holdout_pos = len(holdout_pos)
    if len(holdout_neg) < n_holdout_pos:
        raise ValueError(
            f"Insufficient holdout negatives ({len(holdout_neg)}) for holdout positives ({n_holdout_pos})."
        )
    test_neg = holdout_neg[:n_holdout_pos]

    entries = []

    def add_entries(paths, split, label, location):
        for path in paths:
            entries.append(
                {
                    "split": split,
                    "label": label,
                    "location": location,
                    "file_path": path,
                    "file_name": os.path.basename(path),
                }
            )

    add_entries(train_pos, "train", 1, "pnnn")
    add_entries(train_neg, "train", 0, "pnnn_neg")
    add_entries(validate_pos, "validate", 1, "pnnn")
    add_entries(validate_neg, "validate", 0, "pnnn_neg")
    add_entries(holdout_pos, "test", 1, "dzanga")
    add_entries(test_neg, "test", 0, "pnnn_neg")

    manifest_path = os.path.join(output_audio_folder, "clip_splits.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "location", "file_path", "file_name"])
        for entry in sorted(entries, key=lambda e: (e["split"], e["label"], e["file_name"])):
            writer.writerow([
                entry["split"],
                entry["label"],
                entry["location"],
                entry["file_path"],
                entry["file_name"],
            ])
    print(f"Saved clip split manifest to {manifest_path}")

    train_entries = [e for e in entries if e["split"] == "train"]
    validate_entries = [e for e in entries if e["split"] == "validate"]
    test_entries = [e for e in entries if e["split"] == "test"]

    split_sizes = {
        "train": len(train_entries),
        "validate": len(validate_entries),
        "test": len(test_entries),
    }
    total = sum(split_sizes.values())
    print(
        "Split counts: "
        f"train={split_sizes['train']} ({split_sizes['train'] / total:.3f}), "
        f"validate={split_sizes['validate']} ({split_sizes['validate'] / total:.3f}), "
        f"test={split_sizes['test']} ({split_sizes['test'] / total:.3f})"
    )

    train_dataset = _dataset_from_entries(train_entries)
    validate_dataset = _dataset_from_entries(validate_entries)
    test_dataset = _dataset_from_entries(test_entries)

    # Compute normalization from seen data only (train + validate), not holdout test.
    combined_dataset = reduce(
        lambda d1, d2: d1.concatenate(d2),
        [
            train_dataset.map(lambda x, _: x),
            validate_dataset.map(lambda x, _: x),
        ],
    )
    global_mean, global_std = compute_statistics(combined_dataset)
    del combined_dataset

    train_dataset = train_dataset.map(
        lambda audio, label: ((audio - global_mean) / global_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(20000, reshuffle_each_iteration=False)

    validate_dataset = validate_dataset.map(
        lambda audio, label: ((audio - global_mean) / global_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_dataset = test_dataset.map(
        lambda audio, label: ((audio - global_mean) / global_std, label),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(10000, reshuffle_each_iteration=False)

    write_tfrecords(train_dataset, os.path.join(output_audio_folder, "train"))
    write_tfrecords(validate_dataset, os.path.join(output_audio_folder, "validate"))
    write_tfrecords(test_dataset, os.path.join(output_audio_folder, "test"))


if __name__ == "__main__":
    main()