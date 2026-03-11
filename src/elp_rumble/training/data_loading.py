# training/data_loading.py
"""Shared TFRecord loading utilities for CNN and RNN training scripts."""

import tensorflow as tf

# ── Shape constants ──────────────────────────────────────────────────────────
SPEC_SHAPE = (563, 98, 1)
AUDIO_LENGTH = 20000  # 5 s @ 4 kHz


# ── Generic helpers ──────────────────────────────────────────────────────────

def count_examples(path: str) -> int:
    """Iterate a TFRecord file and return the total number of records."""
    return sum(1 for _ in tf.data.TFRecordDataset([path]))


def make_ds(path, parse_fn, batch_size, shuffle=False, drop_remainder=False):
    """
    Wrap a TFRecord file as a batched tf.data.Dataset.

    Args:
        path: Path to a .tfrecord file.
        parse_fn: Callable (serialized_example) -> (features, label) or
                  (features, label, clip_id).
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle (buffer = 10 000).
        drop_remainder: Whether to drop the final partial batch.

    Returns:
        A prefetched tf.data.Dataset.
    """
    ds = tf.data.TFRecordDataset([path], num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_class_weights(train_path: str, parse_fn) -> dict:
    """
    Compute inverse-frequency class weights from the training set.

    Returns:
        {0: w0, 1: w1} where w_i = n_total / (2 * n_i).
    """
    n0, n1 = 0, 0
    ds = tf.data.TFRecordDataset([train_path])
    for serialized in ds:
        _, label = parse_fn(serialized)
        if int(label.numpy().flat[0]) == 1:
            n1 += 1
        else:
            n0 += 1
    n_total = n0 + n1
    if n_total == 0:  
        raise ValueError(  
            f"Cannot compute class weights: training TFRecord '{train_path}' "
            "contains no examples."
        )  
    if n0 == 0 or n1 == 0:
        raise ValueError(
            f"Cannot compute class weights from '{train_path}': "
            f"class counts are n0={n0}, n1={n1}. Both classes must be present."
        )
    return {0: n_total / (2.0 * n0), 1: n_total / (2.0 * n1)}


# ── Unified parse function ───────────────────────────────────────────────────

def parse_tfrecord_example(serialized, data_type="spectrogram", clip_id=False):
    """
    Parse a single TFRecord example.

    Args:
        serialized: A scalar string tensor (one raw TFRecord example).
        data_type: "spectrogram" → reshapes to SPEC_SHAPE (563, 98, 1).
              "audio"       → reshapes to [AUDIO_LENGTH] (20000,).
        clip_id: If True, also returns the "clip_wav_relpath" feature as a
                 third element. Records without that feature return b"".

    Returns:
        (x, y)          when clip_id=False
        (x, y, clip_id) when clip_id=True
    """
    feature_desc = {
        "sample": tf.io.FixedLenFeature([], tf.string),
        "label":  tf.io.FixedLenFeature([], tf.int64),
    }
    if clip_id:
        feature_desc["clip_wav_relpath"] = tf.io.FixedLenFeature([], tf.string, default_value="")

    ex = tf.io.parse_single_example(serialized, feature_desc)

    x = tf.io.parse_tensor(ex["sample"], out_type=tf.float32)
    if data_type == "spectrogram":
        x = tf.reshape(x, SPEC_SHAPE)
    elif data_type == "audio":
        x = tf.reshape(x, [AUDIO_LENGTH])
    else:
        raise ValueError(f"Unknown data_type: {data_type!r}. Use 'spectrogram' or 'audio'.")

    y = tf.cast(ex["label"], tf.float32)
    y = tf.reshape(y, [1])

    if clip_id:
        return x, y, ex["clip_wav_relpath"]
    return x, y
