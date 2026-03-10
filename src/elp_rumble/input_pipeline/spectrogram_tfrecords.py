# src/elp_rumble/input_pipeline/spectrogram_tfrecords.py
import os
import tensorflow as tf
from elp_rumble.config.paths import TFRECORDS_SPECTROGRAM_DIR

H, W, C = 563, 98, 1
INPUT_SHAPE = (H, W, C)

def parse_spec_example(serialized_example):
    feature_desc = {
        "sample": tf.io.FixedLenFeature([], tf.string),
        "label":  tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(serialized_example, feature_desc)
    x = tf.io.parse_tensor(ex["sample"], out_type=tf.float32)
    x = tf.reshape(x, INPUT_SHAPE)
    y = tf.cast(ex["label"], tf.float32)
    y = tf.reshape(y, [1])
    return x, y

def make_ds(tfrecord_path, batch_size, shuffle=False, downsample_fraction=1, seed=None):
    ds = tf.data.TFRecordDataset([tfrecord_path], num_parallel_reads=tf.data.AUTOTUNE)

    # Randomly keep each example with probability = downsample_fraction
    if downsample_fraction < 1.0:
        def keep(_):
            rnd = tf.random.uniform((), seed=seed)  # per-element draw
            return rnd < downsample_fraction
        ds = ds.filter(keep)

    ds = ds.map(parse_spec_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_spec_paths():
    model = os.getenv("MODEL", "model3").strip()
    base = os.path.join(TFRECORDS_SPECTROGRAM_DIR, model)
    return {
        "train": os.path.join(base, "train.tfrecord"),
        "val":   os.path.join(base, "validate.tfrecord"),
        "test":  os.path.join(base, "test.tfrecord"),
    }