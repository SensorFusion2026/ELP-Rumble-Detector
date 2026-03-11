# training/train_rnn.py
"""RNN training pipeline for the ELP Rumble Detector."""

import csv
import functools
import json
import os
from datetime import datetime

import tensorflow as tf

from elp_rumble.config.paths import RUNS_DIR, TFRECORDS_AUDIO_DIR
from elp_rumble.models.rnn import RNN
from elp_rumble.training.data_loading import (
    count_examples,
    get_class_weights,
    make_ds,
    parse_tfrecord_example,
)

# ── GPU / mixed precision ────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"[train_rnn] GPU detected: {[g.name for g in gpus]} — mixed precision enabled.")
else:
    print("[train_rnn] No GPU detected — running on CPU.")

# ── Module-level constants ───────────────────────────────────────────────────
MODEL = os.getenv("MODEL", "model3")
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", 50))
LEARNING_RATE = 1e-4
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 0.97
DROPOUT_RATE = 0.5
SEQUENCE_LENGTH = 20000  # 5 s @ 4 kHz

_audio_dir = TFRECORDS_AUDIO_DIR / MODEL
TRAIN_PATH = str(_audio_dir / "train.tfrecord")
VAL_PATH   = str(_audio_dir / "validate.tfrecord")
TEST_PATH  = str(_audio_dir / "test.tfrecord")

# ── Parse-function shorthands ─────────────────────────────────────────────────
_parse = functools.partial(parse_tfrecord_example, type="audio", clip_id=False)
_parse_with_id = functools.partial(parse_tfrecord_example, type="audio", clip_id=True)


def main():
    # ── Run directory ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{MODEL}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_e{EPOCHS}_{ts}"
    export_dir = RUNS_DIR / "rnn" / run_name
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_rnn] Run directory: {export_dir}")
    print(f"[train_rnn] Model split  : {MODEL}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = make_ds(TRAIN_PATH, _parse, BATCH_SIZE, shuffle=True)
    val_ds   = make_ds(VAL_PATH,   _parse, BATCH_SIZE, shuffle=False)

    # ── Class weights ────────────────────────────────────────────────────────
    print("[train_rnn] Computing class weights…")
    class_weights = get_class_weights(TRAIN_PATH, _parse)
    print(f"  class_weights = {class_weights}")

    # ── params.json (before training) ────────────────────────────────────────
    params = {
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lr_decay_steps": LR_DECAY_STEPS,
        "lr_decay_rate": LR_DECAY_RATE,
        "dropout_rate": DROPOUT_RATE,
        "sequence_length": SEQUENCE_LENGTH,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "tfrecord_train": TRAIN_PATH,
        "tfrecord_val":   VAL_PATH,
        "tfrecord_test":  TEST_PATH,
    }
    with open(export_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # ── Build model ──────────────────────────────────────────────────────────
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=LR_DECAY_STEPS,
        decay_rate=LR_DECAY_RATE,
    )
    model = RNN(sequence_length=SEQUENCE_LENGTH, dropout_rate=DROPOUT_RATE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.build((None, SEQUENCE_LENGTH))
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(export_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(export_dir / "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(export_dir / "logs")),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ── Save final model ─────────────────────────────────────────────────────
    model.save(str(export_dir / "final_model.keras"))
    print("[train_rnn] Final model saved.")

    # ── Test evaluation (best model) ─────────────────────────────────────────
    print("[train_rnn] Loading best_model.keras for test evaluation…")
    best_model = tf.keras.models.load_model(str(export_dir / "best_model.keras"))

    test_ds = make_ds(TEST_PATH, _parse, BATCH_SIZE, shuffle=False)

    y_trues, y_preds, y_scores = [], [], []
    for x_batch, y_batch in test_ds:
        scores = best_model(x_batch, training=False).numpy().flatten()
        labels = y_batch.numpy().flatten()
        y_scores.extend(scores.tolist())
        y_trues.extend(labels.tolist())
        y_preds.extend((scores >= 0.5).astype(int).tolist())

    from sklearn.metrics import roc_auc_score

    tp = sum(1 for t, p in zip(y_trues, y_preds) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_trues, y_preds) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_trues, y_preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_trues, y_preds) if t == 1 and p == 0)
    n = len(y_trues)

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    auc       = float(roc_auc_score(y_trues, y_scores)) if len(set(y_trues)) > 1 else 0.0

    test_metrics = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "auc":       auc,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n_examples": n,
    }
    with open(export_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # test_predictions.csv
    try:
        id_ds = make_ds(TEST_PATH, _parse_with_id, BATCH_SIZE, shuffle=False)
        clip_ids = []
        for _, _, ids in id_ds:
            clip_ids.extend([s.numpy().decode() for s in ids])
    except Exception:
        clip_ids = [str(i) for i in range(count_examples(TEST_PATH))]

    with open(export_dir / "test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_wav_relpath", "y_true", "y_pred", "y_score"])
        for clip_id, y_true, y_pred, y_score in zip(clip_ids, y_trues, y_preds, y_scores):
            writer.writerow([clip_id, int(y_true), int(y_pred), f"{y_score:.6f}"])

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[train_rnn] Run complete: {export_dir}")
    for artifact in ("params.json", "history.csv", "best_model.keras",
                     "final_model.keras", "test_metrics.json", "test_predictions.csv", "logs/"):
        print(f"  {artifact:<24} ✓")
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
