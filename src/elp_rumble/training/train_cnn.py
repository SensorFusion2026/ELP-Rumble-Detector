# src/elp_rumble/training/train_cnn.py
import os
import argparse
import json
import csv
import tensorflow as tf
from datetime import datetime

from elp_rumble.models.cnn import CNN
from elp_rumble.models.cnn_config import CNNConfig
from elp_rumble.input_pipeline.spectrogram_tfrecords import (
    make_ds,
    get_spec_paths,
    INPUT_SHAPE,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--downsample_frac", type=float, default=None)
    p.add_argument("--export_dir", type=str, default=None)

    return p.parse_args()

def build_cfg(args):
    cfg = CNNConfig({})  # start from defaults
    
    # ---- hyperparams ----
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.downsample_frac is not None:
        cfg.downsample_fraction = args.downsample_frac

    # ---- export directory ----
    if args.export_dir is not None:
        cfg.export_dir = args.export_dir

    return cfg

def main():
    SEED = 42

    args = parse_args()
    cfg = build_cfg(args)

    os.makedirs(cfg.export_dir, exist_ok=True)

    paths = get_spec_paths()

    train_ds = make_ds(paths["train"], cfg.batch_size, shuffle=True, downsample_fraction=cfg.downsample_fraction, seed=SEED)
    val_ds   = make_ds(paths["val"],   cfg.batch_size, shuffle=False, downsample_fraction=cfg.downsample_fraction, seed=SEED)
    test_ds  = make_ds(paths["test"],  cfg.batch_size, shuffle=False, downsample_fraction=cfg.downsample_fraction, seed=SEED)

    model = CNN(model_config=cfg, training=True, input_shape=INPUT_SHAPE)
    model.build((None, *INPUT_SHAPE))
    print("\n--- CNN Summary ---")
    model.summary()

    history = model.train_model(dataset=train_ds, validation_dataset=val_ds)

    print("\n--- Test Evaluation ---")
    results = model.evaluate_model(test_ds)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # --------- save training history as CSV ---------
    history_path = os.path.join(cfg.export_dir, "history.csv")
    hist_dict = history.history  # dict: metric -> list over epochs
    metric_names = list(hist_dict.keys())

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metric_names)
        num_epochs = len(next(iter(hist_dict.values()))) if hist_dict else 0
        for epoch_idx in range(num_epochs):
            row = [epoch_idx] + [hist_dict[m][epoch_idx] for m in metric_names]
            writer.writerow(row)

    # --------- save config / params as JSON ---------
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    params_path = os.path.join(cfg.export_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # --------- save final model ---------
    model.save_model(cfg.export_dir)
    print(f"\n✅ Final model exported to: {cfg.export_dir}")
    print(f"   - best weights: {os.path.join(cfg.export_dir, 'best_weights.weights.h5')}")
    print(f"   - final model: {os.path.join(cfg.export_dir, 'final_model.keras')}")
    print(f"   - history:     {history_path}")
    print(f"   - params:      {params_path}")
    print(f"   - logs:        {os.path.join(cfg.export_dir, 'logs')}")

if __name__ == "__main__":
    main()