# models/cnn_config.py
import os
from datetime import datetime
from data_creation.data_path_config import DataPathConfig

class CNNConfig:
    # Configuration for CNN model. Based on RNNConfig
    def __init__(self, config):
        # ---- core hyperparams ----
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.learning_rate_decay_steps = config.get("learning_rate_decay_steps", 500)
        self.learning_rate_decay = config.get("learning_rate_decay", 0.97)
        self.batch_size = config.get("batch_size", 32)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.activation_function = config.get("activation_function", "LeakyReLU")
        self.num_epochs = config.get("num_epochs", 10)
        self.downsample_fraction = config.get("downsample_fraction", .01)

        # ---- run identity / export directory ----
        timestamp = config.get("run_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        export_root = config.get("export_root", "./exported/cnn_runs")

        default_run_name = config.get(
            "run_name",
            f"cnn_{timestamp}"
            f"_bs{self.batch_size}"
            f"_lr{self.learning_rate}"
            f"_down{self.downsample_fraction}"
        )

        default_export_dir = os.path.join(export_root, default_run_name)

        # Single root directory for this run
        self.export_dir = config.get("export_dir", default_export_dir)

        # ---- misc ----
        self.PATIENCE = 10
        self.MIN_DELTA = 0.0005
        self.PROB_THRESHOLD = 0.5
        self.input_shape = (563, 98, 1)

# class CNNConfig:
#     #Configuration for CNN model. Based on RNNConfig
#     def __init__(self, config):
#         self.learning_rate = config.get("learning_rate", 0.0001)
#         self.learning_rate_decay_steps = config.get("learning_rate_decay_steps", 500)
#         self.learning_rate_decay = config.get("learning_rate_decay", 0.97)
#         self.batch_size = config.get("batch_size", 32)
#         self.dropout_rate = config.get("dropout_rate", 0.5)
#         self.activation_function = config.get("activation_function", "LeakyReLU")
#         self.num_epochs = config.get("num_epochs", 10)
#         self.model_dir = config.get("model_dir", "./model_checkpoints")
#         self.log_dir = config.get("log_dir", "./logs")
#         self.PATIENCE = 10
#         self.MIN_DELTA = 0.0005
#         self.PROB_THRESHOLD = 0.5
#         self.input_shape = (563, 98, 1)
