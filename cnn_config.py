from data_creation.data_path_config import DataPathConfig

class CNNConfig:
    #Configuration for CNN model. Based on RNNConfig
    def __init__(self, config):
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.learning_rate_decay_steps = config.get("learning_rate_decay_steps", 500)
        self.learning_rate_decay = config.get("learning_rate_decay", 0.97)
        self.batch_size = config.get("batch_size", 32)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.activation_function = config.get("activation_function", "LeakyReLU")
        self.num_epochs = config.get("num_epochs", 10)
        self.model_dir = config.get("model_dir", "./model_checkpoints")
        self.log_dir = config.get("log_dir", "./logs")
        self.PATIENCE = 10
        self.MIN_DELTA = 0.0005
        self.PROB_THRESHOLD = 0.5
        self.input_shape = (563, 98, 1)
