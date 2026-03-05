# Configuration used for RNN training
import os
from elp_rumble.config.paths import PROJECT_ROOT, TFRECORDS_AUDIO_DIR

class RNNConfig(object):
    """Configuration class for RNN model settings and paths."""
    
    def __init__(self, config):
        """Initialize RNN configuration with provided config dictionary."""
        self.config = config
        
        # Data related paths
        self.DATASET_FOLDER = TFRECORDS_AUDIO_DIR
        self.TRAIN_FILE = 'train.tfrecord'
        self.VALIDATE_FILE = 'validate.tfrecord'
        self.TEST_FILE = 'test.tfrecord'
        
        # Cross-validation settings
        self.K_FOLDS = 5
        self.MAX_CV_EPOCHS = 5
        
        # Model parameters from config
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.learning_rate_decay_steps = config.get("learning_rate_decay_steps", 500)
        self.learning_rate_decay = config.get("learning_rate_decay", 0.97)
        self.batch_size = config.get("batch_size", 32)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.activation_function = config.get("activation_function", "ReLU")
        self.num_epochs = config.get("num_epochs", 10)
        self.model_dir = config.get("model_dir", str(PROJECT_ROOT / "model_checkpoints"))
        self.log_dir = config.get("log_dir", str(PROJECT_ROOT / "logs"))
        
        # Prediction settings
        self.NUM_CLASSES = 1
        self.PROB_THRESHOLD = 0.5
        
        # Display and saving steps
        self.TRAIN_STEP = 10
        self.VALIDATION_STEP = 50
        self.SAVE_STEP = 5000
        
        # Output paths
        self.SUMMARY_PATH = 'results/summary'
        self.LOG_DIR = 'results/logs'
        self.MODEL_FILE = 'rnn_model'
        self.TESTING_IMAGES = 'rnn_results'
        
        # Early stopping configuration
        self.PATIENCE = 20
        self.MIN_DELTA = 0.0005
        
        # RNN-specific architecture parameters
        self.sequence_length = 20000
        self.chunk_size = 500
        self.num_chunks = self.sequence_length // self.chunk_size
        self.feature_dim = 1
        
        # LSTM layer configurations
        self.first_lstm_units = 128
        self.second_lstm_units = 64
        self.fc1_units = 256
        self.fc2_units = 50