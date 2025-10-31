import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Reshape, Dropout
import numpy as np
import os
from rnn_config import RNNConfig

class RNN(tf.keras.Model):
    """
    Recurrent Neural Network model for audio sequence classification.
    
    This model processes raw audio sequences using a hierarchical LSTM approach:
    1. Splits long sequences into chunks
    2. Processes each chunk with TimeDistributed LSTM
    3. Combines chunk representations with another LSTM
    4. Classifies using fully connected layers
    """
    
    def __init__(self, model_config, training=True, input_shape=None):
        """
        Initialize RNN model.
        
        Args:
            model_config: RNNConfig object containing model configuration
            training: Boolean indicating if model is in training mode
            input_shape: Shape of input data (optional, can be inferred)
        """
        super(RNN, self).__init__()
        
        self.cfg = model_config
        self.training_mode = training
        self.input_shape_provided = input_shape
        
        # Set activation function
        self._set_activation_function()
        
        # Architecture parameters from config
        self.sequence_length = self.cfg.sequence_length
        self.chunk_size = self.cfg.chunk_size
        self.num_chunks = self.cfg.num_chunks
        self.feature_dim = self.cfg.feature_dim
        
        # Build model layers
        self._build_layers()
    
    def _set_activation_function(self):
        """Set the activation function based on configuration."""
        if self.cfg.activation_function == 'ReLU':
            self.activation_function = tf.nn.relu
        elif self.cfg.activation_function == 'LeakyReLU':
            self.activation_function = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        else:
            raise ValueError(f"Unsupported activation function: {self.cfg.activation_function}")
    
    def _build_layers(self):
        """Build all the layers of the RNN model."""
        # Reshape layer to create chunks from sequence
        self.reshape = Reshape((self.num_chunks, self.chunk_size, self.feature_dim))
        
        # Hierarchical LSTM layers
        self.first_lstm = TimeDistributed(
            LSTM(self.cfg.first_lstm_units, return_sequences=False)
        )
        self.second_lstm = LSTM(self.cfg.second_lstm_units, return_sequences=False)
        
        # Fully connected layers with proper initialization
        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        
        self.fc1 = Dense(
            self.cfg.fc1_units, 
            activation=self.activation_function, 
            kernel_initializer=fc_init
        )
        self.dropout1 = Dropout(self.cfg.dropout_rate)
        
        self.fc2 = Dense(
            self.cfg.fc2_units, 
            activation=self.activation_function, 
            kernel_initializer=fc_init
        )
        
        # Output layer for binary classification
        self.output_layer = Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=fc_init
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length)
            training: Boolean indicating training mode
            
        Returns:
            Output predictions of shape (batch_size, 1)
        """
        if training is None:
            training = self.training_mode
            
        # Reshape input to chunks: (batch_size, num_chunks, chunk_size, feature_dim)
        x = self.reshape(inputs)
        
        # Process each chunk with TimeDistributed LSTM
        x = self.first_lstm(x)
        
        # Combine chunk representations with second LSTM
        x = self.second_lstm(x)
        
        # Fully connected layers
        x = self.fc1(x)
        if training:
            x = self.dropout1(x, training=training)
            
        x = self.fc2(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def train_model(self, dataset, validation_dataset=None, callbacks=None):
        """
        Train the RNN model.
        
        Args:
            dataset: Training dataset
            validation_dataset: Validation dataset (optional)
            callbacks: List of Keras callbacks (optional)
            
        Returns:
            Training history
        """
        # Compile the model
        optimizer = self._get_optimizer()
        
        self.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Setup callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train the model
        history = self.fit(
            dataset,
            validation_data=validation_dataset,
            epochs=self.cfg.num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _get_optimizer(self):
        """Get the optimizer based on configuration."""
        if hasattr(self.cfg, 'learning_rate_decay_steps'):
            # Use learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.cfg.learning_rate,
                decay_steps=self.cfg.learning_rate_decay_steps,
                decay_rate=self.cfg.learning_rate_decay
            )
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            return tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
    
    def _get_default_callbacks(self):
        """Get default callbacks for training."""
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.cfg.PATIENCE,
            min_delta=self.cfg.MIN_DELTA,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        if hasattr(self.cfg, 'model_dir'):
            os.makedirs(self.cfg.model_dir, exist_ok=True)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.cfg.model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
            callbacks.append(checkpoint)
        
        # TensorBoard logging
        if hasattr(self.cfg, 'log_dir'):
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=self.cfg.log_dir,
                histogram_freq=1
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def predict_proba(self, dataset):
        """
        Get prediction probabilities.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Prediction probabilities
        """
        return self.predict(dataset)
    
    def predict_classes(self, dataset, threshold=None):
        """
        Get predicted classes.
        
        Args:
            dataset: Dataset to predict on
            threshold: Classification threshold (default from config)
            
        Returns:
            Predicted classes (0 or 1)
        """
        if threshold is None:
            threshold = self.cfg.PROB_THRESHOLD
            
        probabilities = self.predict_proba(dataset)
        return (probabilities >= threshold).astype(int)
    
    def evaluate_model(self, test_dataset):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.evaluate(test_dataset, verbose=1)
        
        # Create results dictionary
        metrics_names = self.metrics_names
        results_dict = dict(zip(metrics_names, results))
        
        return results_dict
    
    def save_model(self, filepath):
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model weights.
        
        Args:
            filepath: Path to load the model from
        """
        self.load_weights(filepath)
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {  
        "learning_rate": 0.0001,
        "learning_rate_decay_steps": 500,
        "learning_rate_decay": 0.97,
        "batch_size": 32,
        "dropout_rate": 0.5,
        "activation_function": "LeakyReLU",
        "num_epochs": 10,
        "model_dir": "./model_checkpoints",
        "log_dir": "./logs"
    }
    
    # Create config object
    model_config = RNNConfig(config)
    
    # Create model
    model = RNN(model_config=model_config, training=True)
    
    # Build the model with example input shape
    # Input shape: (batch_size, sequence_length)
    model.build((None, model_config.sequence_length))
    
    # Print model summary
    print("RNN Model Architecture:")
    model.summary()
    
    print(f"\nModel configuration:")
    print(f"- Sequence length: {model_config.sequence_length}")
    print(f"- Chunk size: {model_config.chunk_size}")
    print(f"- Number of chunks: {model_config.num_chunks}")
    print(f"- Activation function: {model_config.activation_function}")
    print(f"- Dropout rate: {model_config.dropout_rate}")