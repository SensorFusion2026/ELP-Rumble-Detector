# models/cnn.py
import tensorflow as tf
import os
from datetime import datetime
from models.cnn_config import CNNConfig

class CNN(tf.keras.Model):

    def __init__(self, model_config, training=True, input_shape=None):
        #Initialize CNN model.
        #Args:
        #model_config: Configuration object with CNN settings
        #training: Boolean for training mode
        #input_shape: Expected input shape (H, W, C)
        
        super(CNN, self).__init__()
        self.cfg = model_config
        self.training_mode = training

        # Set activation
        self._set_activation_function()

        # Backbone CNN (ResNet50)
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape
        )

        # Flatten output for dense layers
        self.flatten = tf.keras.layers.GlobalAveragePooling2D()

        # Weight initializer
        fc_init = tf.keras.initializers.GlorotNormal(seed=42)

        # Dense layers
        self.fc1 = tf.keras.layers.Dense(
            256, activation=self.activation_function, kernel_initializer=fc_init
        )
        self.dropout1 = tf.keras.layers.Dropout(self.cfg.dropout_rate)
        self.fc2 = tf.keras.layers.Dense(
            50, activation=self.activation_function, kernel_initializer=fc_init
        )
        self.output_layer = tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer=fc_init
        )

    def _set_activation_function(self):
        #Set the activation function based on configuration.
        if self.cfg.activation_function == "ReLU":
            self.activation_function = tf.nn.relu
        elif self.cfg.activation_function == "LeakyReLU":
            self.activation_function = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        else:
            raise ValueError(f"Unsupported activation function: {self.cfg.activation_function}")

    def call(self, x, training=False):
        #Forward pass through CNN.
        # Convert grayscale to RGB if necessary
        # if x.shape[-1] == 1:
        #     x = tf.image.grayscale_to_rgb(x)

        x = self.resnet(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

    def train_model(self, dataset, validation_dataset=None, callbacks=None):
        #Train the CNN model.
        optimizer = self._get_optimizer()
        self.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        if callbacks is None:
            callbacks = self._get_default_callbacks()

        history = self.fit(
            dataset,
            validation_data=validation_dataset,
            epochs=self.cfg.num_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def _get_optimizer(self):
        #Create optimizer (with optional learning rate decay).
        if hasattr(self.cfg, "learning_rate_decay_steps"):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.cfg.learning_rate,
                decay_steps=self.cfg.learning_rate_decay_steps,
                decay_rate=self.cfg.learning_rate_decay,
            )
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        return tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)

    def _get_default_callbacks(self):
        #Default callbacks: early stopping, checkpoint, tensorboard.
        callbacks = []

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.cfg.PATIENCE,
            min_delta=self.cfg.MIN_DELTA,
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)

        # Lightweight best-weights checkpoint (weights only)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.cfg.export_dir, "best_weights.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_cb)

        # Tensorboard logs
        tb_log_dir = os.path.join(self.cfg.export_dir, "logs")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir))

        return callbacks

    def evaluate_model(self, test_dataset):
        #Evaluate the CNN model.
        results = self.evaluate(test_dataset, verbose=1)
        return dict(zip(self.metrics_names, results))

    def predict_proba(self, dataset):
        #Predict probabilities.
        return self.predict(dataset)

    def predict_classes(self, dataset, threshold=None):
        #Predict binary classes.
        if threshold is None:
            threshold = self.cfg.PROB_THRESHOLD
        probs = self.predict(dataset)
        return (probs >= threshold).astype(int)

    def save_model(self, export_dir: str):
        """
        Save the full model (architecture + weights + optimizer, etc.)
        into export_dir/final_model.keras
        """
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, "final_model.keras")
        self.save(filepath)
        print(f"Final model saved to {filepath}")

    # def save_model(self, filepath):
    #     """
    #     Save the model.

    #     Args:
    #         filepath: Path to save the model or directory to put it in.
    #     """
    #     # If the user provides a directory, turn it into a full file path
    #     if not filepath.endswith(".keras"):
    #         filepath = os.path.join(filepath, "saved_model.keras")

    #     dirpath = os.path.dirname(filepath)
    #     if dirpath:
    #         os.makedirs(dirpath, exist_ok=True)

    #     self.save(filepath)
    #     print(f"Model saved to {filepath}")

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
    model_config = CNNConfig(config)

    # Create model
    model = CNN(model_config=model_config, training=True)

    # Build the model with example input shape
    # Input shape: (batch_size, sequence_length)
    model.build((None, model_config.sequence_length))

    # Print model summary
    print("CNN Model Architecture:")
    model.summary()

    print(f"\nModel configuration:")
    print(f"- Sequence length: {model_config.sequence_length}")
    print(f"- Chunk size: {model_config.chunk_size}")
    print(f"- Number of chunks: {model_config.num_chunks}")
    print(f"- Activation function: {model_config.activation_function}")
    print(f"- Dropout rate: {model_config.dropout_rate}")

