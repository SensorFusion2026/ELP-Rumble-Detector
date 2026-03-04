"""Compare RNN and CNN model setup using package-local modules."""

import tensorflow as tf

from elp_rumble.models.cnn import CNN
from elp_rumble.models.cnn_config import CNNConfig
from elp_rumble.models.rnn import RNN
from elp_rumble.models.rnn_config import RNNConfig


def main() -> None:
    """Main function to demonstrate RNN and CNN model usage."""
    print("=== RNN vs CNN Model Comparison ===\n")

    common_config = {
        "learning_rate": 0.0001,
        "learning_rate_decay_steps": 500,
        "learning_rate_decay": 0.97,
        "batch_size": 32,
        "dropout_rate": 0.5,
        "activation_function": "LeakyReLU",
        "num_epochs": 10,
        "model_dir": "./model_checkpoints",
        "log_dir": "./logs",
    }

    print("Setting up RNN model...")
    rnn_config = RNNConfig(common_config)
    rnn_model = RNN(model_config=rnn_config, training=True)

    rnn_input_shape = (None, rnn_config.sequence_length)
    rnn_model.build(rnn_input_shape)

    dummy_input = tf.zeros((1, rnn_config.sequence_length))
    _ = rnn_model(dummy_input)

    print("\n--- RNN Model Summary ---")
    rnn_model.summary()

    print("\nSetting up CNN model...")
    cnn_config = CNNConfig(common_config)
    cnn_input_shape = (563, 98, 1)
    cnn_model = CNN(model_config=cnn_config, training=True, input_shape=cnn_input_shape)

    cnn_model.build((None, *cnn_input_shape))

    print("\n--- CNN Model Summary ---")
    cnn_model.summary()

    print("\n=== Model Configurations ===")
    print(f"RNN - Processes raw audio sequences of length {rnn_config.sequence_length}")
    print(
        f"RNN - Uses hierarchical LSTM with {rnn_config.num_chunks} chunks of "
        f"{rnn_config.chunk_size} samples each"
    )
    print(
        f"RNN - Architecture: TimeDistributed LSTM({rnn_config.first_lstm_units}) -> "
        f"LSTM({rnn_config.second_lstm_units}) -> Dense layers"
    )

    print(f"CNN - Processes spectrogram images of shape {cnn_input_shape}")
    print("CNN - Uses ResNet50 backbone with custom dense layers")

    print("\nCommon settings:")
    print(f"- Learning rate: {common_config['learning_rate']}")
    print(f"- Batch size: {common_config['batch_size']}")
    print(f"- Dropout rate: {common_config['dropout_rate']}")
    print(f"- Activation function: {common_config['activation_function']}")
    print(f"- Training epochs: {common_config['num_epochs']}")


if __name__ == "__main__":
    main()
