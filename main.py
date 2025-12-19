
def main():
    """Main function to demonstrate RNN and CNN model usage."""
    # Import models and configurations
    from rnn import RNN
    from rnn_config import RNNConfig
    from models.cnn import CNN
    from models.cnn_config import CNNConfig
    
    print("=== RNN vs CNN Model Comparison ===\n")
    
    # Common configuration for both models
    common_config = {  
        "learning_rate": 0.0001,
        "learning_rate_decay_steps": 500,
        "learning_rate_decay": 0.97,
        "batch_size": 32,
        "dropout_rate": 0.5,
        "activation_function": "LeakyReLU",  # Options: 'ReLU', 'LeakyReLU'
        "num_epochs": 10,
        "model_dir": "./model_checkpoints",
        "log_dir": "./logs"
    }
    
    # === RNN Model Setup ===
    print("Setting up RNN model...")
    rnn_config = RNNConfig(common_config)
    rnn_model = RNN(model_config=rnn_config, training=True)
    
    # Build RNN model with audio sequence input shape
    rnn_input_shape = (None, rnn_config.sequence_length)  # (batch_size, sequence_length)
    rnn_model.build(rnn_input_shape)
    
    # Create a dummy input to properly initialize the model
    import tensorflow as tf
    dummy_input = tf.zeros((1, rnn_config.sequence_length))
    _ = rnn_model(dummy_input)
    
    print("\n--- RNN Model Summary ---")
    rnn_model.summary()
    
    # === CNN Model Setup (commented out until CNN.py is created) ===
    print("\nSetting up CNN model...")
    cnn_config = CNNConfig(common_config)
    cnn_input_shape = (563, 98, 1)  # Spectrogram dimensions from legacy
    cnn_model = CNN(model_config=cnn_config, training=True, input_shape=cnn_input_shape)
    
    # Build CNN model
    cnn_model.build((None, *cnn_input_shape))
    
    print("\n--- CNN Model Summary ---")
    cnn_model.summary()
    
    print(f"\n=== Model Configurations ===")
    print(f"RNN - Processes raw audio sequences of length {rnn_config.sequence_length}")
    print(f"RNN - Uses hierarchical LSTM with {rnn_config.num_chunks} chunks of {rnn_config.chunk_size} samples each")
    print(f"RNN - Architecture: TimeDistributed LSTM({rnn_config.first_lstm_units}) -> LSTM({rnn_config.second_lstm_units}) -> Dense layers")
    
    # CNN info will be added when CNN.py is ready
    print(f"CNN - Processes spectrogram images of shape {cnn_input_shape}")
    print(f"CNN - Uses ResNet50 backbone with custom dense layers")
    
    print(f"\nCommon settings:")
    print(f"- Learning rate: {common_config['learning_rate']}")
    print(f"- Batch size: {common_config['batch_size']}")
    print(f"- Dropout rate: {common_config['dropout_rate']}")
    print(f"- Activation function: {common_config['activation_function']}")
    print(f"- Training epochs: {common_config['num_epochs']}")


if __name__ == "__main__":
    main()