

def main():
    from cnn import CNN
    from cnn_config import CNNConfig
    
    

    config = {  
            "learning_rate": 0.0001,
            "learning_rate_decay_steps": 500,
            "learning_rate_decay": 0.97,
            "batch_size": 32,
            "dropout_rate": 0.5,
            "activation_function": "ReLU",  # Options: 'ReLU', 'LeakyReLU'
            "num_epochs": 10,
            "model_dir": "./model_checkpoints",
            "log_dir": "./logs"
    }

    model_config = CNNConfig(config)

    # Example input shape (height, width, channels)
    input_shape = (128, 128, 1)  # Adjust as needed

    # Instantiate the model
    model = CNN(model_config=model_config, training=True, input_shape=input_shape)

    # Print model summary
    model.build((None, *input_shape))  # Batch size is None for dynamic batching
    model.summary()