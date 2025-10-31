#!/usr/bin/env python3
"""
Test script to verify RNN functionality with dummy data.
"""

import tensorflow as tf
import numpy as np
from rnn import RNN
from rnn_config import RNNConfig

def test_rnn_model():
    """Test the RNN model with dummy data."""
    print("=== Testing RNN Model with Dummy Data ===\n")
    
    # Configuration
    config = {  
        "learning_rate": 0.001,
        "batch_size": 4,
        "dropout_rate": 0.3,
        "activation_function": "LeakyReLU",
        "num_epochs": 2,
        "model_dir": "./test_model_checkpoints",
        "log_dir": "./test_logs"
    }
    
    # Create model
    rnn_config = RNNConfig(config)
    model = RNN(model_config=rnn_config, training=True)
    
    # Create dummy data
    batch_size = 4
    sequence_length = rnn_config.sequence_length
    
    print(f"Creating dummy data...")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {sequence_length}")
    
    # Generate random audio-like data
    X_dummy = np.random.randn(batch_size, sequence_length).astype(np.float32)
    y_dummy = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    
    print(f"- Input shape: {X_dummy.shape}")
    print(f"- Output shape: {y_dummy.shape}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    predictions = model(X_dummy, training=False)
    print(f"- Prediction shape: {predictions.shape}")
    print(f"- Sample predictions: {predictions.numpy().flatten()[:4]}")
    
    # Test training mode
    print(f"\nTesting training mode...")
    predictions_train = model(X_dummy, training=True)
    print(f"- Training predictions shape: {predictions_train.shape}")
    
    # Test compilation
    print(f"\nCompiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Test prediction methods
    print(f"\nTesting prediction methods...")
    probas = model.predict_proba(X_dummy)
    classes = model.predict_classes(X_dummy, threshold=0.5)
    
    print(f"- Probabilities shape: {probas.shape}")
    print(f"- Classes shape: {classes.shape}")
    print(f"- Sample probabilities: {probas.flatten()[:4]}")
    print(f"- Sample classes: {classes.flatten()[:4]}")
    
    # Test a small training step
    print(f"\nTesting single training step...")
    history = model.fit(X_dummy, y_dummy, epochs=1, verbose=1)
    
    print(f"\n✅ RNN Model Test Completed Successfully!")
    print(f"- Model has {model.count_params():,} parameters")
    print(f"- All forward passes working correctly")
    print(f"- Training functionality verified")

if __name__ == "__main__":
    test_rnn_model()