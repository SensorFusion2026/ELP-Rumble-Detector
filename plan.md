## Plan for ML setup


### Riley:
    - Build CNN.py
        - build in OOP
            CNN model class being defined in CNN.py
            model being trained in main.py by model.train(dataset=dataset).
    - add CNN to main
    - plan 

### Stevie: ✅ COMPLETED
    - ✅ build RNN.py
        - ✅ build in OOP
        - ✅ Created RNN class with hierarchical LSTM architecture
        - ✅ Added training, prediction, and evaluation methods
        - ✅ Integrated with main.py
    - ✅ Created rnn_config.py for configuration management
    - ✅ Added proper model initialization and summary display

## Current Status:
- **RNN Model**: Complete and functional ✅
  - 145,509 trainable parameters
  - Processes 20,000 sample audio sequences
  - Hierarchical LSTM: TimeDistributed LSTM(128) → LSTM(64) → Dense layers
  - Ready for training with audio data
  
- **CNN Model**: Pending Riley's completion
  - Will process spectrogram images (563x98x1)
  - Uses ResNet50 backbone
  
- **Integration**: Ready in main.py for both models 
