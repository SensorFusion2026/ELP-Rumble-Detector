import keras
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Reshape, Dropout


@keras.saving.register_keras_serializable(package="elp_rumble")
class RNN(tf.keras.Model):
    """
    Recurrent Neural Network model for audio sequence classification.

    Processes raw audio sequences using a hierarchical LSTM approach:
    1. Splits long sequences into chunks
    2. Processes each chunk with TimeDistributed LSTM
    3. Combines chunk representations with another LSTM
    4. Classifies using fully connected layers
    """

    def __init__(
        self,
        sequence_length=20000,
        chunk_size=500,
        feature_dim=1,
        first_lstm_units=128,
        second_lstm_units=64,
        fc1_units=256,
        fc2_units=50,
        dropout_rate=0.5,
        activation="ReLU",
        **kwargs,
    ):
        super(RNN, self).__init__(**kwargs)
        self._sequence_length = sequence_length
        self._chunk_size = chunk_size
        self._feature_dim = feature_dim
        self._first_lstm_units = first_lstm_units
        self._second_lstm_units = second_lstm_units
        self._fc1_units = fc1_units
        self._fc2_units = fc2_units
        self._dropout_rate = dropout_rate
        self._activation = activation

        self._set_activation_function(activation)

        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.num_chunks = sequence_length // chunk_size
        self.feature_dim = feature_dim

        self._build_layers(first_lstm_units, second_lstm_units, fc1_units, fc2_units, dropout_rate)

    def _set_activation_function(self, activation):
        if activation == "ReLU":
            self.activation_function = tf.nn.relu
        elif activation == "LeakyReLU":
            self.activation_function = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _build_layers(self, first_lstm_units, second_lstm_units, fc1_units, fc2_units, dropout_rate):
        self.reshape = Reshape((self.num_chunks, self.chunk_size, self.feature_dim))

        self.first_lstm = TimeDistributed(LSTM(first_lstm_units, return_sequences=False))
        self.second_lstm = LSTM(second_lstm_units, return_sequences=False)

        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.fc1 = Dense(fc1_units, activation=self.activation_function, kernel_initializer=fc_init)
        self.dropout1 = Dropout(dropout_rate)
        self.fc2 = Dense(fc2_units, activation=self.activation_function, kernel_initializer=fc_init)
        self.output_layer = Dense(1, activation="sigmoid", kernel_initializer=fc_init)

    def build(self, input_shape):
        # input_shape: (batch, sequence_length)
        reshaped = (input_shape[0], self.num_chunks, self.chunk_size, self.feature_dim)
        self.reshape.build(input_shape)
        self.first_lstm.build(reshaped)
        # TimeDistributed(LSTM(units)) output: (batch, num_chunks, units)
        first_out = (input_shape[0], self.num_chunks, self._first_lstm_units)
        self.second_lstm.build(first_out)
        second_out = (input_shape[0], self._second_lstm_units)
        self.fc1.build(second_out)
        fc1_out = self.fc1.compute_output_shape(second_out)
        self.dropout1.build(fc1_out)
        self.fc2.build(fc1_out)
        fc2_out = self.fc2.compute_output_shape(fc1_out)
        self.output_layer.build(fc2_out)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.first_lstm(x)
        x = self.second_lstm(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "sequence_length": self._sequence_length,
            "chunk_size": self._chunk_size,
            "feature_dim": self._feature_dim,
            "first_lstm_units": self._first_lstm_units,
            "second_lstm_units": self._second_lstm_units,
            "fc1_units": self._fc1_units,
            "fc2_units": self._fc2_units,
            "dropout_rate": self._dropout_rate,
            "activation": self._activation,
        })
        return base
