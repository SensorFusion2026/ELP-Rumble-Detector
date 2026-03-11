# models/cnn.py
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="elp_rumble")
class CNN(tf.keras.Model):

    def __init__(self, input_shape=(563, 98, 1), dropout_rate=0.5, activation="LeakyReLU", **kwargs):
        super(CNN, self).__init__(**kwargs)
        self._input_shape_arg = tuple(input_shape)
        self._dropout_rate = dropout_rate
        self._activation = activation

        if activation == "ReLU":
            self.activation_function = tf.nn.relu
        elif activation == "LeakyReLU":
            self.activation_function = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )
        self.flatten = tf.keras.layers.GlobalAveragePooling2D()

        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.fc1 = tf.keras.layers.Dense(256, activation=self.activation_function, kernel_initializer=fc_init)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(50, activation=self.activation_function, kernel_initializer=fc_init)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=fc_init)

    def build(self, input_shape):
        self.resnet.build(input_shape)
        resnet_out = self.resnet.compute_output_shape(input_shape)
        self.flatten.build(resnet_out)
        pool_out = self.flatten.compute_output_shape(resnet_out)
        self.fc1.build(pool_out)
        fc1_out = self.fc1.compute_output_shape(pool_out)
        self.dropout1.build(fc1_out)
        self.fc2.build(fc1_out)
        fc2_out = self.fc2.compute_output_shape(fc1_out)
        self.output_layer.build(fc2_out)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.resnet(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "input_shape": self._input_shape_arg,
            "dropout_rate": self._dropout_rate,
            "activation": self._activation,
        })
        return base
