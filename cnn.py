import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, input_shape=None, dropout_rate=0.2, alpha=0.2):
        # To enable summary before model receives data, pass input_shape=(H, W, 3)

        super().__init__()

        # Backbone model - ResNet50
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False, 
            weights=None,
            input_shape=input_shape # if None, shape will be inferred on first call    
        )
        
        # Flattening output from ResNet50 is necessary to pass into Dense layer(s)
        # Choose ONE of these options to flatten output:
        self.flatten = tf.keras.layers.Flatten() # larger, slower
        # self.flatten = tf.keras.layers.GlobalAveragePooling2D() # smaller, faster
        
        # Glorot normal initializer for dense layer weights — helps maintain stable gradients
        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        
        # Define leaky relu activation function
        leaky = lambda t: tf.nn.leaky_relu(t, alpha=alpha)

        # Fully connected, dropout, and output layers
        self.fc1 = tf.keras.layers.Dense(256, activation=leaky, kernel_initializer=fc_init)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(50, activation=leaky, kernel_initializer=fc_init)
        self.out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=fc_init)
        
        return
    
    def call(self, x, training=False):
        # Ensure 3 channels for ResNet
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        
        x = self.resnet(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.out(x)
        return x

