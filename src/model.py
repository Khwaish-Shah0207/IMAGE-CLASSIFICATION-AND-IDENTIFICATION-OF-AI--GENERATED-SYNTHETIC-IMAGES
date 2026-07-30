import tensorflow as tf
def build_model():
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Input(shape=(32, 32, 3)),
        # First Convolution Block
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Second Convolution Block
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Third Convolution Block
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),
        # Convert Feature Maps into a Vector
        tf.keras.layers.Flatten(),
        # Fully Connected Layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        # Prevent Overfitting
        tf.keras.layers.Dropout(0.5),
        # Output Layer
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])
    return model
# Testing the model
if __name__ == "__main__":
    model = build_model()
    model.summary()
