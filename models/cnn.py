import tensorflow as tf
from keras import layers
import keras
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


class CNN(keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(
            32,
            (5, 5),
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.pool1 = layers.MaxPool2D((2, 2))

        self.conv2 = layers.Conv2D(
            64,
            (5, 5),
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.pool2 = layers.MaxPool2D((2, 2))

        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(1024, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(10, dtype="float32", activation="softmax")

        self.augment = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ]
        )

    def call(self, x, training=False):
        if training:
            x = self.augment(x, training=training)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)

        x = self.dropout(x, training=training)
        x = self.out(x)
        return x
