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


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("Train images:", x_train.shape)
print("Train labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

batch_size = 128

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(50000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model = CNN()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=2000, decay_rate=0.9
)
model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(train_ds, epochs=10, validation_data=test_ds, verbose=1)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy: ", test_acc)
