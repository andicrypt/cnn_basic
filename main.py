from models.build_model import CNN
import keras
import tensorflow as tf



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("Train images:", x_train.shape)
print("Train labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

batch_size = 512

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(50000)
    .map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
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
