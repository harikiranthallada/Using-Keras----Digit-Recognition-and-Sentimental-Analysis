import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import mnist


train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation="softmax"),
])

# Achieves ~99% accuracy.
# model = Sequential([
#     Conv2D(32, input_shape=(28, 28, 1), activation="relu"),
#     MaxPooling2D(pool_size=pool_size),
#     Conv2D(64, activation="relu"),
#     MaxPooling2D(pool_size=pool_size),
#     Flatten(),
#     Dropout(0.5),
#     Dense(10, activation="softmax"),
# ])

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels))
)

# model.evaluate(
#     test_images,
#     to_categorical(test_labels)
# )
# model.save_weights('model.h5')





