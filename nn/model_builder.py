from typing import Callable

import numpy as np
from keras import Sequential
from keras.src.layers import Conv2D, Dense, Flatten, MaxPooling2D, Rescaling
from keras.src.saving import load_model
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf


# Query from a list using a list of indices
def many_indices(_indices: list[int], _list: list) -> list:
    res = []
    for i in _indices:
        res.append(_list[i])

    return res


# Conditionally partition a list (first list is elements for which condition is true)
def conditional_partition(_condition: Callable, _list: list) -> tuple[list, list]:
    res1 = []
    res2 = []
    for elem in _list:
        if _condition(elem):
            res1.append(elem)
        else:
            res2.append(elem)
    return res1, res2


def set_me_label(image, _):
    return image, np.array([0])


def set_notme_label(image, _):
    return image, np.array([1])


me_train_dataset = image_dataset_from_directory(
    "data",
    image_size=(224, 224),
    batch_size=1,
    class_names=["me"],
    validation_split=0.1,
    subset="training",
    seed=1,
    color_mode='grayscale',
)

me_train_dataset = me_train_dataset.map(set_me_label)

me_val_dataset = image_dataset_from_directory(
    "data",
    image_size=(224, 224),
    batch_size=1,
    class_names=["me"],
    validation_split=0.1,
    subset="validation",
    seed=1,
    color_mode='grayscale',
)

me_val_dataset = me_val_dataset.map(set_me_label)

notme_train_dataset = image_dataset_from_directory(
    "data",
    image_size=(224, 224),
    batch_size=1,
    class_names=["notme"],
    validation_split=0.1,
    subset="training",
    seed=2,
    color_mode='grayscale',
)

notme_train_dataset = notme_train_dataset.map(set_notme_label)

notme_val_dataset = image_dataset_from_directory(
    "data",
    image_size=(224, 224),
    batch_size=1,
    class_names=["notme"],
    validation_split=0.1,
    subset="validation",
    seed=2,
    color_mode='grayscale',
)

notme_val_dataset = notme_val_dataset.map(set_notme_label)

train_ds = me_train_dataset.concatenate(notme_train_dataset).shuffle(buffer_size=56)
val_ds = me_val_dataset.concatenate(notme_val_dataset).shuffle(buffer_size=14)

# Also use the horizontally flipped images for training and testing (to test mirror images)


def flip_image(image, label):
    flipped_img = tf.image.flip_left_right(image)
    return image, label


train_ds_flipped = train_ds.map(flip_image)
train_ds = train_ds.concatenate(train_ds_flipped)

val_ds_flipped = val_ds.map(flip_image)
val_ds = val_ds.concatenate(val_ds_flipped)

model = Sequential()

# Rescaling
model.add(Rescaling(1./255))

# First convolutional block
model.add(Conv2D(32, (3, 3), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Second convolutional block
model.add(Conv2D(64, (3, 3), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolutional block
model.add(Conv2D(128, (3, 3), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='leaky_relu'))

# Output layer (2 classes: 0 for "me", 1 for "not me")
model.add(Dense(2, activation='softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, acc = model.evaluate(
    x=np.concatenate([x.numpy() for x, _ in val_ds]),
    y=np.concatenate([y.numpy() for _, y in val_ds]),
)

print(f"Accuracy: {acc: .4f}")

# saved_loss, saved_acc = load_model("model/model.keras").evaluate(
#     x=np.concatenate([x.numpy() for x, _ in val_ds]),
    # y=np.concatenate([y.numpy() for _, y in val_ds]),
# )

# if acc > saved_acc:
model.save("model/model.keras")
