import numpy as np
from keras import Sequential
from keras.src.layers import Conv2D, Dense, Flatten, MaxPooling2D, Rescaling
from keras.src.saving import load_model
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf


def set_me_label(image, _):
    return image, np.array([0])


def set_notme_label(image, _):
    return image, np.array([1])


me_train_dataset = image_dataset_from_directory(
    "data",
    image_size=(512, 512),
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
    image_size=(512, 512),
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
    image_size=(512, 512),
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
    image_size=(512, 512),
    batch_size=1,
    class_names=["notme"],
    validation_split=0.1,
    subset="validation",
    seed=2,
    color_mode='grayscale',
)

notme_val_dataset = notme_val_dataset.map(set_notme_label)

train_ds = me_train_dataset.concatenate(notme_train_dataset)
val_ds = me_val_dataset.concatenate(notme_val_dataset)

# Also use the horizontally flipped images and zoomed in for training and testing (to test mirror and cropped images)
def flip_image(image, label):
    flipped_img = tf.image.flip_left_right(image)
    return flipped_img, label


train_ds_flipped = train_ds.map(flip_image)
train_ds = train_ds.concatenate(train_ds_flipped)
train_ds = train_ds.shuffle(buffer_size=len(list(train_ds)))

val_ds_flipped = val_ds.map(flip_image)
val_ds = val_ds.concatenate(val_ds_flipped)
val_ds = val_ds.shuffle(buffer_size=len(list(val_ds)))

model = Sequential()

# Rescaling
model.add(Rescaling(1./255))

# First convolutional block
model.add(Conv2D(32, (2, 2), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Second convolutional block
model.add(Conv2D(64, (3, 3), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolutional block
model.add(Conv2D(128, (4, 4), activation='leaky_relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(Flatten())

model.add(Dense(64, activation='leaky_relu'))
model.add(Dense(64, activation='leaky_relu'))

# Output layer (2 classes: 0 for "me", 1 for "not me")
model.add(Dense(2, activation='softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, acc = model.evaluate(
    x=np.concatenate([x.numpy() for x, _ in val_ds]),
    y=np.concatenate([y.numpy() for _, y in val_ds]),
)

# noinspection PyBroadException
try:
    saved_loss, saved_acc = load_model("model/model.keras").evaluate(
        x=np.concatenate([x.numpy() for x, _ in val_ds]),
        y=np.concatenate([y.numpy() for _, y in val_ds]),
    )
    print(f"Saved model accuracy: {saved_acc: .4f}")

    print(f"Accuracy: {acc: .4f}")

    save_model = input("Save model? (y/n) ")

    if save_model == "y":
        model.save("model/model.keras")

except Exception:
    print(f"Accuracy: {acc: .4f}")

    save_model = input("Save model? (y/n) ")

    if save_model == "y":
        model.save("model/model.keras")
