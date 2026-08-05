"""End-to-end Multi-class Dog Breed Classification script.

This script is converted from the original Jupyter notebook `dog_vision (1).ipynb`.
It trains a TensorFlow model on the Kaggle Dog Breed Identification dataset.
"""

import os
import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base directory of this script (so paths don't depend on CWD)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (adjust these if your data lives somewhere else)
DATA_DIR = os.path.join(BASE_DIR, "dog-breed-identification")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

# Image and training settings
IMG_SIZE = 224
BATCH_SIZE = 32
# Use the full dataset by default (set to an int to limit)
NUM_IMAGES = None
NUM_EPOCHS = 10

# Model export paths
MODEL_PATH = os.path.join(BASE_DIR, "dog_breed_model.keras")
BREEDS_PATH = os.path.join(BASE_DIR, "dog_breeds.npy")


# -----------------------------------------------------------------------------
# GPU check
# -----------------------------------------------------------------------------

print("GPU", "available (YESSSS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")


# -----------------------------------------------------------------------------
# Data loading and label preparation
# -----------------------------------------------------------------------------

def load_labels(labels_csv_path: str) -> pd.DataFrame:
    labels_csv = pd.read_csv(labels_csv_path)
    print(labels_csv.describe())
    print(labels_csv.head())
    return labels_csv


def build_filenames(labels_csv: pd.DataFrame) -> List[str]:
    filenames = [os.path.join(TRAIN_DIR, f"{fname}.jpg") for fname in labels_csv["id"]]

    # Basic checks
    if os.path.isdir(TRAIN_DIR):
        train_files = os.listdir(TRAIN_DIR)
        print("Number of files in train dir:", len(train_files))
        if len(train_files) == len(filenames):
            print("Filenames match actual amount of files!!! Proceed.")
        else:
            print("WARNING: Filenames do not match actual amount of files, check the target directory")
    else:
        print(f"WARNING: Train directory not found: {TRAIN_DIR}")

    return filenames


def build_labels(labels_csv: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.array(labels_csv["breed"])
    print("Number of labels:", len(labels))

    unique_breeds = np.unique(labels)
    print("Number of unique breeds:", len(unique_breeds))

    # Turn labels into boolean arrays
    boolean_labels = [label == unique_breeds for label in labels]
    boolean_labels = np.array(boolean_labels)
    print("Boolean labels shape:", boolean_labels.shape)

    return unique_breeds, boolean_labels


# -----------------------------------------------------------------------------
# Image preprocessing and batching
# -----------------------------------------------------------------------------

def process_image(image_path: tf.Tensor, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Takes an image file path and turns the image into a resized float32 tensor."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    return image


def get_image_label(image_path: tf.Tensor, label: tf.Tensor):
    """Takes an image file path and the associated label and returns (image, label)."""
    image = process_image(image_path)
    return image, label


def create_data_batches(
    X: List[str],
    Y=None,
    batch_size: int = BATCH_SIZE,
    valid_data: bool = False,
    test_data: bool = False,
):
    """Creates tf.data batches for training/validation/test data."""

    X_tensor = tf.constant(X)

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(X_tensor)
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((X_tensor, tf.constant(Y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((X_tensor, tf.constant(Y)))
    data = data.shuffle(buffer_size=len(X))
    data = data.map( get_image_label,
    num_parallel_calls=tf.data.AUTOTUNE)
    data = data.prefetch(tf.data.AUTOTUNE)
    data_batch = data.batch(batch_size)
    return data_batch


def show_25_images(images: np.ndarray, labels: np.ndarray, unique_breeds: np.ndarray) -> None:
    """Displays 25 images and their labels from a data batch."""
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Model building
# -----------------------------------------------------------------------------




def create_model(output_shape=0,
                 model_url="https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5"):

    print("Building model with:", model_url)

    model = keras.Sequential([
        hub.KerasLayer(
            model_url,
            trainable=False
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(
            output_shape,
            activation="softmax"
        )
    ])

    model.build([None, IMG_SIZE, IMG_SIZE, 3])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


def create_tensorboard_callback(log_root: str = "logs") -> keras.callbacks.TensorBoard:
    os.makedirs(log_root, exist_ok=True)
    logdir = os.path.join(
        log_root,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    return keras.callbacks.TensorBoard(logdir)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
)


# -----------------------------------------------------------------------------
# Training wrapper
# -----------------------------------------------------------------------------


def train_model(
    train_data,
    val_data,
    unique_breeds,
):
    model = create_model(output_shape=len(unique_breeds))

    history = model.fit(
        train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
    )

    return model


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    # Load labels CSV
    labels_csv = load_labels(LABELS_CSV)

    # Filepaths and labels
    filenames = build_filenames(labels_csv)
    unique_breeds, boolean_labels = build_labels(labels_csv)

    # Optionally limit to a subset for quicker experiments
    X = filenames
    Y = boolean_labels
    if NUM_IMAGES is not None and len(X) > NUM_IMAGES:
        X = X[:NUM_IMAGES]
        Y = Y[:NUM_IMAGES]

    # Train/validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )
    print(len(X_train), len(Y_train), len(X_val), len(Y_val))

    # Create batches
    train_data = create_data_batches(X_train, Y_train)
    val_data = create_data_batches(X_val, Y_val, valid_data=True)

    # Optional: visualize a batch
    try:
        train_images, train_labels = next(train_data.as_numpy_iterator())
        print("Train batch shape:", train_images.shape, train_labels.shape)
        show_25_images(train_images, train_labels, unique_breeds)
    except StopIteration:
        print("No training data available to visualize.")

    # Train model
    model = train_model(train_data, val_data, unique_breeds)
    model.summary()

    # Save model and class names for later use in the Streamlit app
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print(f"Saving breed names to {BREEDS_PATH}")
    np.save(BREEDS_PATH, unique_breeds)


if __name__ == "__main__":
    main()
