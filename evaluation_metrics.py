"""Evaluation helpers for the dog breed classifier.

This module can load the saved Keras model, build a validation dataset,
and report accuracy, precision, recall, F1 score, log loss, and a
confusion matrix.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dog-breed-identification")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
MODEL_PATH = os.path.join(BASE_DIR, "dog_breed_model.keras")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_IMAGES = None


def load_labels(labels_csv_path: str = LABELS_CSV) -> pd.DataFrame:
    """Load the Kaggle dog breed labels CSV."""

    return pd.read_csv(labels_csv_path)


def build_filenames(labels_csv: pd.DataFrame) -> list[str]:
    """Convert image ids into filepaths inside the train directory."""

    return [os.path.join(TRAIN_DIR, f"{fname}.jpg") for fname in labels_csv["id"]]


def build_labels(labels_csv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the breed names and one-hot encoded labels."""

    labels = np.array(labels_csv["breed"])
    unique_breeds = np.unique(labels)
    boolean_labels = np.array([label == unique_breeds for label in labels])
    return unique_breeds, boolean_labels


def process_image(image_path: tf.Tensor, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Read, decode, resize, and normalize a single image."""

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    return image


def create_data_batches(
    X: list[str],
    Y=None,
    batch_size: int = BATCH_SIZE,
    valid_data: bool = False,
    test_data: bool = False,
):
    """Create TensorFlow dataset batches for prediction/evaluation."""

    X_tensor = tf.constant(X)

    if test_data:
        data = tf.data.Dataset.from_tensor_slices(X_tensor)
        return data.map(process_image).batch(batch_size)

    if valid_data:
        data = tf.data.Dataset.from_tensor_slices((X_tensor, tf.constant(Y)))

        def get_image_label(image_path: tf.Tensor, label: tf.Tensor):
            return process_image(image_path), label

        return data.map(get_image_label).batch(batch_size)

    data = tf.data.Dataset.from_tensor_slices((X_tensor, tf.constant(Y)))
    data = data.shuffle(buffer_size=len(X))

    def get_image_label(image_path: tf.Tensor, label: tf.Tensor):
        return process_image(image_path), label

    return data.map(get_image_label).batch(batch_size)


def load_keras_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load the saved Keras model that was trained with TensorFlow Hub."""

    return tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})


def prepare_validation_dataset(num_images: Optional[int] = NUM_IMAGES) -> tuple[tf.data.Dataset, np.ndarray]:
    """Build the validation dataset used for model evaluation."""

    labels_csv = load_labels()
    filenames = build_filenames(labels_csv)
    unique_breeds, boolean_labels = build_labels(labels_csv)

    X = filenames
    Y = boolean_labels
    if num_images is not None and len(X) > num_images:
        X = X[:num_images]
        Y = Y[:num_images]

    _, X_val, _, Y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    val_data = create_data_batches(X_val, Y_val, valid_data=True)
    return val_data, unique_breeds


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return core classification metrics for multi-class predictions.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    y_prob:
        Optional predicted class probabilities. Required for log loss.
    target_names:
        Optional class names used in the classification report.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["log_loss"] = log_loss(y_true, np.asarray(y_prob))

    if target_names is not None:
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            target_names=list(target_names),
            zero_division=0,
        )

    return metrics


def print_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[np.ndarray] = None,
) -> None:
    """Print accuracy and precision metrics in a readable format."""

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob, target_names=target_names)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 score (weighted): {metrics['f1_weighted']:.4f}")

    if "log_loss" in metrics:
        print(f"Log loss: {metrics['log_loss']:.4f}")

    report = metrics.get("classification_report")
    if report:
        print("\nClassification report:\n")
        print(report)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[np.ndarray] = None,
    normalize: bool = False,
) -> None:
    """Plot a confusion matrix for the classifier outputs."""

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    labels = list(target_names) if target_names is not None else None
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(12, 12))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=90, colorbar=True)
    title = "Normalized confusion matrix" if normalize else "Confusion matrix"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def evaluate_model_on_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    target_names: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate a Keras model directly on a TensorFlow dataset."""

    y_true_batches = []
    y_prob_batches = []

    for images, labels in dataset:
        batch_probs = model.predict(images, verbose=0)
        y_prob_batches.append(batch_probs)
        y_true_batches.append(labels.numpy())

    y_prob = np.concatenate(y_prob_batches, axis=0)
    y_true_one_hot = np.concatenate(y_true_batches, axis=0)

    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob, target_names=target_names)
    print_evaluation_metrics(y_true, y_pred, y_prob=y_prob, target_names=target_names)
    plot_confusion_matrix(y_true, y_pred, target_names=target_names)
    return metrics


def main() -> None:
    """Load the saved model and evaluate it on the validation dataset."""

    model = load_keras_model()
    dataset, unique_breeds = prepare_validation_dataset()
    metrics = evaluate_model_on_dataset(model, dataset, target_names=unique_breeds)

    print("\nSummary keys:", sorted(metrics.keys()))


if __name__ == "__main__":
    main()
