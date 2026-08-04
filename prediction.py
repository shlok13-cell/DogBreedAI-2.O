"""Inference and post-processing prediction functions for the Dog Breed Classifier."""

from typing import Tuple

import numpy as np
import tensorflow as tf
import tf_keras as keras


def predict_breed(model: keras.Model, input_tensor: tf.Tensor) -> np.ndarray:
    """Run model inference on a preprocessed input image tensor.

    Args:
        model (keras.Model): Trained Keras model.
        input_tensor (tf.Tensor): Preprocessed image tensor of shape (1, H, W, 3).

    Returns:
        np.ndarray: 1D array of class probability scores of shape (num_classes,).
    """
    predictions = model.predict(input_tensor, verbose=0)[0]
    return predictions


def get_top_prediction_index(predictions: np.ndarray) -> int:
    """Return the index of the highest probability prediction score.

    Args:
        predictions (np.ndarray): 1D array of prediction probability scores.

    Returns:
        int: Index of the highest scoring class.
    """
    return int(np.argmax(predictions))


def get_top_k_predictions(
    predictions: np.ndarray, breeds: np.ndarray, top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-K highest probability breeds and their confidence scores.

    Args:
        predictions (np.ndarray): 1D array of prediction probability scores.
        breeds (np.ndarray): 1D array of breed label names.
        top_k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of top-K breed names and array of top-K confidence scores.
    """
    top_indices = predictions.argsort()[::-1][:top_k]
    top_breeds = breeds[top_indices]
    top_scores = predictions[top_indices]
    return top_breeds, top_scores
