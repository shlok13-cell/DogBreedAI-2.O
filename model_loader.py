"""Model loading utilities for the Dog Breed Classifier."""

import os
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tf_keras as keras
import tf_keras.layers as tf_layers
# Patch InputLayer.from_config to strip Keras 3 'optional' argument
_orig_input_from_config = tf_layers.InputLayer.from_config
def _patched_input_from_config(config):
    if isinstance(config, dict):
        config.pop('optional', None)
    return _orig_input_from_config(config)
tf_layers.InputLayer.from_config = _patched_input_from_config

from utils import BREEDS_PATH, MODEL_PATH


@st.cache_resource(show_spinner=True)
def load_model_and_labels(
    model_path: str = MODEL_PATH, breeds_path: str = BREEDS_PATH
) -> Tuple[keras.Model, np.ndarray]:
    """Load the trained Keras dog classification model and class labels array.

    Args:
        model_path (str, optional): Absolute filepath to saved Keras model (.keras). Defaults to MODEL_PATH.
        breeds_path (str, optional): Absolute filepath to saved breed labels (.npy). Defaults to BREEDS_PATH.

    Raises:
        FileNotFoundError: If either the model or breed label file does not exist.

    Returns:
        Tuple[keras.Model, np.ndarray]: Loaded Keras Model object and numpy array of breed class labels.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train and save the model by running main.py first."
        )
    if not os.path.exists(breeds_path):
        raise FileNotFoundError(
            f"Breed label file not found at {breeds_path}. Train and save the model by running main.py first."
        )

    custom_objects = {
        "KerasLayer": hub.KerasLayer,
        "keras_layer": hub.KerasLayer,
        "hub>KerasLayer": hub.KerasLayer,
        "tensorflow_hub>KerasLayer": hub.KerasLayer,
    }
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    breeds = np.load(breeds_path, allow_pickle=True)
    return model, breeds
