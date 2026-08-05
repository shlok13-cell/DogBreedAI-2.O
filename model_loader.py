"""Model loading utilities for the Dog Breed Classifier."""

import os
# Enable legacy Keras (Keras 2) compatibility – prevents 'optional' arg errors
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Fallback shim for pkg_resources required by tensorflow_hub
try:
    import pkg_resources
except ImportError:
    import sys
    import types
    from packaging import version
    _pkg_resources = types.ModuleType("pkg_resources")
    _pkg_resources.parse_version = version.parse
    sys.modules["pkg_resources"] = _pkg_resources

from typing import Tuple
import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tf_keras as keras

from utils import BREEDS_PATH, MODEL_PATH

# Pre-register KerasLayer in global custom object registries
keras.utils.get_custom_objects()["KerasLayer"] = hub.KerasLayer
keras.utils.get_custom_objects()["keras_layer"] = hub.KerasLayer


@st.cache_resource(show_spinner=True)
def load_model_and_labels(
    model_path: str = MODEL_PATH, breeds_path: str = BREEDS_PATH
) -> Tuple[keras.Model, np.ndarray]:
    """Load the trained Keras dog classification model and class labels array."""
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
