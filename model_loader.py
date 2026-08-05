"""Model loading utilities for the Dog Breed Classifier."""

import os
# Enable legacy Keras (Keras 2) compatibility – prevents 'optional' arg errors
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import types
try:
    import pkg_resources
except ImportError:
    try:
        from packaging import version
        _parse_v = version.parse
    except ImportError:
        _parse_v = lambda v: v
    _pkg_res = types.ModuleType("pkg_resources")
    _pkg_res.parse_version = _parse_v
    _pkg_res.PackagingVersion = _parse_v
    sys.modules["pkg_resources"] = _pkg_res

from typing import Tuple
import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tf_keras as keras
import tf_keras.layers as tf_layers

# Pre-register KerasLayer in both tf_keras and standard Keras global custom object registries
keras.utils.get_custom_objects()["KerasLayer"] = hub.KerasLayer
keras.utils.get_custom_objects()["keras_layer"] = hub.KerasLayer

try:
    import keras as _k3
    _k3.utils.get_custom_objects()["KerasLayer"] = hub.KerasLayer
    _k3.utils.get_custom_objects()["keras_layer"] = hub.KerasLayer
except Exception:
    pass

# Patch InputLayer.from_config to strip Keras 3 'optional' argument if present
_orig_input_from_config = tf_layers.InputLayer.from_config
def _patched_input_from_config(config):
    if isinstance(config, dict):
        config.pop("optional", None)
    return _orig_input_from_config(config)
tf_layers.InputLayer.from_config = _patched_input_from_config

# MobileNetV2 feature vector URL used during model training
HUB_MODULE_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5"

from utils import BREEDS_PATH, MODEL_PATH


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

    # Ensure MobileNetV2 TF-Hub module weights are downloaded and cached before deserialization
    try:
        _ = hub.KerasLayer(HUB_MODULE_URL, trainable=False)
    except Exception:
        pass

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
