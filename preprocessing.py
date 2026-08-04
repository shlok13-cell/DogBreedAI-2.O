"""Image preprocessing utilities for the Dog Breed Classifier model."""

from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import IMG_SIZE


def preprocess_image(
    image: Image.Image, target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)
) -> tf.Tensor:
    """Preprocess a PIL Image into a normalized tensor ready for model inference.

    Args:
        image (Image.Image): Input image loaded via PIL.
        target_size (Tuple[int, int], optional): Target (height, width) dimensions. Defaults to (224, 224).

    Returns:
        tf.Tensor: Preprocessed image tensor of shape (1, height, width, 3) with float32 values in range [0, 1].
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    return tf.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
