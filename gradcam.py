"""Grad-CAM (Gradient-weighted Class Activation Mapping) explanation module for Dog Breed Classifier."""

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
import tf_keras as keras
from PIL import Image


@st.cache_resource(show_spinner=False)
def load_gradcam_extractor() -> keras.Model:
    """Load and cache the MobileNetV2 feature extractor backbone for Grad-CAM map generation.

    Returns:
        keras.Model: Pre-trained MobileNetV2 model (alpha=1.4) without top classification head.
    """
    extractor = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        alpha=1.4,
        weights="imagenet",
    )
    return extractor


def compute_gradcam_heatmap(
    model: keras.Model,
    input_tensor: tf.Tensor,
    target_class_index: int,
) -> np.ndarray:
    """Compute 2D spatial Grad-CAM activation heatmap for a target predicted breed class index.

    Args:
        model (keras.Model): Trained dog classification model containing the output Dense layer.
        input_tensor (tf.Tensor): Preprocessed input image tensor of shape (1, 224, 224, 3).
        target_class_index (int): Index of the predicted breed class.

    Returns:
        np.ndarray: Normalized 2D spatial heatmap array of shape (7, 7) with values in range [0, 1].
    """
    extractor = load_gradcam_extractor()
    conv_features = extractor(input_tensor)  # shape (1, 7, 7, 1792)

    # Extract classification weights from the model's final Dense layer
    dense_layer = model.layers[2]
    weights, _ = dense_layer.get_weights()  # shape (1792, 120)

    # Class weight vector for the top prediction
    class_weights = weights[:, target_class_index]  # shape (1792,)

    # Compute dot product across channel activations
    cam = np.dot(conv_features[0].numpy(), class_weights)  # shape (7, 7)
    cam = np.maximum(cam, 0)  # ReLU activation

    # Min-max normalization
    cam_max = np.max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    return cam


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a 2D Grad-CAM heatmap onto a PIL image using a JET colormap blend.

    Args:
        original_image (Image.Image): Original input PIL Image.
        heatmap (np.ndarray): 2D normalized spatial heatmap array of shape (7, 7).
        alpha (float, optional): Heatmap transparency blending weight in range [0, 1]. Defaults to 0.45.

    Returns:
        Image.Image: Blended RGB PIL Image containing original image with Grad-CAM heatmap overlay.
    """
    orig_rgb = original_image.convert("RGB")
    width, height = orig_rgb.size

    # Resize 7x7 spatial heatmap to original image dimensions with bicubic interpolation
    heatmap_pil = Image.fromarray((heatmap * 255.0).astype(np.uint8))
    heatmap_resized = heatmap_pil.resize((width, height), resample=Image.Resampling.BICUBIC)
    heatmap_norm = np.array(heatmap_resized) / 255.0

    # Apply JET colormap to convert single-channel heatmap into RGB
    colored_heatmap_array = (cm.jet(heatmap_norm)[:, :, :3] * 255.0).astype(np.uint8)
    colored_heatmap_pil = Image.fromarray(colored_heatmap_array)

    # Blend original image and colormapped heatmap
    overlayed_image = Image.blend(orig_rgb, colored_heatmap_pil, alpha=alpha)
    return overlayed_image


def generate_gradcam_explanation(
    model: keras.Model,
    input_tensor: tf.Tensor,
    target_class_index: int,
    original_image: Image.Image,
) -> Image.Image:
    """High-level helper to generate a Grad-CAM explanation PIL Image for an analyzed dog photo.

    Args:
        model (keras.Model): Trained dog breed classifier model.
        input_tensor (tf.Tensor): Preprocessed image tensor of shape (1, 224, 224, 3).
        target_class_index (int): Index of the predicted breed class.
        original_image (Image.Image): Original user-uploaded PIL Image.

    Returns:
        Image.Image: PIL Image of the original photo overlaid with the Grad-CAM heatmap.
    """
    heatmap = compute_gradcam_heatmap(model, input_tensor, target_class_index)
    explanation_image = overlay_heatmap_on_image(original_image, heatmap)
    return explanation_image
