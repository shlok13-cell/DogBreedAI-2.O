"""Utility functions and path configurations for the Dog Breed Classifier application."""

import os
from typing import Union

import numpy as np

# Path definitions and image configuration constants
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.path.join(BASE_DIR, "dog_breed_model.keras")
BREEDS_PATH: str = os.path.join(BASE_DIR, "dog_breeds.npy")
IMG_SIZE: int = 224


def format_file_size(file_size_bytes: int) -> str:
    """Format file size in bytes to human-readable string (KB or MB).

    Args:
        file_size_bytes (int): Size of the file in bytes.

    Returns:
        str: Formatted file size string (e.g., '124.5 KB' or '3.20 MB').
    """
    if file_size_bytes < 1024 * 1024:
        return f"{file_size_bytes / 1024:.1f} KB"
    return f"{file_size_bytes / (1024 * 1024):.2f} MB"


def format_breed_name(breed_name: Union[str, np.str_]) -> str:
    """Format raw dog breed label into title case with spaces.

    Args:
        breed_name (Union[str, np.str_]): Raw breed string, typically underscore-separated.

    Returns:
        str: Cleaned human-readable breed title (e.g., 'golden_retriever' -> 'Golden Retriever').
    """
    return str(breed_name).replace("_", " ").title()
