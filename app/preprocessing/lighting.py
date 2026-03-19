"""
Adaptive lighting normalization for uniform image acquisition.

The FIG-Cell uses controlled LED panels, but ambient variations and bottle
color differences still affect raw images. This module compensates for
those variations so that downstream models see consistent intensity levels.

Two normalization modes are provided:

* ``normalize_global`` – simple affine rescaling to reach a target mean and
  standard deviation across the whole image.
* ``normalize_clahe`` – Contrast-Limited Adaptive Histogram Equalization
  (CLAHE) applied per-channel in the LAB color space for perceptually
  uniform enhancement without color distortion.
"""

import cv2
import numpy as np

from config.config import (
    TARGET_MEAN_BRIGHTNESS,
    TARGET_STD_BRIGHTNESS,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)


def normalize_global(image: np.ndarray) -> np.ndarray:
    """Affine-rescale *image* to match target mean/std brightness.

    Parameters
    ----------
    image:
        BGR uint8 image.

    Returns
    -------
    numpy.ndarray
        Normalized BGR uint8 image.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    img_float = image.astype(np.float32)
    mean = img_float.mean()
    std = img_float.std()

    if std < 1e-6:
        std = 1.0

    normalized = (img_float - mean) * (TARGET_STD_BRIGHTNESS / std) + TARGET_MEAN_BRIGHTNESS
    return np.clip(normalized, 0, 255).astype(np.uint8)


def normalize_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE in LAB space for adaptive contrast enhancement.

    Works in the LAB color space so that only luminance is equalized,
    preserving the oil's natural color for downstream analysis.

    Parameters
    ----------
    image:
        BGR uint8 image.

    Returns
    -------
    numpy.ndarray
        Contrast-enhanced BGR uint8 image.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def apply_lighting_normalization(image: np.ndarray) -> np.ndarray:
    """Apply the full lighting normalization pipeline.

    Combines global normalization and CLAHE for optimal image quality.

    Parameters
    ----------
    image:
        BGR uint8 image.

    Returns
    -------
    numpy.ndarray
        Fully normalized BGR uint8 image.
    """
    globally_normalized = normalize_global(image)
    return normalize_clahe(globally_normalized)
