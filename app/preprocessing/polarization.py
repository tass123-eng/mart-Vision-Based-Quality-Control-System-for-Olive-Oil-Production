"""
Cross-polarization simulation for glare suppression on glass bottles.

Real cross-polarization uses two polarizing filters oriented at 90° to each
other. This module simulates the effect computationally so that the rest of
the pipeline can be developed and tested without dedicated hardware.

Algorithm
---------
1. Convert the image to HSV color space.
2. Detect highlight regions (very bright pixels) using an adaptive threshold
   derived from the *HIGHLIGHT_SUPPRESSION_PERCENTILE* config value.
3. Suppress those regions by reducing the V-channel in proportion to the
   *POLARIZATION_FILTER_STRENGTH* setting.
4. Convert back to BGR.
"""

import cv2
import numpy as np

from config.config import POLARIZATION_FILTER_STRENGTH, HIGHLIGHT_SUPPRESSION_PERCENTILE


def apply_cross_polarization(image: np.ndarray) -> np.ndarray:
    """Return a glare-suppressed version of *image*.

    Parameters
    ----------
    image:
        BGR image as a ``uint8`` NumPy array with shape ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        Processed BGR image of the same shape and dtype.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_channel = hsv[:, :, 2]

    threshold = float(np.percentile(v_channel, HIGHLIGHT_SUPPRESSION_PERCENTILE))
    # Clamp below 255 so that pixels exactly at the threshold value are still
    # suppressed even when the percentile lands at the maximum possible value.
    threshold = min(threshold, 254.0)

    highlight_mask = v_channel >= threshold
    v_range = 255.0 - threshold  # guaranteed >= 1.0 because threshold <= 254
    suppression = np.where(
        highlight_mask,
        POLARIZATION_FILTER_STRENGTH * (v_channel - threshold) / v_range,
        0.0,
    )
    v_channel = np.clip(v_channel - suppression * (v_channel - threshold), 0, 255)

    hsv[:, :, 2] = v_channel
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_internal_defects(image: np.ndarray) -> np.ndarray:
    """Increase visibility of internal and external bottle defects.

    Applies a sharpening kernel after glare suppression to enhance edges
    corresponding to cracks, inclusions, and surface irregularities.

    Parameters
    ----------
    image:
        BGR image as a ``uint8`` NumPy array.

    Returns
    -------
    numpy.ndarray
        Sharpened BGR image.
    """
    polarized = apply_cross_polarization(image)

    sharpening_kernel = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(polarized, ddepth=-1, kernel=sharpening_kernel)
    return sharpened
