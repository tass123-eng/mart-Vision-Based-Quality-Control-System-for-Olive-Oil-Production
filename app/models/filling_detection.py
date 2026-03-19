"""
Model 1 – Filling-level detection.

Analyses whether the olive oil bottle is filled to the correct level using
classical computer-vision techniques:

1. Detect the bottle contour to establish the region of interest (ROI).
2. Convert the ROI to grayscale and apply an edge-detection pass.
3. Locate the liquid surface by finding the dominant horizontal edge in the
   upper portion of the bottle body.
4. Compute the fill ratio and compare it against the configured thresholds.

Returns
-------
dict with keys:
    ``is_valid``      – bool, True when fill level is within limits.
    ``fill_ratio``    – float in [0, 1], measured fill fraction.
    ``fill_level_px`` – int, y-coordinate of the detected liquid surface.
    ``message``       – human-readable verdict.
"""

import cv2
import numpy as np

from config.config import (
    FILL_LEVEL_MIN,
    FILL_LEVEL_MAX,
    FILL_ROI_TOP_RATIO,
    FILL_ROI_BOTTOM_RATIO,
)


def _detect_bottle_roi(image: np.ndarray):
    """Return the bounding rectangle of the largest vertical contour.

    Returns ``(x, y, w, h)`` or ``None`` if no plausible bottle is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = image.shape[0] * image.shape[1]
    candidates = [
        c for c in contours
        if cv2.contourArea(c) > 0.01 * img_area
    ]
    if not candidates:
        return None

    largest = max(candidates, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def _find_liquid_surface(gray_roi: np.ndarray) -> int:
    """Return the y-coordinate (within *gray_roi*) of the liquid surface.

    Uses horizontal Sobel edges: the liquid surface is the dominant strong
    horizontal edge in the upper half of the bottle body.
    """
    sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel = np.abs(sobel_y)

    search_region = abs_sobel[: gray_roi.shape[0] // 2, :]
    row_energy = search_region.sum(axis=1)
    return int(np.argmax(row_energy))


def analyze_fill_level(image: np.ndarray) -> dict:
    """Analyse the fill level of an olive oil bottle.

    Parameters
    ----------
    image:
        BGR uint8 image of the bottle.

    Returns
    -------
    dict
        ``is_valid``, ``fill_ratio``, ``fill_level_px``, ``message``.

    Raises
    ------
    ValueError
        If *image* is not a valid 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    h, w = image.shape[:2]
    roi_rect = _detect_bottle_roi(image)

    if roi_rect is None:
        return {
            "is_valid": False,
            "fill_ratio": 0.0,
            "fill_level_px": 0,
            "message": "Bottle not detected in the image.",
        }

    bx, by, bw, bh = roi_rect

    top = by + int(bh * FILL_ROI_TOP_RATIO)
    bottom = by + int(bh * FILL_ROI_BOTTOM_RATIO)
    body_roi = image[top:bottom, bx : bx + bw]

    if body_roi.size == 0:
        return {
            "is_valid": False,
            "fill_ratio": 0.0,
            "fill_level_px": 0,
            "message": "Could not extract bottle body region.",
        }

    gray_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
    surface_y_in_roi = _find_liquid_surface(gray_body)
    surface_y_abs = top + surface_y_in_roi

    body_height = bottom - top
    fill_ratio = 1.0 - (surface_y_in_roi / body_height) if body_height > 0 else 0.0
    fill_ratio = float(np.clip(fill_ratio, 0.0, 1.0))

    is_valid = FILL_LEVEL_MIN <= fill_ratio <= FILL_LEVEL_MAX

    if fill_ratio < FILL_LEVEL_MIN:
        message = f"Underfilled: fill ratio {fill_ratio:.2%} is below minimum {FILL_LEVEL_MIN:.2%}."
    elif fill_ratio > FILL_LEVEL_MAX:
        message = f"Overfilled: fill ratio {fill_ratio:.2%} exceeds maximum {FILL_LEVEL_MAX:.2%}."
    else:
        message = f"Fill level OK: {fill_ratio:.2%}."

    return {
        "is_valid": is_valid,
        "fill_ratio": fill_ratio,
        "fill_level_px": surface_y_abs,
        "message": message,
    }
