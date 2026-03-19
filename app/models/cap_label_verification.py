"""
Model 2 – Cap and label verification.

Verifies that:
* The bottle cap is present, correctly sized, and centred at the top.
* The label is present, correctly positioned in the middle of the bottle,
  and properly aligned (not tilted or offset).

Algorithm
---------
Cap detection
    Crop the top region of the image and find the dominant bright/dark
    elliptical object using contour analysis.

Label detection
    Crop the middle region of the image and look for a large rectangular
    region with consistent internal color (printed label area).

Returns
-------
dict with keys:
    ``cap_present``     – bool
    ``cap_valid``       – bool (present *and* correctly positioned)
    ``label_present``   – bool
    ``label_valid``     – bool (present *and* correctly aligned)
    ``is_valid``        – bool (all checks pass)
    ``message``         – human-readable verdict
"""

import cv2
import numpy as np

from config.config import (
    CAP_AREA_MIN_RATIO,
    CAP_AREA_MAX_RATIO,
    LABEL_AREA_MIN_RATIO,
    LABEL_AREA_MAX_RATIO,
    LABEL_ALIGNMENT_TOLERANCE,
)


def _detect_cap(image: np.ndarray):
    """Detect the bottle cap in the top portion of the image.

    Returns ``(present, valid, bounding_rect)`` or ``(False, False, None)``.
    """
    h, w = image.shape[:2]
    cap_region = image[: int(h * 0.20), :]
    img_area = h * w

    gray = cv2.cvtColor(cap_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, False, None

    candidates = [
        c for c in contours
        if CAP_AREA_MIN_RATIO * img_area < cv2.contourArea(c) < CAP_AREA_MAX_RATIO * img_area
    ]
    if not candidates:
        return False, False, None

    best = max(candidates, key=cv2.contourArea)
    rect = cv2.boundingRect(best)
    bx, _, bw, _ = rect

    cap_centre_x = bx + bw / 2
    img_centre_x = w / 2
    offset_ratio = abs(cap_centre_x - img_centre_x) / w

    valid = offset_ratio < LABEL_ALIGNMENT_TOLERANCE * 2
    return True, valid, rect


def _detect_label(image: np.ndarray):
    """Detect the label in the middle portion of the image.

    Returns ``(present, valid, bounding_rect)`` or ``(False, False, None)``.
    """
    h, w = image.shape[:2]
    label_region = image[int(h * 0.25) : int(h * 0.75), :]
    img_area = h * w

    gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 20, 80)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, False, None

    candidates = [
        c for c in contours
        if LABEL_AREA_MIN_RATIO * img_area < cv2.contourArea(c) < LABEL_AREA_MAX_RATIO * img_area
    ]
    if not candidates:
        return False, False, None

    best = max(candidates, key=cv2.contourArea)
    rect = cv2.boundingRect(best)
    bx, _, bw, _ = rect

    label_centre_x = bx + bw / 2
    img_centre_x = w / 2
    offset_ratio = abs(label_centre_x - img_centre_x) / w

    valid = offset_ratio < LABEL_ALIGNMENT_TOLERANCE
    return True, valid, rect


def verify_cap_and_label(image: np.ndarray) -> dict:
    """Verify the cap and label of an olive oil bottle.

    Parameters
    ----------
    image:
        BGR uint8 image of the bottle.

    Returns
    -------
    dict
        ``cap_present``, ``cap_valid``, ``label_present``, ``label_valid``,
        ``is_valid``, ``message``.

    Raises
    ------
    ValueError
        If *image* is not a valid 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    cap_present, cap_valid, _ = _detect_cap(image)
    label_present, label_valid, _ = _detect_label(image)

    is_valid = cap_valid and label_valid

    issues = []
    if not cap_present:
        issues.append("cap missing")
    elif not cap_valid:
        issues.append("cap misaligned")
    if not label_present:
        issues.append("label missing")
    elif not label_valid:
        issues.append("label misaligned")

    if is_valid:
        message = "Cap and label verification passed."
    else:
        message = "Cap/label defect(s) detected: " + ", ".join(issues) + "."

    return {
        "cap_present": cap_present,
        "cap_valid": cap_valid,
        "label_present": label_present,
        "label_valid": label_valid,
        "is_valid": is_valid,
        "message": message,
    }
