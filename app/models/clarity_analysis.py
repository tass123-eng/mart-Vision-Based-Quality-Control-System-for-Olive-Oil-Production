"""
Model 3 – Oil clarity and impurity analysis.

Evaluates the optical quality of the olive oil inside the bottle by:

1. Isolating the oil region using the bottle contour.
2. Converting to HSV and measuring color consistency against the expected
   golden-yellow hue range of quality olive oil.
3. Computing a turbidity score from the variance in the saturation channel:
   turbid (cloudy) oil shows high local variance.
4. Detecting dark impurity particles using morphological operations on a
   threshold of the value channel.

Returns
-------
dict with keys:
    ``is_valid``        – bool, True when all clarity checks pass.
    ``clarity_score``   – float in [0, 100], higher is clearer.
    ``turbidity_score`` – float in [0, 100], lower is better.
    ``impurity_ratio``  – float in [0, 1], fraction of oil area with impurities.
    ``color_valid``    – bool, oil color within expected hue range.
    ``message``         – human-readable verdict.
"""

import cv2
import numpy as np

from config.config import (
    TURBIDITY_THRESHOLD,
    IMPURITY_AREA_MAX_RATIO,
    CLARITY_SCORE_MIN,
    HUE_MIN,
    HUE_MAX,
)


def _extract_oil_region(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of the oil region within the bottle.

    The oil region is approximated as the central 60 % of the image width
    and the lower 70 % of the image height (below any cap/neck area).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    y1 = int(h * 0.15)
    y2 = int(h * 0.90)
    mask[y1:y2, x1:x2] = 255
    return mask


def _compute_turbidity(hsv_roi: np.ndarray) -> float:
    """Compute a turbidity score [0, 100] from local saturation variance.

    Turbid oil scatters light unevenly, producing high local variance in
    both the saturation and value channels.
    """
    s_channel = hsv_roi[:, :, 1]
    local_var = cv2.Laplacian(s_channel, cv2.CV_32F).var()
    score = float(np.clip(local_var / 5.0, 0.0, 100.0))
    return score


def _detect_impurities(hsv_roi: np.ndarray, oil_mask_roi: np.ndarray) -> float:
    """Return the ratio of impurity pixels to total oil-region pixels.

    Dark particles (impurities, sediment) have a very low value in HSV.
    """
    v_channel = hsv_roi[:, :, 2]
    _, dark = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    impurity_pixels = int(cv2.countNonZero(cv2.bitwise_and(cleaned, oil_mask_roi)))
    oil_pixels = int(cv2.countNonZero(oil_mask_roi))

    if oil_pixels == 0:
        return 0.0
    return float(impurity_pixels) / float(oil_pixels)


def _check_color(hsv_roi: np.ndarray, oil_mask_roi: np.ndarray) -> bool:
    """Check whether the median hue of the oil region is within range."""
    h_channel = hsv_roi[:, :, 0]
    oil_hues = h_channel[oil_mask_roi > 0]
    if oil_hues.size == 0:
        return False
    median_hue = float(np.median(oil_hues))
    return HUE_MIN <= median_hue <= HUE_MAX


def analyze_clarity(image: np.ndarray) -> dict:
    """Analyse the clarity and purity of olive oil in a bottle image.

    Parameters
    ----------
    image:
        BGR uint8 image of the bottle.

    Returns
    -------
    dict
        ``is_valid``, ``clarity_score``, ``turbidity_score``,
        ``impurity_ratio``, ``color_valid``, ``message``.

    Raises
    ------
    ValueError
        If *image* is not a valid 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    oil_mask = _extract_oil_region(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    turbidity_score = _compute_turbidity(hsv)
    impurity_ratio = _detect_impurities(hsv, oil_mask)
    color_valid = _check_color(hsv, oil_mask)

    clarity_score = float(
        np.clip(
            100.0
            - turbidity_score * 0.5
            - impurity_ratio * 100.0 * 0.5,
            0.0,
            100.0,
        )
    )

    is_valid = (
        turbidity_score <= TURBIDITY_THRESHOLD
        and impurity_ratio <= IMPURITY_AREA_MAX_RATIO
        and clarity_score >= CLARITY_SCORE_MIN
        and color_valid
    )

    issues = []
    if turbidity_score > TURBIDITY_THRESHOLD:
        issues.append(f"turbidity too high ({turbidity_score:.1f} > {TURBIDITY_THRESHOLD})")
    if impurity_ratio > IMPURITY_AREA_MAX_RATIO:
        issues.append(
            f"impurity ratio too high ({impurity_ratio:.3%} > {IMPURITY_AREA_MAX_RATIO:.3%})"
        )
    if clarity_score < CLARITY_SCORE_MIN:
        issues.append(f"clarity score too low ({clarity_score:.1f} < {CLARITY_SCORE_MIN})")
    if not color_valid:
        issues.append("oil color outside expected range")

    if is_valid:
        message = f"Oil clarity OK (score {clarity_score:.1f}/100)."
    else:
        message = "Oil quality issue(s): " + ", ".join(issues) + "."

    return {
        "is_valid": is_valid,
        "clarity_score": clarity_score,
        "turbidity_score": turbidity_score,
        "impurity_ratio": impurity_ratio,
        "color_valid": color_valid,
        "message": message,
    }
