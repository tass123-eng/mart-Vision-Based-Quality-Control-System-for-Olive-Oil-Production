"""
FIG-Cell Optical Gravitaire – Bottle alignment and stabilisation system.

The FIG-Cell (Floating Image Gravity Cell) is a purpose-built optomechanical
concept that centres the bottle on a gravimetric cradle and acquires images
from a fixed focal distance. This module provides the software side of the
FIG-Cell: it analyses a raw camera frame, detects the bottle, measures its
geometric alignment, and returns corrective guidance (or raises an alert when
the misalignment is too large to correct in software).

Checks performed
----------------
* **Presence** – a plausible bottle contour must be found.
* **Centring** – the bottle axis must be within ``ALIGNMENT_TOLERANCE_PX``
  pixels of the horizontal image centre.
* **Height** – the bottle must occupy at least ``BOTTLE_HEIGHT_MIN_RATIO``
  of the image height (ensures correct focal distance).
* **Tilt** – the bottle's minimum bounding rectangle must be within
  ``TILT_ANGLE_MAX_DEG`` degrees of vertical.

Returns
-------
dict with keys:
    ``bottle_detected``  – bool
    ``is_aligned``       – bool (all geometric checks passed)
    ``centre_offset_px`` – float, horizontal offset from image centre
    ``tilt_angle_deg``   – float, tilt angle in degrees (0 = perfectly vertical)
    ``bottle_height_ratio`` – float, bottle height / image height
    ``message``          – human-readable status
"""

import cv2
import numpy as np

from config.config import (
    BOTTLE_HEIGHT_MIN_RATIO,
    BOTTLE_WIDTH_MIN_RATIO,
    ALIGNMENT_TOLERANCE_PX,
    TILT_ANGLE_MAX_DEG,
)


def _find_bottle_contour(image: np.ndarray):
    """Return the largest vertical contour that resembles a bottle.

    Returns the contour array or ``None``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image.shape[:2]
    min_area = BOTTLE_HEIGHT_MIN_RATIO * h * BOTTLE_WIDTH_MIN_RATIO * w

    candidates = [c for c in contours if cv2.contourArea(c) > min_area]
    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


def check_bottle_alignment(image: np.ndarray) -> dict:
    """Check whether the bottle is correctly positioned in the FIG-Cell frame.

    Parameters
    ----------
    image:
        BGR uint8 image captured by the FIG-Cell camera.

    Returns
    -------
    dict
        ``bottle_detected``, ``is_aligned``, ``centre_offset_px``,
        ``tilt_angle_deg``, ``bottle_height_ratio``, ``message``.

    Raises
    ------
    ValueError
        If *image* is not a valid 3-channel uint8 array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (BGR) uint8 image.")

    h, w = image.shape[:2]
    contour = _find_bottle_contour(image)

    if contour is None:
        return {
            "bottle_detected": False,
            "is_aligned": False,
            "centre_offset_px": 0.0,
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.0,
            "message": "No bottle detected in the frame.",
        }

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rect_w, rect_h), angle = rect

    if rect_w > rect_h:
        rect_w, rect_h = rect_h, rect_w
        angle += 90.0

    tilt_angle_deg = float(abs(angle % 90.0))
    if tilt_angle_deg > 45.0:
        tilt_angle_deg = 90.0 - tilt_angle_deg

    centre_offset_px = float(cx - w / 2)
    bottle_height_ratio = float(rect_h / h)

    alignment_ok = abs(centre_offset_px) <= ALIGNMENT_TOLERANCE_PX
    height_ok = bottle_height_ratio >= BOTTLE_HEIGHT_MIN_RATIO
    tilt_ok = tilt_angle_deg <= TILT_ANGLE_MAX_DEG
    is_aligned = alignment_ok and height_ok and tilt_ok

    issues = []
    if not alignment_ok:
        issues.append(
            f"horizontal offset {centre_offset_px:+.1f} px "
            f"(limit ±{ALIGNMENT_TOLERANCE_PX} px)"
        )
    if not height_ok:
        issues.append(
            f"bottle too small ({bottle_height_ratio:.1%} < {BOTTLE_HEIGHT_MIN_RATIO:.1%})"
        )
    if not tilt_ok:
        issues.append(f"tilt {tilt_angle_deg:.1f}° > {TILT_ANGLE_MAX_DEG}°")

    if is_aligned:
        message = "FIG-Cell alignment OK."
    else:
        message = "Alignment issue(s): " + ", ".join(issues) + "."

    return {
        "bottle_detected": True,
        "is_aligned": is_aligned,
        "centre_offset_px": centre_offset_px,
        "tilt_angle_deg": tilt_angle_deg,
        "bottle_height_ratio": bottle_height_ratio,
        "message": message,
    }


def get_alignment_guidance(alignment_result: dict) -> dict:
    """Translate an alignment result into corrective actions.

    Parameters
    ----------
    alignment_result:
        The dict returned by :func:`check_bottle_alignment`.

    Returns
    -------
    dict
        ``action``   – one of ``"OK"``, ``"REPOSITION"``, ``"REJECT"``.
        ``direction`` – ``"left"``, ``"right"``, or ``None``.
        ``guidance``  – human-readable instruction for the operator / conveyor.
    """
    if not alignment_result.get("bottle_detected"):
        return {
            "action": "REJECT",
            "direction": None,
            "guidance": "No bottle detected. Check camera and conveyor.",
        }

    if alignment_result.get("is_aligned"):
        return {
            "action": "OK",
            "direction": None,
            "guidance": "Bottle correctly positioned. Proceed with inspection.",
        }

    offset = alignment_result.get("centre_offset_px", 0.0)
    if offset < 0:
        direction = "right"
    else:
        direction = "left"

    tilt = alignment_result.get("tilt_angle_deg", 0.0)
    if tilt > TILT_ANGLE_MAX_DEG:
        return {
            "action": "REJECT",
            "direction": None,
            "guidance": f"Bottle tilt ({tilt:.1f}°) exceeds limit. Manual intervention required.",
        }

    return {
        "action": "REPOSITION",
        "direction": direction,
        "guidance": (
            f"Move bottle {direction} by "
            f"{abs(offset):.0f} px to centre it."
        ),
    }
