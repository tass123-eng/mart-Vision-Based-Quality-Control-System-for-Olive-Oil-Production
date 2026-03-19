"""
Shared image utility helpers.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes (JPEG / PNG / BMP / TIFF) into a BGR NumPy array.

    Parameters
    ----------
    data:
        Raw image bytes from an uploaded file or network stream.

    Returns
    -------
    numpy.ndarray
        BGR uint8 image.

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from the provided bytes.")
    return image


def encode_image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR NumPy array to a base64 string.

    Parameters
    ----------
    image:
        BGR uint8 NumPy array.
    fmt:
        Output format extension, e.g. ``".jpg"`` or ``".png"``.

    Returns
    -------
    str
        Base64-encoded image string (without MIME prefix).
    """
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError(f"Could not encode image to {fmt}.")
    return base64.b64encode(buffer).decode("utf-8")


def draw_result_overlay(image: np.ndarray, results: dict) -> np.ndarray:
    """Overlay quality-control results on the bottle image.

    Draws a colored border and a status text block on a copy of *image*.

    Parameters
    ----------
    image:
        BGR uint8 image.
    results:
        The combined result dict returned by the full inspection pipeline.

    Returns
    -------
    numpy.ndarray
        Annotated BGR uint8 image.
    """
    annotated = image.copy()
    overall_valid = results.get("overall_valid", False)
    border_color = (0, 200, 0) if overall_valid else (0, 0, 220)

    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), border_color, thickness=8)

    status_text = "PASS" if overall_valid else "FAIL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(w, h) / 400.0)
    thickness = max(2, int(font_scale * 2))

    text_size, _ = cv2.getTextSize(status_text, font, font_scale, thickness)
    text_x = (w - text_size[0]) // 2
    text_y = text_size[1] + 20

    cv2.rectangle(
        annotated,
        (text_x - 10, 5),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        annotated,
        status_text,
        (text_x, text_y),
        font,
        font_scale,
        border_color,
        thickness,
        cv2.LINE_AA,
    )

    detail_lines = [
        f"Fill: {'OK' if results.get('fill', {}).get('is_valid') else 'FAIL'}",
        f"Cap/Label: {'OK' if results.get('cap_label', {}).get('is_valid') else 'FAIL'}",
        f"Clarity: {'OK' if results.get('clarity', {}).get('is_valid') else 'FAIL'}",
        f"Alignment: {'OK' if results.get('alignment', {}).get('is_aligned') else 'WARN'}",
    ]

    line_height = int(20 * font_scale)
    for i, line in enumerate(detail_lines):
        cv2.putText(
            annotated,
            line,
            (10, h - 10 - (len(detail_lines) - 1 - i) * line_height),
            font,
            font_scale * 0.6,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return annotated


def resize_for_display(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Resize *image* so its longest dimension is at most *max_dim* pixels."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
