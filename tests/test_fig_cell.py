"""
Tests for the FIG-Cell optical alignment system.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from app.fig_cell.optical_system import check_bottle_alignment, get_alignment_guidance
from config.config import ALIGNMENT_TOLERANCE_PX, TILT_ANGLE_MAX_DEG


# ── Image builders ────────────────────────────────────────────────────────────


def make_centred_bottle_image(
    height: int = 500,
    width: int = 200,
    offset_x: int = 0,
) -> np.ndarray:
    """Centred bottle image, optionally shifted horizontally by *offset_x* px."""
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    cx = width // 2 + offset_x
    bw = 60
    cv2.rectangle(img, (cx - bw // 2, 10), (cx + bw // 2, height - 10), (50, 50, 50), 4)
    cv2.rectangle(img, (cx - bw // 2 + 2, 12), (cx + bw // 2 - 2, height - 12),
                  (200, 160, 40), -1)
    return img


def make_empty_image(height: int = 200, width: int = 200) -> np.ndarray:
    """Blank white image – no bottle present."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def make_random_image(height: int = 200, width: int = 200) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# ── check_bottle_alignment tests ──────────────────────────────────────────────


class TestCheckBottleAlignment:
    def test_returns_dict_with_required_keys(self):
        img = make_centred_bottle_image()
        result = check_bottle_alignment(img)
        for key in ("bottle_detected", "is_aligned", "centre_offset_px",
                    "tilt_angle_deg", "bottle_height_ratio", "message"):
            assert key in result, f"Missing key: {key}"

    def test_numeric_fields_are_float(self):
        img = make_centred_bottle_image()
        result = check_bottle_alignment(img)
        assert isinstance(result["centre_offset_px"], float)
        assert isinstance(result["tilt_angle_deg"], float)
        assert isinstance(result["bottle_height_ratio"], float)

    def test_tilt_angle_non_negative(self):
        img = make_centred_bottle_image()
        result = check_bottle_alignment(img)
        assert result["tilt_angle_deg"] >= 0.0

    def test_bottle_height_ratio_in_range(self):
        img = make_centred_bottle_image()
        result = check_bottle_alignment(img)
        assert 0.0 <= result["bottle_height_ratio"] <= 1.0

    def test_message_is_string(self):
        img = make_centred_bottle_image()
        result = check_bottle_alignment(img)
        assert isinstance(result["message"], str)

    def test_random_image_does_not_crash(self):
        img = make_random_image()
        result = check_bottle_alignment(img)
        assert isinstance(result, dict)

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            check_bottle_alignment(None)

    def test_invalid_grayscale_raises_value_error(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            check_bottle_alignment(gray)


# ── get_alignment_guidance tests ──────────────────────────────────────────────


class TestGetAlignmentGuidance:
    def test_no_bottle_returns_reject(self):
        alignment = {
            "bottle_detected": False,
            "is_aligned": False,
            "centre_offset_px": 0.0,
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.0,
            "message": "No bottle detected.",
        }
        guidance = get_alignment_guidance(alignment)
        assert guidance["action"] == "REJECT"
        assert guidance["direction"] is None

    def test_aligned_bottle_returns_ok(self):
        alignment = {
            "bottle_detected": True,
            "is_aligned": True,
            "centre_offset_px": 0.0,
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.8,
            "message": "FIG-Cell alignment OK.",
        }
        guidance = get_alignment_guidance(alignment)
        assert guidance["action"] == "OK"

    def test_offset_left_suggests_right(self):
        alignment = {
            "bottle_detected": True,
            "is_aligned": False,
            "centre_offset_px": -(ALIGNMENT_TOLERANCE_PX + 10),
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.8,
            "message": "",
        }
        guidance = get_alignment_guidance(alignment)
        assert guidance["action"] == "REPOSITION"
        assert guidance["direction"] == "right"

    def test_offset_right_suggests_left(self):
        alignment = {
            "bottle_detected": True,
            "is_aligned": False,
            "centre_offset_px": ALIGNMENT_TOLERANCE_PX + 10,
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.8,
            "message": "",
        }
        guidance = get_alignment_guidance(alignment)
        assert guidance["action"] == "REPOSITION"
        assert guidance["direction"] == "left"

    def test_excess_tilt_returns_reject(self):
        alignment = {
            "bottle_detected": True,
            "is_aligned": False,
            "centre_offset_px": 0.0,
            "tilt_angle_deg": TILT_ANGLE_MAX_DEG + 5.0,
            "bottle_height_ratio": 0.8,
            "message": "",
        }
        guidance = get_alignment_guidance(alignment)
        assert guidance["action"] == "REJECT"

    def test_guidance_has_required_keys(self):
        alignment = {
            "bottle_detected": True,
            "is_aligned": True,
            "centre_offset_px": 0.0,
            "tilt_angle_deg": 0.0,
            "bottle_height_ratio": 0.8,
            "message": "OK",
        }
        guidance = get_alignment_guidance(alignment)
        for key in ("action", "direction", "guidance"):
            assert key in guidance
