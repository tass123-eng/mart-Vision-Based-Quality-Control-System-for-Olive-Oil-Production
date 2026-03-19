"""
Tests for the three AI quality-control models:
  - Filling detection
  - Cap and label verification
  - Clarity analysis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from app.models.filling_detection import analyze_fill_level
from app.models.cap_label_verification import verify_cap_and_label
from app.models.clarity_analysis import analyze_clarity


# ── Image builders ────────────────────────────────────────────────────────────


def make_bottle_image(height: int = 400, width: int = 150) -> np.ndarray:
    """Synthetic bottle image: white background, dark bottle outline, golden oil."""
    img = np.full((height, width, 3), 230, dtype=np.uint8)

    # Bottle body (dark grey outline)
    cv2.rectangle(img, (30, 20), (width - 30, height - 20), (60, 60, 60), 3)

    # Oil region (golden color – fill 90 % of bottle body)
    oil_top = 20 + int((height - 40) * 0.10)
    oil_color = (30, 140, 200)  # BGR ≈ golden-yellow
    cv2.rectangle(img, (31, oil_top), (width - 31, height - 21), oil_color, -1)

    # Cap (dark block at top)
    cv2.rectangle(img, (25, 5), (width - 25, 25), (40, 40, 40), -1)

    # Label (mid-section lighter rectangle)
    label_top = height // 3
    label_bot = 2 * height // 3
    cv2.rectangle(img, (32, label_top), (width - 32, label_bot), (200, 180, 160), -1)
    cv2.rectangle(img, (32, label_top), (width - 32, label_bot), (80, 80, 80), 2)

    return img


def make_random_image(height: int = 200, width: int = 200) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# ── Filling detection tests ───────────────────────────────────────────────────


class TestFillingDetection:
    def test_returns_dict_with_required_keys(self):
        img = make_bottle_image()
        result = analyze_fill_level(img)
        assert isinstance(result, dict)
        for key in ("is_valid", "fill_ratio", "fill_level_px", "message"):
            assert key in result, f"Missing key: {key}"

    def test_fill_ratio_in_valid_range(self):
        img = make_bottle_image()
        result = analyze_fill_level(img)
        assert 0.0 <= result["fill_ratio"] <= 1.0

    def test_fill_level_px_non_negative(self):
        img = make_bottle_image()
        result = analyze_fill_level(img)
        assert result["fill_level_px"] >= 0

    def test_message_is_string(self):
        img = make_bottle_image()
        result = analyze_fill_level(img)
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 0

    def test_random_image_does_not_crash(self):
        img = make_random_image()
        result = analyze_fill_level(img)
        assert isinstance(result, dict)

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            analyze_fill_level(None)

    def test_invalid_grayscale_raises_value_error(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            analyze_fill_level(gray)


# ── Cap and label verification tests ─────────────────────────────────────────


class TestCapLabelVerification:
    def test_returns_dict_with_required_keys(self):
        img = make_bottle_image()
        result = verify_cap_and_label(img)
        for key in ("cap_present", "cap_valid", "label_present", "label_valid",
                    "is_valid", "message"):
            assert key in result, f"Missing key: {key}"

    def test_booleans_are_bool(self):
        img = make_bottle_image()
        result = verify_cap_and_label(img)
        assert isinstance(result["cap_present"], bool)
        assert isinstance(result["cap_valid"], bool)
        assert isinstance(result["label_present"], bool)
        assert isinstance(result["label_valid"], bool)
        assert isinstance(result["is_valid"], bool)

    def test_is_valid_consistent_with_sub_flags(self):
        img = make_bottle_image()
        result = verify_cap_and_label(img)
        if result["is_valid"]:
            assert result["cap_valid"] and result["label_valid"]

    def test_message_is_string(self):
        img = make_bottle_image()
        result = verify_cap_and_label(img)
        assert isinstance(result["message"], str)

    def test_random_image_does_not_crash(self):
        img = make_random_image()
        result = verify_cap_and_label(img)
        assert isinstance(result, dict)

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            verify_cap_and_label(None)


# ── Clarity analysis tests ────────────────────────────────────────────────────


class TestClarityAnalysis:
    def test_returns_dict_with_required_keys(self):
        img = make_bottle_image()
        result = analyze_clarity(img)
        for key in ("is_valid", "clarity_score", "turbidity_score",
                    "impurity_ratio", "color_valid", "message"):
            assert key in result, f"Missing key: {key}"

    def test_scores_in_range(self):
        img = make_bottle_image()
        result = analyze_clarity(img)
        assert 0.0 <= result["clarity_score"] <= 100.0
        assert 0.0 <= result["turbidity_score"] <= 100.0
        assert 0.0 <= result["impurity_ratio"] <= 1.0

    def test_color_valid_is_bool(self):
        img = make_bottle_image()
        result = analyze_clarity(img)
        assert isinstance(result["color_valid"], bool)

    def test_message_is_string(self):
        img = make_bottle_image()
        result = analyze_clarity(img)
        assert isinstance(result["message"], str)

    def test_high_impurity_image_flags_fail(self):
        """Image with dark blobs should score low on clarity."""
        img = make_bottle_image(400, 200)
        # Add many dark impurity particles in the oil region
        for y in range(80, 320, 15):
            for x in range(50, 150, 15):
                cv2.circle(img, (x, y), 5, (10, 10, 10), -1)
        result = analyze_clarity(img)
        assert result["impurity_ratio"] > 0.0

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            analyze_clarity(None)

    def test_random_image_does_not_crash(self):
        img = make_random_image()
        result = analyze_clarity(img)
        assert isinstance(result, dict)
