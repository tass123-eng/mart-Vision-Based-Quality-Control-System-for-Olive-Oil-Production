"""
Tests for the preprocessing modules:
  - Cross-polarization (polarization.py)
  - Lighting normalization (lighting.py)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.preprocessing.polarization import apply_cross_polarization, enhance_internal_defects
from app.preprocessing.lighting import (
    normalize_global,
    normalize_clahe,
    apply_lighting_normalization,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_grey_image():
    """A 100×100 uniform mid-grey BGR image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def bright_highlight_image():
    """A BGR image with a bright white spot (simulates glare)."""
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    img[20:40, 20:40] = 255  # bright highlight region
    return img


@pytest.fixture
def dark_image():
    """A very dark 100×100 BGR image."""
    return np.full((100, 100, 3), 20, dtype=np.uint8)


@pytest.fixture
def random_image():
    """A 200×200 random BGR image with known seed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)


# ── Cross-polarization tests ──────────────────────────────────────────────────


class TestCrossPolarization:
    def test_output_shape_matches_input(self, random_image):
        result = apply_cross_polarization(random_image)
        assert result.shape == random_image.shape

    def test_output_dtype_uint8(self, random_image):
        result = apply_cross_polarization(random_image)
        assert result.dtype == np.uint8

    def test_highlights_suppressed(self, bright_highlight_image):
        result = apply_cross_polarization(bright_highlight_image)
        highlight_region = result[20:40, 20:40]
        original_region = bright_highlight_image[20:40, 20:40]
        assert highlight_region.mean() < original_region.mean(), (
            "Highlight region should be dimmer after polarization."
        )

    def test_non_highlight_pixels_mostly_unchanged(self, bright_highlight_image):
        result = apply_cross_polarization(bright_highlight_image)
        non_highlight_orig = bright_highlight_image[60:80, 60:80].mean()
        non_highlight_result = result[60:80, 60:80].mean()
        assert abs(non_highlight_result - non_highlight_orig) < 20

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            apply_cross_polarization(None)

    def test_invalid_single_channel_raises_value_error(self):
        gray = np.full((50, 50), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_cross_polarization(gray)

    def test_enhance_internal_defects_returns_uint8(self, random_image):
        result = enhance_internal_defects(random_image)
        assert result.dtype == np.uint8
        assert result.shape == random_image.shape


# ── Lighting normalization tests ──────────────────────────────────────────────


class TestLightingNormalization:
    def test_normalize_global_output_shape(self, random_image):
        result = normalize_global(random_image)
        assert result.shape == random_image.shape

    def test_normalize_global_output_dtype(self, random_image):
        result = normalize_global(random_image)
        assert result.dtype == np.uint8

    def test_normalize_global_mean_closer_to_target(self, dark_image):
        from config.config import TARGET_MEAN_BRIGHTNESS
        result = normalize_global(dark_image)
        original_diff = abs(float(dark_image.mean()) - TARGET_MEAN_BRIGHTNESS)
        result_diff = abs(float(result.mean()) - TARGET_MEAN_BRIGHTNESS)
        assert result_diff < original_diff, (
            "Global normalization should bring mean closer to target."
        )

    def test_normalize_clahe_output_shape(self, random_image):
        result = normalize_clahe(random_image)
        assert result.shape == random_image.shape

    def test_normalize_clahe_output_dtype(self, random_image):
        result = normalize_clahe(random_image)
        assert result.dtype == np.uint8

    def test_apply_lighting_normalization_shape(self, random_image):
        result = apply_lighting_normalization(random_image)
        assert result.shape == random_image.shape

    def test_apply_lighting_normalization_dtype(self, random_image):
        result = apply_lighting_normalization(random_image)
        assert result.dtype == np.uint8

    def test_normalize_global_invalid_input(self):
        with pytest.raises(ValueError):
            normalize_global(None)

    def test_normalize_clahe_invalid_input(self):
        with pytest.raises(ValueError):
            normalize_clahe(np.zeros((50, 50), dtype=np.uint8))

    def test_uniform_image_is_handled(self, uniform_grey_image):
        result = normalize_global(uniform_grey_image)
        assert result.dtype == np.uint8
        assert result.shape == uniform_grey_image.shape
