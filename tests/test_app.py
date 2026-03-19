"""
Tests for the Flask web application (app/main.py).
"""

import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from app.main import app, run_full_inspection


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _make_jpeg_bytes(height: int = 400, width: int = 150) -> bytes:
    """Create a synthetic bottle image as JPEG bytes."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (30, 20), (width - 30, height - 20), (60, 60, 60), 3)
    oil_top = 20 + int((height - 40) * 0.10)
    cv2.rectangle(img, (31, oil_top), (width - 31, height - 21), (30, 140, 200), -1)
    cv2.rectangle(img, (25, 5), (width - 25, 25), (40, 40, 40), -1)
    label_top = height // 3
    cv2.rectangle(img, (32, label_top), (width - 32, 2 * height // 3), (200, 180, 160), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Health check ──────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


# ── Index route ───────────────────────────────────────────────────────────────


def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Olive Oil" in resp.data


# ── /api/inspect ──────────────────────────────────────────────────────────────


def test_inspect_missing_file_returns_400(client):
    resp = client.post("/api/inspect")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_inspect_empty_filename_returns_400(client):
    data = {"image": (io.BytesIO(b""), "")}
    resp = client.post("/api/inspect", content_type="multipart/form-data", data=data)
    assert resp.status_code == 400


def test_inspect_unsupported_type_returns_400(client):
    data = {"image": (io.BytesIO(b"fake"), "file.txt")}
    resp = client.post("/api/inspect", content_type="multipart/form-data", data=data)
    assert resp.status_code == 400


def test_inspect_valid_image_returns_200(client):
    jpeg_bytes = _make_jpeg_bytes()
    data = {"image": (io.BytesIO(jpeg_bytes), "bottle.jpg")}
    resp = client.post("/api/inspect", content_type="multipart/form-data", data=data)
    assert resp.status_code == 200
    result = resp.get_json()
    assert "overall_valid" in result
    assert "fill" in result
    assert "cap_label" in result
    assert "clarity" in result
    assert "alignment" in result
    assert "annotated_image" in result


def test_inspect_result_has_boolean_overall_valid(client):
    jpeg_bytes = _make_jpeg_bytes()
    data = {"image": (io.BytesIO(jpeg_bytes), "bottle.jpg")}
    resp = client.post("/api/inspect", content_type="multipart/form-data", data=data)
    result = resp.get_json()
    assert isinstance(result["overall_valid"], bool)


def test_inspect_corrupt_bytes_returns_422(client):
    data = {"image": (io.BytesIO(b"not-an-image"), "bottle.jpg")}
    resp = client.post("/api/inspect", content_type="multipart/form-data", data=data)
    assert resp.status_code == 422


# ── run_full_inspection unit test ─────────────────────────────────────────────


def test_run_full_inspection_returns_expected_keys():
    jpeg_bytes = _make_jpeg_bytes()
    result = run_full_inspection(jpeg_bytes)
    for key in ("overall_valid", "fill", "cap_label", "clarity", "alignment", "annotated_image"):
        assert key in result, f"Missing key: {key}"


def test_run_full_inspection_annotated_image_is_string():
    jpeg_bytes = _make_jpeg_bytes()
    result = run_full_inspection(jpeg_bytes)
    assert isinstance(result["annotated_image"], str)
    assert len(result["annotated_image"]) > 0
