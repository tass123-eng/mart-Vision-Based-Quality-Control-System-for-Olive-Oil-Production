"""Flask web application for the Olive Oil Quality Control System."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request

from app.fig_cell.optical_system import check_bottle_alignment, get_alignment_guidance
from app.models.cap_label_verification import verify_cap_and_label
from app.models.clarity_analysis import analyze_clarity
from app.models.filling_detection import analyze_fill_level
from app.preprocessing.lighting import apply_lighting_normalization
from app.preprocessing.polarization import apply_cross_polarization
from app.utils.image_utils import (
    decode_image_from_bytes,
    draw_result_overlay,
    encode_image_to_base64,
    resize_for_display,
)
from config.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def run_full_inspection(image_bytes: bytes) -> dict:
    """Run the complete quality-control pipeline on raw image bytes.

    Pipeline steps
    --------------
    1. Decode image.
    2. FIG-Cell alignment check.
    3. Cross-polarization pre-processing.
    4. Lighting normalization.
    5. Model 1 – fill-level detection.
    6. Model 2 – cap & label verification.
    7. Model 3 – clarity & impurity analysis.
    8. Compose overall verdict.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the uploaded image file.

    Returns
    -------
    dict
        Full inspection result including per-model sub-dicts and an
        ``overall_valid`` flag.
    """
    image = decode_image_from_bytes(image_bytes)

    alignment = check_bottle_alignment(image)
    guidance = get_alignment_guidance(alignment)

    preprocessed = apply_cross_polarization(image)
    preprocessed = apply_lighting_normalization(preprocessed)

    fill_result = analyze_fill_level(preprocessed)
    cap_label_result = verify_cap_and_label(preprocessed)
    clarity_result = analyze_clarity(preprocessed)

    overall_valid = (
        fill_result["is_valid"]
        and cap_label_result["is_valid"]
        and clarity_result["is_valid"]
    )

    results = {
        "overall_valid": overall_valid,
        "alignment": {**alignment, "guidance": guidance},
        "fill": fill_result,
        "cap_label": cap_label_result,
        "clarity": clarity_result,
    }

    annotated = draw_result_overlay(resize_for_display(image), results)
    results["annotated_image"] = encode_image_to_base64(annotated)

    return results


@app.route("/")
def index():
    """Serve the main inspection UI."""
    return render_template("index.html")


@app.route("/api/inspect", methods=["POST"])
def inspect():
    """REST endpoint: POST an image, get quality-control results.

    Accepts ``multipart/form-data`` with a field named ``image``.

    Returns
    -------
    JSON
        Full inspection results (see :func:`run_full_inspection`).
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    try:
        results = run_full_inspection(image_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # pragma: no cover – unexpected errors
        return jsonify({"error": f"Internal error: {exc}"}), 500

    return jsonify(results), 200


@app.route("/api/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    from config.config import DEBUG, HOST, PORT

    app.run(host=HOST, port=PORT, debug=DEBUG)
