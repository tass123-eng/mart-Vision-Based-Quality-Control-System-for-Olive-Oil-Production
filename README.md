# Vision-Based Quality Control System for Olive Oil Production

> **Green Tech AI Initiative** – An intelligent, computer-vision-driven solution
> that automates and optimises quality control for bottled olive oil, reducing
> manual inspection errors and improving production throughput.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Key Components](#key-components)
   - [Cross-Polarization Pre-processing](#1-cross-polarization-pre-processing)
   - [FIG-Cell Optical Gravity System](#2-fig-cell-optical-gravity-system)
   - [Model 1 – Filling Detection](#3-model-1--filling-detection)
   - [Model 2 – Cap & Label Verification](#4-model-2--cap--label-verification)
   - [Model 3 – Clarity & Impurity Analysis](#5-model-3--clarity--impurity-analysis)
   - [Controlled Lighting Normalisation](#6-controlled-lighting-normalisation)
   - [Web Application](#7-web-application)
5. [Getting Started](#getting-started)
6. [Running Tests](#running-tests)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)

---

## Overview

Manual quality control of olive oil bottles is slow, expensive, and prone to
human error. This project implements an end-to-end **AI + computer-vision**
pipeline that inspects every bottle on the production line in real time,
checking:

| Check | Method |
|---|---|
| Fill level | Edge-based liquid surface detection |
| Cap & label | Contour-based presence and alignment check |
| Oil clarity | Turbidity and impurity particle analysis |
| Bottle alignment | FIG-Cell geometric alignment verification |

The pipeline is wrapped in a Flask web application with a responsive UI,
allowing operators to inspect bottles directly from a browser by uploading
images captured by the production-line camera.

---

## Architecture

```
Camera frame
    │
    ▼
┌──────────────────────────────┐
│  FIG-Cell alignment check    │  ← checks position, tilt, height
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Cross-polarization filter   │  ← suppresses glass glare
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Lighting normalisation      │  ← global normalise + CLAHE
└──────────────┬───────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
  Model 1  Model 2  Model 3
  Fill     Cap/     Clarity
  Level    Label    Analysis
       │       │       │
       └───────┴───────┘
               │
               ▼
      Overall PASS / FAIL
       + annotated image
```

---

## Project Structure

```
mart-Vision-Based-Quality-Control-System-for-Olive-Oil-Production/
├── app/
│   ├── fig_cell/
│   │   └── optical_system.py      # FIG-Cell alignment & guidance
│   ├── models/
│   │   ├── filling_detection.py   # Model 1 – fill level
│   │   ├── cap_label_verification.py # Model 2 – cap & label
│   │   └── clarity_analysis.py    # Model 3 – oil clarity
│   ├── preprocessing/
│   │   ├── polarization.py        # Cross-polarization simulation
│   │   └── lighting.py            # Adaptive lighting normalisation
│   ├── utils/
│   │   └── image_utils.py         # Shared image helpers
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   ├── templates/
│   │   └── index.html
│   └── main.py                    # Flask application entry point
├── config/
│   └── config.py                  # All tuneable thresholds & settings
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_fig_cell.py
│   └── test_app.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Key Components

### 1. Cross-Polarization Pre-processing

**File:** `app/preprocessing/polarization.py`

Simulates the effect of crossed polarising filters to suppress specular
reflections (glare) on glass bottles. Real hardware uses two polarising
filters at 90° to each other; the software equivalent:

1. Converts to HSV colour space.
2. Identifies highlight pixels above the 97th-percentile brightness threshold.
3. Reduces their value channel proportionally to their excess brightness,
   controlled by `POLARIZATION_FILTER_STRENGTH` (default 0.6).
4. Applies a sharpening kernel to enhance edge visibility of surface defects.

### 2. FIG-Cell Optical Gravity System

**File:** `app/fig_cell/optical_system.py`

The **FIG-Cell** (Floating Image Gravity Cell) is a novel optomechanical
concept where the bottle is centred on a gravimetric cradle for repeatable,
precisely positioned image acquisition. The software module:

* Detects the bottle contour in the frame.
* Measures horizontal offset from the image centre (alignment tolerance:
  ±20 px by default).
* Measures the tilt angle from vertical (max 5° by default).
* Returns corrective guidance (`OK`, `REPOSITION`, or `REJECT`).

### 3. Model 1 – Filling Detection

**File:** `app/models/filling_detection.py`

Detects the liquid surface inside the bottle using horizontal Sobel edge
detection on the upper half of the bottle body. Computes the **fill ratio**
(liquid height / bottle body height) and validates it against configurable
`FILL_LEVEL_MIN` / `FILL_LEVEL_MAX` thresholds (default 85 %–100 %).

### 4. Model 2 – Cap & Label Verification

**File:** `app/models/cap_label_verification.py`

* **Cap detection** – Finds a compact object in the top 20 % of the image
  using Otsu thresholding + contour analysis; validates its centring.
* **Label detection** – Finds a large rectangular region in the central 50 %
  of the image using Canny + morphological closing; validates its horizontal
  alignment (tolerance: 5 % of image width by default).

### 5. Model 3 – Clarity & Impurity Analysis

**File:** `app/models/clarity_analysis.py`

Evaluates oil optical quality in three dimensions:

| Metric | Method |
|---|---|
| **Turbidity** | Laplacian variance of HSV saturation channel |
| **Impurity ratio** | Dark-particle detection via value-channel thresholding |
| **Colour validity** | Median hue check against expected golden-yellow range |

The three metrics are combined into a single **clarity score** (0–100).

### 6. Controlled Lighting Normalisation

**File:** `app/preprocessing/lighting.py`

Two-stage pipeline to compensate for LED panel variations:

1. **Global normalisation** – affine rescaling to reach a target mean/standard
   deviation across the image.
2. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) in LAB colour
   space – enhances local contrast without distorting colour.

### 7. Web Application

**File:** `app/main.py`

A Flask application exposing:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Responsive inspection UI |
| `/api/inspect` | POST | Upload image, receive full JSON result |
| `/api/health` | GET | Service health check |

---

## Getting Started

### Prerequisites

* Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/tass123-eng/mart-Vision-Based-Quality-Control-System-for-Olive-Oil-Production.git
cd mart-Vision-Based-Quality-Control-System-for-Olive-Oil-Production

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python app/main.py
```

Then open your browser at **http://localhost:5000**.

Upload a bottle image (PNG / JPG / BMP / TIFF, max 16 MB) and the system
will run the full inspection pipeline, returning a visual result with a
PASS / FAIL verdict.

---

## Running Tests

```bash
pytest
```

All 61 tests should pass. To run with coverage:

```bash
pytest --cov=app --cov=config --cov-report=term-missing
```

---

## Configuration

All tuneable parameters are centralised in `config/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `FILL_LEVEL_MIN` | `0.85` | Minimum acceptable fill ratio |
| `FILL_LEVEL_MAX` | `1.00` | Maximum acceptable fill ratio |
| `CAP_AREA_MIN_RATIO` | `0.005` | Min cap area (fraction of image) |
| `LABEL_AREA_MIN_RATIO` | `0.05` | Min label area (fraction of image) |
| `LABEL_ALIGNMENT_TOLERANCE` | `0.05` | Max horizontal offset (fraction of width) |
| `TURBIDITY_THRESHOLD` | `30.0` | Max allowed turbidity score (0–100) |
| `IMPURITY_AREA_MAX_RATIO` | `0.01` | Max impurity area in oil region |
| `CLARITY_SCORE_MIN` | `70.0` | Minimum clarity score (0–100) |
| `HUE_MIN` / `HUE_MAX` | `18` / `38` | Expected HSV hue range for olive oil |
| `POLARIZATION_FILTER_STRENGTH` | `0.6` | Glare suppression strength (0–1) |
| `ALIGNMENT_TOLERANCE_PX` | `20` | Max bottle offset from centre (px) |
| `TILT_ANGLE_MAX_DEG` | `5.0` | Max allowed bottle tilt (degrees) |

---

## API Reference

### `POST /api/inspect`

**Content-Type:** `multipart/form-data`  
**Field:** `image` – image file (PNG / JPG / BMP / TIFF)

**Response (200):**
```json
{
  "overall_valid": true,
  "alignment": {
    "bottle_detected": true,
    "is_aligned": true,
    "centre_offset_px": -2.5,
    "tilt_angle_deg": 0.8,
    "bottle_height_ratio": 0.82,
    "message": "FIG-Cell alignment OK.",
    "guidance": { "action": "OK", "direction": null, "guidance": "..." }
  },
  "fill": {
    "is_valid": true,
    "fill_ratio": 0.92,
    "fill_level_px": 45,
    "message": "Fill level OK: 92.00%."
  },
  "cap_label": {
    "cap_present": true,
    "cap_valid": true,
    "label_present": true,
    "label_valid": true,
    "is_valid": true,
    "message": "Cap and label verification passed."
  },
  "clarity": {
    "is_valid": true,
    "clarity_score": 88.4,
    "turbidity_score": 12.1,
    "impurity_ratio": 0.0023,
    "colour_valid": true,
    "message": "Oil clarity OK (score 88.4/100)."
  },
  "annotated_image": "<base64-encoded JPEG>"
}
```

**Error responses:** `400` (missing/invalid file), `422` (undecipherable image).
