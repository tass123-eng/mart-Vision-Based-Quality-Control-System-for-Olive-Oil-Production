"""
Configuration settings for the Olive Oil Quality Control System.
"""

# ── Filling detection thresholds ──────────────────────────────────────────────
FILL_LEVEL_MIN = 0.85   # Minimum acceptable fill ratio (0-1)
FILL_LEVEL_MAX = 1.00   # Maximum acceptable fill ratio (0-1)
FILL_ROI_TOP_RATIO = 0.05    # Top boundary of the bottle body ROI (fraction of height)
FILL_ROI_BOTTOM_RATIO = 0.95  # Bottom boundary of the bottle body ROI

# ── Cap and label verification thresholds ─────────────────────────────────────
CAP_AREA_MIN_RATIO = 0.005    # Min cap area relative to full image
CAP_AREA_MAX_RATIO = 0.10     # Max cap area relative to full image
LABEL_AREA_MIN_RATIO = 0.05   # Min label area relative to full image
LABEL_AREA_MAX_RATIO = 0.40   # Max label area relative to full image
LABEL_ALIGNMENT_TOLERANCE = 0.05  # Max horizontal misalignment (fraction of width)

# ── Clarity / impurity analysis thresholds ────────────────────────────────────
TURBIDITY_THRESHOLD = 30.0    # Max allowed turbidity score (0-100)
IMPURITY_AREA_MAX_RATIO = 0.01  # Max impurity area relative to oil region
CLARITY_SCORE_MIN = 70.0      # Minimum clarity score (0-100)
HUE_MIN = 18                  # Min HSV hue for valid olive oil color
HUE_MAX = 38                  # Max HSV hue for valid olive oil color

# ── Cross-polarization preprocessing ─────────────────────────────────────────
POLARIZATION_FILTER_STRENGTH = 0.6  # 0 = no effect, 1 = full polarization
HIGHLIGHT_SUPPRESSION_PERCENTILE = 97  # Percentile used for highlight suppression

# ── Lighting normalization ─────────────────────────────────────────────────────
TARGET_MEAN_BRIGHTNESS = 128    # Target mean pixel intensity after normalization
TARGET_STD_BRIGHTNESS = 50      # Target standard deviation after normalization
CLAHE_CLIP_LIMIT = 2.0          # CLAHE clip limit for adaptive histogram equalization
CLAHE_TILE_GRID_SIZE = (8, 8)   # CLAHE tile grid size

# ── FIG-Cell optical system ───────────────────────────────────────────────────
BOTTLE_HEIGHT_MIN_RATIO = 0.50   # Min bottle height relative to image height
BOTTLE_WIDTH_MIN_RATIO = 0.05    # Min bottle width relative to image width
ALIGNMENT_TOLERANCE_PX = 20      # Max allowed deviation from image centre (pixels)
TILT_ANGLE_MAX_DEG = 5.0         # Max allowed bottle tilt angle (degrees)

# ── Web application ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 16
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
DEBUG = False
HOST = "0.0.0.0"
PORT = 5000
