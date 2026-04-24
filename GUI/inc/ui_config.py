"""
gui/inc/ui_config.py — UI-level constants consumed by main.py and widgets.
"""

# ── application identity ──────────────────────────────────────────────────────
APP_NAME   = "Depth Obstacle Detector"

# ── window geometry ───────────────────────────────────────────────────────────
WINDOW_MIN_W = 1100
WINDOW_MIN_H = 680

# ── obstacle distance thresholds (metres) ─────────────────────────────────────
THRESHOLD_DANGER  = 1.0   # red   — obstacle closer than this
THRESHOLD_WARNING = 3.0   # amber — obstacle between danger and this

# ── zone colour hex codes (used by both Qt widgets and OpenCV overlays) ───────
COLOR_DANGER  = "#E24B4A"
COLOR_WARNING = "#EF9F27"
COLOR_SAFE    = "#1D9E75"

# ── overlay / bounding-box drawing ───────────────────────────────────────────
BBOX_THICKNESS   = 2       # pixels
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS  = 1

# ── radar map geometry ────────────────────────────────────────────────────────
RADAR_WIDTH_PX  = 250
RADAR_HEIGHT_PX = 300
RADAR_MAX_DEPTH = 8.0     # metres — farthest ring drawn on the radar

# ── alert panel ──────────────────────────────────────────────────────────────
MAX_ALERT_ROWS = 10       # maximum rows shown in the alert list

# ── frame display ─────────────────────────────────────────────────────────────
DISPLAY_FPS = 30          # QTimer interval = 1000 // DISPLAY_FPS ms