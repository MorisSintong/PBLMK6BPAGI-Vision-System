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
BBOX_THICKNESS   = 2
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS  = 1

# ── radar map geometry ────────────────────────────────────────────────────────
RADAR_WIDTH_PX  = 300
RADAR_HEIGHT_PX = 180
RADAR_MAX_DEPTH = 8.0

# ── alert panel ──────────────────────────────────────────────────────────────
MAX_ALERT_ROWS = 10

# ── frame display ─────────────────────────────────────────────────────────────
DISPLAY_FPS = 30

# ── depth visualization range (metres) ───────────────────────────────────────
DEPTH_MIN_M = 0.30
DEPTH_MAX_M = 5.00

# ── action thresholds (metres) ────────────────────────────────────────────────
ACTION_STOP_DISTANCE     = 1.0
ACTION_SLOWDOWN_DISTANCE = 3.0

# ── zone labels ───────────────────────────────────────────────────────────────
ZONE_LEFT   = "LEFT"
ZONE_CENTER = "CENTER"
ZONE_RIGHT  = "RIGHT"

# ── action labels ─────────────────────────────────────────────────────────────
ACTION_STOP       = "STOP"
ACTION_SLOWDOWN   = "PERLAMBAT"
ACTION_TURN_RIGHT = "BELOK KANAN"
ACTION_TURN_LEFT  = "BELOK KIRI"
ACTION_GO         = "LANJUT"