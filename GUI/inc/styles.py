GLOBAL_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QLabel {
    color: #cdd6f4;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 8px;
    padding: 8px;
    background-color: #1e1e2e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #89b4fa;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: #45475a;
}

QPushButton:disabled {
    color: #585b70;
}

QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px;
}
"""

# ── Status bar (SAFE / WARNING / DANGER) ──────────────────────────────────────
STATUS_SAFE    = "background-color: #1D9E75; color: white; border-radius: 6px; padding: 3px; font-weight: bold;"
STATUS_WARNING = "background-color: #EF9F27; color: #1e1e2e; border-radius: 6px; padding: 3px; font-weight: bold;"
STATUS_DANGER  = "background-color: #E24B4A; color: #1e1e2e; border-radius: 6px; padding: 3px; font-weight: bold;"

# ── Camera status ─────────────────────────────────────────────────────────────
STATUS_ACTIVE   = "color: #a6e3a1;"
STATUS_INACTIVE = "color: #f38ba8;"

# ── Info label ────────────────────────────────────────────────────────────────
STATUS_ERROR   = "color: #f38ba8;"
STATUS_SUCCESS = "color: #a6e3a1;"

# ── View mode buttons ─────────────────────────────────────────────────────────
VIEW_ACTIVE  = "background-color: #89b4fa; color: #1e1e2e; font-weight: bold;"
VIEW_DEFAULT = "background-color: #313244; font-weight: normal; color: #cdd6f4;"

# ── Info box (Alert Panel) ────────────────────────────────────────────────────
INFOBOX_DEFAULT = "background-color: #313244; border: 2px solid #45475a; border-radius: 15px;"
INFOBOX_DANGER  = "background-color: #E24B4A; border-radius: 15px;"
INFOBOX_WARNING = "background-color: #EF9F27; border-radius: 15px;"
INFOBOX_SAFE    = "background-color: #313244; border-radius: 15px;"

# ── Text colors ───────────────────────────────────────────────────────────────
TEXT_DARK  = "color: #1e1e2e; background: transparent;"
TEXT_LIGHT = "color: #cdd6f4; background: transparent;"
TEXT_WHITE = "color: white; background: transparent;"
TEXT_BLUE  = "color: #89b4fa; background: transparent;"
TEXT_GREEN = "color: #a6e3a1; background: transparent;"
# ── Radar View ────────────────────────────────────────────────────────────────
RADAR_BG          = "#1e1e2e"
RADAR_BORDER      = "#45475a"
RADAR_SWEEP       = "#89b4fa"
RADAR_LABEL_MUTED = "#585b70"
RADAR_BLIP_CENTER = "#E24B4A"
RADAR_BLIP_SIDE   = "#EF9F27"
RADAR_BLIP_SAFE   = "#1D9E75"