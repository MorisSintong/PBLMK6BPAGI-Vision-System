"""
GUI/src/radar_view.py — 90° FOV top-down radar widget.

Role 6 (Adel) — GUI.

Renders obstacle positions as blips on a 90° forward-facing radar arc, with a
sweeping line for visual feedback and a steering arrow derived from
NavigationStage output.

Performance:
  - Pre-parsed QColor/QPen/QBrush objects (no per-paint string parsing)
  - Cached static background pixmap (rings, labels, FOV lines pre-rendered)
  - Only the sweep + blips are re-drawn per frame via QTimer
  - 20 FPS animation cap (independent of pipeline FPS)
"""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget
from GUI.inc.styles import (
    RADAR_BG,
    RADAR_BLIP_CENTER,
    RADAR_BLIP_SAFE,
    RADAR_BLIP_SIDE,
    RADAR_BORDER,
    RADAR_LABEL_MUTED,
    RADAR_SWEEP,
)
from GUI.inc.ui_config import RADAR_HEIGHT_PX, RADAR_MAX_DEPTH, RADAR_WIDTH_PX


class RadarView(QWidget):
    # Pre-parsed color/pen/brush objects (avoid per-paint string parsing)
    _C_BLIP_CENTER = QColor(RADAR_BLIP_CENTER)
    _C_BLIP_SAFE = QColor(RADAR_BLIP_SAFE)
    _C_BLIP_SIDE = QColor(RADAR_BLIP_SIDE)
    _C_SWEEP = QColor(RADAR_SWEEP)
    _C_BORDER = QColor(RADAR_BORDER)
    _C_BG = QColor(RADAR_BG)
    _C_LABEL = QColor(RADAR_LABEL_MUTED)
    _C_ARROW_STOP = QColor("#f38ba8")
    _C_ARROW_AVOID = QColor("#fab387")
    _C_ARROW_CLEAR = QColor("#50f050")

    _BRUSH_BLIP_CENTER = QBrush(_C_BLIP_CENTER)
    _BRUSH_BLIP_SAFE = QBrush(_C_BLIP_SAFE)
    _BRUSH_BLIP_SIDE = QBrush(_C_BLIP_SIDE)
    _BRUSH_BG = QBrush(_C_BG)
    _BRUSH_SWEEP_DOT = QBrush(_C_SWEEP)

    _PEN_SWEEP = QPen(_C_SWEEP, 1.5)
    _PEN_BORDER = QPen(_C_BORDER, 1)
    _PEN_ARROW_STOP = QPen(_C_ARROW_STOP, 3)
    _PEN_ARROW_AVOID = QPen(_C_ARROW_AVOID, 3)
    _PEN_ARROW_CLEAR = QPen(_C_ARROW_CLEAR, 3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(RADAR_WIDTH_PX, RADAR_HEIGHT_PX)

        self._obstacles = []
        self._steering_angle = 0.0
        self._nav_status = "IDLE"
        self._sweep = 45.0
        self._sweep_dir = 1
        self._animating = False

        # Cache the static radar background (rings, labels, FOV lines) once
        self._static_bg: QPixmap = None
        self._render_static_bg()

        # Slower sweep (50ms = 20fps) — radar doesn't need 25fps
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # Don't auto-start; only animate when a source is active
        # (see start_animation / stop_animation)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def start_animation(self):
        """Start the sweep animation (call when camera/playback begins)."""
        if not self._animating:
            self._animating = True
            self._timer.start(50)

    def stop_animation(self):
        """Stop the sweep animation (call when camera/playback ends)."""
        if self._animating:
            self._animating = False
            self._timer.stop()
            self.update()  # One final repaint to clear sweep

    def update_obstacles(self, obstacles: list):
        # Skip if data hasn't changed (avoid redundant repaint)
        if obstacles == self._obstacles:
            return
        self._obstacles = obstacles
        self.update()

    def update_navigation(self, nav_data: dict):
        """Update steering arrow from NavigationStage output."""
        if not nav_data:
            return
        new_steer = nav_data.get("steering_angle_deg", 0.0)
        new_status = nav_data.get("status", "IDLE")
        # Skip if nothing changed
        if (new_steer == self._steering_angle
                and new_status == self._nav_status):
            return
        self._steering_angle = new_steer
        self._nav_status = new_status
        self.update()

    def clear_obstacles(self):
        self._obstacles = []
        self._steering_angle = 0.0
        self._nav_status = "IDLE"
        self.update()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #
    def _tick(self):
        self._sweep += 3 * self._sweep_dir
        if self._sweep >= 135:
            self._sweep = 135
            self._sweep_dir = -1
        elif self._sweep <= 45:
            self._sweep = 45
            self._sweep_dir = 1
        self.update()

    def _render_static_bg(self):
        """Pre-render all static radar elements into a cached pixmap."""
        self._static_bg = QPixmap(self.width(), self.height())
        self._static_bg.fill(Qt.GlobalColor.transparent)

        p = QPainter(self._static_bg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx = w / 2
        cy = h - 15
        r = min(w / 2 - 40, h - 30)

        # Background
        p.setPen(QPen(QColor(RADAR_BORDER), 1))
        p.setBrush(QBrush(QColor(RADAR_BG)))
        p.drawRoundedRect(0, 0, w, h, 12, 12)

        # Title
        p.setPen(QPen(QColor(RADAR_SWEEP)))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0, 6, w, 16), Qt.AlignmentFlag.AlignCenter, "RADAR VIEW")

        # Distance rings
        rings = 4
        for i in range(1, rings + 1):
            ri = r * i / rings
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - ri, cy - ri, ri * 2, ri * 2), 45 * 16, 90 * 16)

            dist_label = f"{int(RADAR_MAX_DEPTH * i / rings)}m"
            p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
            p.setFont(QFont("Segoe UI", 7))
            lx = cx + ri * math.cos(math.radians(135))
            ly = cy - ri * math.sin(math.radians(135))
            p.drawText(QRectF(lx - 25, ly - 5, 20, 12), Qt.AlignmentFlag.AlignRight, dist_label)

        # FOV boundary lines
        p.setPen(QPen(QColor(RADAR_BORDER), 1.5))
        for deg in [45, 135]:
            p.drawLine(
                QPointF(cx, cy),
                QPointF(cx + r * math.cos(math.radians(deg)), cy - r * math.sin(math.radians(deg))),
            )

        # Center & sub-zone lines
        for deg in [75, 90, 105]:
            pen = QPen(QColor(RADAR_BORDER), 0.8)
            if deg != 90:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(
                QPointF(cx, cy),
                QPointF(cx + r * math.cos(math.radians(deg)), cy - r * math.sin(math.radians(deg))),
            )

        # Zone labels
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
        p.drawText(QRectF(cx - 25, cy - r - 25, 50, 14), Qt.AlignmentFlag.AlignCenter, "CENTER")
        p.drawText(QRectF(cx - r - 15, cy - r + 20, 35, 14), Qt.AlignmentFlag.AlignCenter, "LEFT")
        p.drawText(QRectF(cx + r - 20, cy - r + 20, 35, 14), Qt.AlignmentFlag.AlignCenter, "RIGHT")

        # Outer ring
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 45 * 16, 90 * 16)

        # Center dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(RADAR_SWEEP)))
        p.drawEllipse(QPointF(cx, cy), 3, 3)

        # Bottom label
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(QRectF(0, h - 14, w, 12), Qt.AlignmentFlag.AlignCenter, "Intel RealSense D455")

        p.end()

    # ------------------------------------------------------------------ #
    #  Paint — only draw dynamic parts (sweep + blips) on top of cache    #
    # ------------------------------------------------------------------ #
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx = w / 2
        cy = h - 15
        r = min(w / 2 - 40, h - 30)

        # Blit cached static background
        p.drawPixmap(0, 0, self._static_bg)

        # Sweep line (dynamic) — use cached pen
        sx = cx + r * math.cos(math.radians(self._sweep))
        sy = cy - r * math.sin(math.radians(self._sweep))
        p.setPen(self._PEN_SWEEP)
        p.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # Obstacle blips (dynamic) — use cached brushes
        for obs in self._obstacles:
            distance_m = obs.get("distance_m", 0)
            if distance_m <= 0:
                continue

            if "angle_deg" in obs:
                angle_deg = obs["angle_deg"]
            elif "bbox" in obs:
                x1, y1, x2, y2 = obs["bbox"]
                bbox_cx = (x1 + x2) / 2
                angle_deg = 135 - (bbox_cx / 640.0) * 90
            else:
                zone = obs.get("zone", "center").upper()
                angle_deg = {"LEFT": 120, "CENTER": 90, "RIGHT": 60}.get(zone, 90)

            angle_deg = max(45, min(135, angle_deg))

            dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
            bx = cx + dist_frac * r * math.cos(math.radians(angle_deg))
            by = cy - dist_frac * r * math.sin(math.radians(angle_deg))

            zone_str = obs.get("zone", "center").upper()
            if zone_str == "CENTER":
                brush = self._BRUSH_BLIP_CENTER
            elif zone_str in ("LEFT", "RIGHT"):
                brush = self._BRUSH_BLIP_SIDE
            else:
                brush = self._BRUSH_BLIP_SAFE

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(brush)
            p.drawEllipse(QPointF(bx, by), 5, 5)

        # Steering recommendation arrow (dynamic) — use cached pens
        if self._nav_status not in ("IDLE", ""):
            # Map steering angle (-45..+45) to radar angle (135..45)
            radar_angle = 90 - self._steering_angle
            arrow_r = r * 0.7
            ax = cx + arrow_r * math.cos(math.radians(radar_angle))
            ay = cy - arrow_r * math.sin(math.radians(radar_angle))

            if self._nav_status in ("STOPPED", "BLOCKED"):
                pen = self._PEN_ARROW_STOP
                brush_color = self._C_ARROW_STOP
            elif self._nav_status == "AVOIDING":
                pen = self._PEN_ARROW_AVOID
                brush_color = self._C_ARROW_AVOID
            else:
                pen = self._PEN_ARROW_CLEAR
                brush_color = self._C_ARROW_CLEAR

            p.setPen(pen)
            p.drawLine(QPointF(cx, cy), QPointF(ax, ay))
            # Arrowhead
            p.setBrush(QBrush(brush_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(ax, ay), 4, 4)

        p.end()
