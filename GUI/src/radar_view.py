# GUI/src/radar_view.py

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QConicalGradient
)

from ui_config import RADAR_MAX_DEPTH, RADAR_WIDTH_PX, RADAR_HEIGHT_PX
from styles import (
    RADAR_BG, RADAR_BORDER, RADAR_SWEEP,
    RADAR_LABEL_MUTED, RADAR_BLIP_CENTER,
    RADAR_BLIP_SIDE, RADAR_BLIP_SAFE
)


class RadarView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(RADAR_WIDTH_PX, RADAR_HEIGHT_PX)

        # Data obstacle — diisi dari luar saat pipeline sudah siap
        self._obstacles = []

        # Sweep angle
        self._sweep = 0.0

        # Timer sweep
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def update_obstacles(self, obstacles: list):
        """
        Menerima data obstacle dari pipeline.

        Parameters:
            obstacles: list of dict {
                'angle_deg'  : float,
                'distance_m' : float,
                'zone'       : str — 'LEFT', 'CENTER', 'RIGHT'
            }
        """
        self._obstacles = obstacles
        self.update()

    def clear_obstacles(self):
        """Reset radar — dipanggil saat kamera stop."""
        self._obstacles = []
        self.update()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #
    def _tick(self):
        self._sweep = (self._sweep + 3) % 360
        self.update()

    # ------------------------------------------------------------------ #
    #  Paint                                                               #
    # ------------------------------------------------------------------ #
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h   = self.width(), self.height()
        cx, cy = w / 2, (h - 30) / 2 + 10
        r      = min(cx, cy) - 20

        # ── 1. Background ─────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_BORDER), 1))
        p.setBrush(QBrush(QColor(RADAR_BG)))
        p.drawRoundedRect(0, 0, w, h, 12, 12)

        # ── 2. Title ──────────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_SWEEP)))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(
            QRectF(0, 8, w, 20),
            Qt.AlignmentFlag.AlignCenter,
            "RADAR VIEW"
        )

        # ── 3. Cincin Jarak ───────────────────────────────────────────
        rings = 4
        for i in range(1, rings + 1):
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            ri = r * i / rings
            p.drawEllipse(QPointF(cx, cy), ri, ri)

            dist_label = f"{int(RADAR_MAX_DEPTH * i / rings)}m"
            p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(
                QRectF(cx + 3, cy - ri - 10, 20, 12),
                Qt.AlignmentFlag.AlignLeft,
                dist_label
            )

        # ── 4. Garis Silang ───────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
        for deg in range(0, 180, 45):
            rad = math.radians(deg)
            p.drawLine(
                QPointF(cx - r * math.cos(rad), cy - r * math.sin(rad)),
                QPointF(cx + r * math.cos(rad), cy + r * math.sin(rad))
            )

        # ── 5. Zona LEFT / CENTER / RIGHT ─────────────────────────────
        for zone_angle in [-30, 30]:
            rad = math.radians(90 - zone_angle)
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8,
                          Qt.PenStyle.DashLine))
            p.drawLine(
                QPointF(cx, cy),
                QPointF(cx + r * math.cos(rad), cy - r * math.sin(rad))
            )

        zone_y = cy + r + 14
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
        p.drawText(QRectF(cx - r - 10, zone_y, r - 5, 14),
                   Qt.AlignmentFlag.AlignCenter, "LEFT")
        p.drawText(QRectF(cx - 20, zone_y, 40, 14),
                   Qt.AlignmentFlag.AlignCenter, "CENTER")
        p.drawText(QRectF(cx + 10, zone_y, r - 5, 14),
                   Qt.AlignmentFlag.AlignCenter, "RIGHT")

        # ── 6. Sweep Cone ─────────────────────────────────────────────
        sg = QConicalGradient(QPointF(cx, cy), 90 - self._sweep)
        sg.setColorAt(0.00, QColor(137, 180, 250, 120))
        sg.setColorAt(0.12, QColor(137, 180, 250, 15))
        sg.setColorAt(0.13, Qt.GlobalColor.transparent)
        sg.setColorAt(1.00, Qt.GlobalColor.transparent)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(sg))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Garis sweep
        sr = math.radians(90 - self._sweep)
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.5))
        p.drawLine(
            QPointF(cx, cy),
            QPointF(cx + r * math.cos(sr), cy - r * math.sin(sr))
        )

        # ── 7. Cincin Luar ────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # ── 8. Titik Pusat ────────────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(RADAR_SWEEP)))
        p.drawEllipse(QPointF(cx, cy), 3, 3)

        # ── 9. Obstacle Blips (dari data nyata) ───────────────────────
        for obs in self._obstacles:
            angle_deg  = obs.get("angle_deg", 0)
            distance_m = obs.get("distance_m", 0)
            zone       = obs.get("zone", "CENTER")

            dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
            brad = math.radians(90 - angle_deg)
            bx = cx + dist_frac * r * math.cos(brad)
            by = cy - dist_frac * r * math.sin(brad)

            if zone == "CENTER":
                color = QColor(RADAR_BLIP_CENTER)
            elif zone in ("LEFT", "RIGHT"):
                color = QColor(RADAR_BLIP_SIDE)
            else:
                color = QColor(RADAR_BLIP_SAFE)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(bx, by), 5, 5)

        # ── 10. Bottom label ──────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(
            QRectF(0, h - 16, w, 14),
            Qt.AlignmentFlag.AlignCenter,
            "Intel RealSense D455"
        )

        p.end()