# GUI/src/radar_view.py

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QConicalGradient, QFont, QPainter, QPen
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(RADAR_WIDTH_PX, RADAR_HEIGHT_PX)

        self._obstacles = []
        self._sweep = 0.0
        self._sweep_dir = 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def update_obstacles(self, obstacles: list):
        self._obstacles = obstacles
        self.update()

    def clear_obstacles(self):
        self._obstacles = []
        self.update()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #
    def _tick(self):
        self._sweep += 3 * self._sweep_dir
        if self._sweep >= 180:
            self._sweep = 180
            self._sweep_dir = -1
        elif self._sweep <= 0:
            self._sweep = 0
            self._sweep_dir = 1
        self.update()

    # ------------------------------------------------------------------ #
    #  Paint                                                               #
    # ------------------------------------------------------------------ #
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx = w / 2
        cy = h - 15
        r = min(w / 2 - 40, h - 30)

        # ── 1. Background ─────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_BORDER), 1))
        p.setBrush(QBrush(QColor(RADAR_BG)))
        p.drawRoundedRect(0, 0, w, h, 12, 12)

        # ── 2. Title ──────────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_SWEEP)))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0, 6, w, 16), Qt.AlignmentFlag.AlignCenter, "RADAR VIEW")

        # ── 3. Setengah Cincin Jarak ──────────────────────────────────
        rings = 4
        for i in range(1, rings + 1):
            ri = r * i / rings
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - ri, cy - ri, ri * 2, ri * 2), 0 * 16, 180 * 16)

            # Label jarak
            dist_label = f"{int(RADAR_MAX_DEPTH * i / rings)}m"
            p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(
                QRectF(cx + 3, cy - ri - 10, 20, 12),
                Qt.AlignmentFlag.AlignLeft,
                dist_label,
            )

        # ── 4. Garis Dasar (Horizontal) ───────────────────────────────
        p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
        p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))

        # ── 5. Garis Sudut ────────────────────────────────────────────
        for deg in [45, 90, 135]:
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
            p.drawLine(
                QPointF(cx, cy),
                QPointF(
                    cx + r * math.cos(math.radians(180 - deg)),
                    cy - r * math.sin(math.radians(180 - deg)),
                ),
            )

        # ── 6. Garis Zona (60 & 120 derajat) ─────────────────────────
        for deg in [60, 120]:
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8, Qt.PenStyle.DashLine))
            p.drawLine(
                QPointF(cx, cy),
                QPointF(
                    cx - r * math.cos(math.radians(deg)),
                    cy - r * math.sin(math.radians(deg)),
                ),
            )

        # ── 7. Label Zona ─────────────────────────────────────────────
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))

        # CENTER — di atas tengah
        p.drawText(
            QRectF(cx - 25, cy - r - 25, 50, 14), Qt.AlignmentFlag.AlignCenter, "CENTER"
        )

        # LEFT — di luar kiri
        p.drawText(
            QRectF(cx - r - 35, cy - 8, 35, 14), Qt.AlignmentFlag.AlignCenter, "LEFT"
        )

        # RIGHT — di luar kanan
        p.drawText(
            QRectF(cx + r + 2, cy - 8, 35, 14), Qt.AlignmentFlag.AlignCenter, "RIGHT"
        )

        # ── 8. Sweep Line ─────────────────────────────────────────────
        sx = cx + r * math.cos(math.radians(180 - self._sweep))
        sy = cy - r * math.sin(math.radians(180 - self._sweep))
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.5))
        p.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # ── 9. Cincin Luar (setengah) ─────────────────────────────────
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 0 * 16, 180 * 16)

        # ── 10. Titik Pusat ───────────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(RADAR_SWEEP)))
        p.drawEllipse(QPointF(cx, cy), 3, 3)

        # ── 11. Obstacle Blips ────────────────────────────────────────
        ZONE_TO_ANGLE = {"left": 45, "center": 90, "right": 135}
        for obs in self._obstacles:
            zone = obs.get("zone", "center")
            angle_deg = obs.get("angle_deg") or ZONE_TO_ANGLE.get(zone.upper(), 90)
            distance_m = obs.get("distance_m", 0)

            dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
            bx = cx + dist_frac * r * math.cos(math.radians(180 - angle_deg))
            by = cy - dist_frac * r * math.sin(math.radians(180 - angle_deg))

            if zone == "CENTER":
                color = QColor(RADAR_BLIP_CENTER)
            elif zone in ("LEFT", "RIGHT"):
                color = QColor(RADAR_BLIP_SIDE)
            else:
                color = QColor(RADAR_BLIP_SAFE)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(bx, by), 5, 5)

        # ── 12. Bottom label ──────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(
            QRectF(0, h - 14, w, 12),
            Qt.AlignmentFlag.AlignCenter,
            "Intel RealSense D455",
        )

        p.end()
