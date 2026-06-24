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
        self._sweep = 45.0  # Mulai dari ujung kanan FOV (45 derajat)
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
        if self._sweep >= 135:
            self._sweep = 135
            self._sweep_dir = -1
        elif self._sweep <= 45:
            self._sweep = 45
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

        # ── 3. Cincin Jarak (90 derajat FOV) ──────────────────────────
        rings = 4
        for i in range(1, rings + 1):
            ri = r * i / rings
            p.setPen(QPen(QColor(RADAR_BORDER), 0.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            # PyQt drawArc: startAngle and spanAngle in 1/16ths of a degree.
            # 45 degrees start, 90 degrees span (draws from 45 to 135)
            p.drawArc(QRectF(cx - ri, cy - ri, ri * 2, ri * 2), 45 * 16, 90 * 16)

            # Label jarak (ditaruh di sisi kiri wedge)
            dist_label = f"{int(RADAR_MAX_DEPTH * i / rings)}m"
            p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))
            p.setFont(QFont("Segoe UI", 7))
            
            lx = cx + ri * math.cos(math.radians(135))
            ly = cy - ri * math.sin(math.radians(135))
            p.drawText(
                QRectF(lx - 25, ly - 5, 20, 12),
                Qt.AlignmentFlag.AlignRight,
                dist_label,
            )

        # ── 4. Garis Batas FOV (Kiri & Kanan) ─────────────────────────
        p.setPen(QPen(QColor(RADAR_BORDER), 1.5)) # Garis luar lebih tebal
        for deg in [45, 135]:
            p.drawLine(
                QPointF(cx, cy),
                QPointF(
                    cx + r * math.cos(math.radians(deg)),
                    cy - r * math.sin(math.radians(deg)),
                ),
            )

        # ── 5. Garis Tengah & Sub-zona ────────────────────────────────
        for deg in [75, 90, 105]:
            pen = QPen(QColor(RADAR_BORDER), 0.8)
            if deg != 90:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(
                QPointF(cx, cy),
                QPointF(
                    cx + r * math.cos(math.radians(deg)),
                    cy - r * math.sin(math.radians(deg)),
                ),
            )

        # ── 6. Label Zona ─────────────────────────────────────────────
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QColor(RADAR_LABEL_MUTED)))

        # CENTER
        p.drawText(
            QRectF(cx - 25, cy - r - 25, 50, 14), Qt.AlignmentFlag.AlignCenter, "CENTER"
        )
        # LEFT
        p.drawText(
            QRectF(cx - r - 15, cy - r + 20, 35, 14), Qt.AlignmentFlag.AlignCenter, "LEFT"
        )
        # RIGHT
        p.drawText(
            QRectF(cx + r - 20, cy - r + 20, 35, 14), Qt.AlignmentFlag.AlignCenter, "RIGHT"
        )

        # ── 7. Sweep Line ─────────────────────────────────────────────
        sx = cx + r * math.cos(math.radians(self._sweep))
        sy = cy - r * math.sin(math.radians(self._sweep))
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.5))
        p.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # ── 8. Cincin Luar ────────────────────────────────────────────
        p.setPen(QPen(QColor(RADAR_SWEEP), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 45 * 16, 90 * 16)

        # ── 10. Titik Pusat ───────────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(RADAR_SWEEP)))
        p.drawEllipse(QPointF(cx, cy), 3, 3)

        # ── 10. Obstacle Blips ────────────────────────────────────────
        for obs in self._obstacles:
            distance_m = obs.get("distance_m", 0)
            if distance_m <= 0:
                continue

            # Hitung sudut berdasarkan koordinat horizontal dari Bounding Box
            if "bbox" in obs:
                x1, y1, x2, y2 = obs["bbox"]
                bbox_cx = (x1 + x2) / 2
                # Asumsi resolusi kamera 640px. Pemetaan FOV 90 derajat:
                # Kiri layar (0px) -> 135°, Kanan layar (640px) -> 45°
                angle_deg = 135 - (bbox_cx / 640.0) * 90
            else:
                zone = obs.get("zone", "center").upper()
                angle_deg = {"LEFT": 120, "CENTER": 90, "RIGHT": 60}.get(zone, 90)

            # Batasi agar blip tidak keluar dari wedge radar
            angle_deg = max(45, min(135, angle_deg))

            dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
            bx = cx + dist_frac * r * math.cos(math.radians(angle_deg))
            by = cy - dist_frac * r * math.sin(math.radians(angle_deg))

            zone_str = obs.get("zone", "center").upper()
            if zone_str == "CENTER":
                color = QColor(RADAR_BLIP_CENTER)
            elif zone_str in ("LEFT", "RIGHT"):
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
