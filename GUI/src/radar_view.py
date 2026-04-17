import math
import random
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPointF, QRect
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QConicalGradient, QRadialGradient, QPainterPath

# Warna Palette HUD
GREEN   = QColor(0,  255, 100)
BG_DARK = QColor(0,   12,  24, 215)
BORDER  = QColor(0,  200, 255, 200)

def mf(size: int, bold: bool = False) -> QFont:
    return QFont("Courier New", size, QFont.Weight.Bold if bold else QFont.Weight.Normal)

class RadarView(QWidget):
    SIZE = 250 # Ukuran widget diperbesar sedikit agar lebih jelas

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.SIZE, self.SIZE)
        
        self._sweep = 0.0
        rng = random.Random()
        # Membuat titik blip [angle_deg, dist_fraction, alpha]
        self._blips = [[rng.uniform(0, 360), rng.uniform(0.15, 0.90), 0] for _ in range(6)]
        
        # Timer untuk memutar radar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(45)

    def _tick(self):
        self._sweep = (self._sweep + 4) % 360
        for b in self._blips:
            if (self._sweep - b[0]) % 360 < 5:
                b[2] = 255
            else:
                b[2] = max(0, b[2] - 5)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Menggambar Background "Kaca" (GlassPanel)
        p.setPen(QPen(QColor(0, 230, 90, 225), 1))
        p.setBrush(QBrush(QColor(0, 7, 3, 235)))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(cx, cy) - 20

        # 2. Menggambar Cincin Radar
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(1, 5):
            p.setPen(QPen(QColor(0, 210, 80, 60 + i * 25), 1))
            p.drawEllipse(QPointF(cx, cy), r * i / 4, r * i / 4)

        # 3. Menggambar Garis Silang (Sumbu)
        p.setPen(QPen(QColor(0, 180, 70, 100), 1))
        for deg in range(0, 180, 45):
            rad = math.radians(deg)
            p.drawLine(
                QPointF(cx - r * math.cos(rad), cy - r * math.sin(rad)),
                QPointF(cx + r * math.cos(rad), cy + r * math.sin(rad))
            )

        # 4. Menggambar Sapuan Cahaya Radar (Sweep Cone)
        sg = QConicalGradient(QPointF(cx, cy), 90 - self._sweep)
        sg.setColorAt(0.00, QColor(0, 255, 80, 175))
        sg.setColorAt(0.10, QColor(0, 255, 80, 20))
        sg.setColorAt(0.11, Qt.GlobalColor.transparent)
        sg.setColorAt(1.00, Qt.GlobalColor.transparent)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(sg))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Garis sapuan tegas
        sr = math.radians(90 - self._sweep)
        p.setPen(QPen(QColor(0, 255, 100), 2))
        p.drawLine(QPointF(cx, cy), QPointF(cx + r * math.cos(sr), cy - r * math.sin(sr)))

        # 5. Menggambar Titik Halangan (Blips)
        for b in self._blips:
            if b[2] > 8:
                br = math.radians(90 - b[0])
                bx = cx + b[1] * r * math.cos(br)
                by = cy - b[1] * r * math.sin(br)
                glow = QRadialGradient(QPointF(bx, by), 13)
                glow.setColorAt(0, QColor(0, 255, 80, int(b[2])))
                glow.setColorAt(1, Qt.GlobalColor.transparent)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(bx, by), 13, 13)
                p.setBrush(QBrush(QColor(140, 255, 170, int(b[2]))))
                p.drawEllipse(QPointF(bx, by), 3.5, 3.5)

        # 6. Menggambar Cincin Luar & Label
        p.setPen(QPen(QColor(0, 230, 90, 225), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setBrush(QBrush(GREEN))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 3, 3)

        p.setPen(QPen(QColor(0, 220, 80)))
        p.setFont(mf(8))
        p.drawText(QRect(0, int(h - 20), int(w), 14), Qt.AlignmentFlag.AlignCenter, "━  RADAR  ━")
        p.end()