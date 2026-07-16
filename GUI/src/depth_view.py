"""
GUI/src/depth_view.py — RGB / Depth display widget.

Role 6 (Adel) — GUI.

QStackedWidget with two pages (RGB, Depth) and an "Offline" placeholder for
each. setScaledContents(True) is set once in init; only the label of the
currently-visible page receives a new pixmap per frame.

Public API:
  update_frame_pair(rgb_qimage, depth_qimage) — main entry point from
    CameraThread / VideoPlaybackThread. Routes the pair to the right page
    based on view_mode (0=RGB, 1=Depth, 2=Auto-follow-light).
  set_view_mode(mode) — manual override; in Auto mode the view follows
    light_mode_changed(bool).
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class DepthView(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # Page 0: RGB only
        self.label_rgb = self._create_screen("KAMERA OFFLINE\n(Mode RGB)")
        self.stacked_widget.addWidget(self.label_rgb)

        # Page 1: Depth only
        self.label_depth = self._create_screen("KAMERA OFFLINE\n(Mode Depth)")
        self.stacked_widget.addWidget(self.label_depth)

    def _create_screen(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("""
            background-color: #1e1e1e;
            color: #aaaaaa;
            font-size: 20px;
            font-weight: bold;
            border: 2px dashed #555555;
        """)
        lbl.setScaledContents(True)
        return lbl

    def set_view_mode(self, mode_index):
        self.stacked_widget.setCurrentIndex(mode_index)

    def update_frames(self, rgb_image=None, depth_image=None):
        # Only convert+set the pixmap for the currently visible pane.
        # This eliminates ~1 wasted QPixmap.fromImage() call per frame.
        # NOTE: Do NOT pre-scale here — the label may not be laid out yet
        # (size would be 0,0 on first frames), making the scaled pixmap invalid.
        # Qt handles scaling on paint via setScaledContents(True) in _create_screen.
        current = self.stacked_widget.currentIndex()
        if current == 0 and rgb_image is not None and not rgb_image.isNull():
            self.label_rgb.setPixmap(QPixmap.fromImage(rgb_image))
        elif current == 1 and depth_image is not None and not depth_image.isNull():
            self.label_depth.setPixmap(QPixmap.fromImage(depth_image))
