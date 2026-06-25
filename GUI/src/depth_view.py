from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget
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

        # Page 2: Split screen
        self.overlay_page = self._create_overlay_page()
        self.stacked_widget.addWidget(self.overlay_page)

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

    def _create_overlay_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label_overlay_rgb = self._create_screen("KAMERA OFFLINE\n(RGB)")
        self.label_overlay_depth = self._create_screen("DEPTH BELUM TERSEDIA\n(Depth)")

        layout.addWidget(self.label_overlay_rgb, 1)
        layout.addWidget(self.label_overlay_depth, 1)
        return page

    def set_view_mode(self, mode_index):
        self.stacked_widget.setCurrentIndex(mode_index)

    def update_frames(self, rgb_image=None, depth_image=None):
        if rgb_image is not None and not rgb_image.isNull():
            rgb_pixmap = QPixmap.fromImage(rgb_image)
            current = self.stacked_widget.currentIndex()
            if current == 0:
                self.label_rgb.setPixmap(rgb_pixmap)
            elif current == 2:
                self.label_overlay_rgb.setPixmap(rgb_pixmap)

        if depth_image is not None and not depth_image.isNull():
            depth_pixmap = QPixmap.fromImage(depth_image)
            current = self.stacked_widget.currentIndex()
            if current == 1:
                self.label_depth.setPixmap(depth_pixmap)
            elif current == 2:
                self.label_overlay_depth.setPixmap(depth_pixmap)
