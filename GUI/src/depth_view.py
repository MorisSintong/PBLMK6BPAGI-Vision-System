from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt

class DepthView(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Menggunakan QStackedWidget untuk menumpuk layar
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # ── Halaman 0: Layar RGB Saja ──
        self.label_rgb = self._create_screen("KAMERA OFFLINE\n(Mode RGB)")
        self.stacked_widget.addWidget(self.label_rgb)

        # ── Halaman 1: Layar Depth Saja ──
        self.label_depth = self._create_screen("KAMERA OFFLINE\n(Mode Depth)")
        self.stacked_widget.addWidget(self.label_depth)

        # ── Halaman 2: Layar Keduanya (Split Screen) ──
        self.label_combined = self._create_screen("KAMERA OFFLINE\n(Mode Overlay Object)")
        self.stacked_widget.addWidget(self.label_combined)

    def _create_screen(self, text):
        """Fungsi helper untuk membuat label layar hitam dengan format seragam"""
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #aaaaaa; 
            font-size: 20px;
            font-weight: bold;
            border: 2px dashed #555555;
        """)
        return lbl

    # ── FUNGSI KONTROL DARI GUI ──
    def set_view_mode(self, mode_index):
        """Mengubah halaman yang sedang aktif (0=RGB, 1=Depth, 2=Keduanya)"""
        self.stacked_widget.setCurrentIndex(mode_index)

    # ── FUNGSI PENERIMA DARI TIM VISION ──
    def update_frames(self, rgb_pixmap=None, depth_pixmap=None):
        """
        Fungsi ini siap menerima 2 gambar sekaligus dari camera_thread.
        """
        if rgb_pixmap is not None:
            # Update ke layar RGB tunggal dan layar overlay
            self.label_rgb.setPixmap(rgb_pixmap)
            self.label_rgb.setScaledContents(True)
            self.label_combined.setPixmap(rgb_pixmap)
            self.label_combined.setScaledContents(True)

        if depth_pixmap is not None:
            # Update ke layar Depth tunggal
            self.label_depth.setPixmap(depth_pixmap)
            self.label_depth.setScaledContents(True)
