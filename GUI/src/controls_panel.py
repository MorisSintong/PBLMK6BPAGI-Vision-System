from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox,
    QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class ControlsPanel(QWidget):
    # ── Tambahkan Sinyal Baru ──
    camera_start_requested = pyqtSignal()
    camera_stop_requested  = pyqtSignal()
    thresholds_changed     = pyqtSignal(float, float)
    view_mode_changed      = pyqtSignal(int) # Sinyal untuk mode layar (0, 1, atau 2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_running = False
        self._build_ui()
        self._apply_style()
        self._on_view_change(0) # Default saat aplikasi dibuka: Mode RGB (0)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        title = QLabel("🎛️  Controls Panel")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # ── 1. Camera Control (TETAP SAMA SEPERTI SEBELUMNYA) ──
        cam_group = QGroupBox("📷  Kamera Intel RealSense")
        cam_group.setFont(QFont("Segoe UI", 10))
        cam_layout = QVBoxLayout(cam_group)

        self.camera_status_label = QLabel("Status: Tidak Aktif")
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_status_label.setFont(QFont("Segoe UI", 10))
        cam_layout.addWidget(self.camera_status_label)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_stop  = QPushButton("⏹  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        cam_layout.addLayout(btn_row)
        main_layout.addWidget(cam_group)

        # ── 2. View Mode Control (BARU DITAMBAHKAN) ──
        view_group = QGroupBox("📺  Pilih Tampilan")
        view_group.setFont(QFont("Segoe UI", 10))
        view_layout = QHBoxLayout(view_group)

        self.btn_view_rgb = QPushButton("RGB")
        self.btn_view_depth = QPushButton("Depth")
        self.btn_view_both = QPushButton("Overlay View")

        # Hubungkan tombol ke fungsi pengubah layar
        self.btn_view_rgb.clicked.connect(lambda: self._on_view_change(0))
        self.btn_view_depth.clicked.connect(lambda: self._on_view_change(1))
        self.btn_view_both.clicked.connect(lambda: self._on_view_change(2))

        view_layout.addWidget(self.btn_view_rgb)
        view_layout.addWidget(self.btn_view_depth)
        view_layout.addWidget(self.btn_view_both)
        main_layout.addWidget(view_group)

        # ── 3. Threshold Settings (TETAP SAMA SEPERTI SEBELUMNYA) ──
        thr_group = QGroupBox("⚙️  Threshold Jarak (meter)")
        # ... (Sisa kode threshold biarkan sama persis seperti file asli kamu) ...
        # [PASTE KODE THRESHOLD KAMU DI SINI]

        # Tambahkan ke main layout
        main_layout.addWidget(thr_group)
        main_layout.addStretch()

    def _apply_style(self):
        # ... (Sisa kode style biarkan sama persis) ...
        self.setStyleSheet("""
            QWidget       { background-color: #1e1e2e; color: #cdd6f4; }
            QGroupBox     { border: 1px solid #45475a; border-radius: 8px;
                            margin-top: 8px; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                               color: #89b4fa; }
            QPushButton   { background-color: #313244; border: 1px solid #45475a;
                            border-radius: 6px; padding: 6px 14px; }
            QPushButton:hover   { background-color: #45475a; }
            QPushButton:disabled { color: #585b70; }
            QDoubleSpinBox { background-color: #313244; border: 1px solid #45475a;
                             border-radius: 4px; padding: 3px; }
        """)

    # ── FUNGSI BARU UNTUK MENGATUR TOMBOL TAMPILAN ──
    def _on_view_change(self, mode_index):
        # Reset semua warna tombol ke default
        default_style = "background-color: #313244; font-weight: normal; color: #cdd6f4;"
        active_style  = "background-color: #89b4fa; color: #1e1e2e; font-weight: bold;"

        self.btn_view_rgb.setStyleSheet(active_style if mode_index == 0 else default_style)
        self.btn_view_depth.setStyleSheet(active_style if mode_index == 1 else default_style)
        self.btn_view_both.setStyleSheet(active_style if mode_index == 2 else default_style)

        # Pancarkan sinyal ke main_window
        self.view_mode_changed.emit(mode_index)
# ------------------------------------------------------------------ #
    #  Slot handlers                                                       #
    # ------------------------------------------------------------------ #
    def _on_start(self):
        self._camera_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.camera_status_label.setText("Status: ✅ Aktif")
        self.camera_status_label.setStyleSheet("color: #a6e3a1;")
        self.camera_start_requested.emit()

    def _on_stop(self):
        self._camera_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_status_label.setText("Status: ⛔ Tidak Aktif")
        self.camera_status_label.setStyleSheet("color: #f38ba8;")
        self.camera_stop_requested.emit()

    def _on_apply_threshold(self):
        warning = self.spin_warning.value()
        danger  = self.spin_danger.value()

        # Validasi: danger harus < warning
        if danger >= warning:
            self.thr_info.setText("❌ DANGER harus lebih kecil dari WARNING!")
            self.thr_info.setStyleSheet("color: #f38ba8;")
            return

        self.thr_info.setText(f"✔ Applied: WARN={warning}m  |  DANGER={danger}m")
        self.thr_info.setStyleSheet("color: #a6e3a1;")
        self.thresholds_changed.emit(warning, danger)
    # ... (Sisa slot handlers seperti _on_start, _on_stop, _on_apply biarkan sama) ...