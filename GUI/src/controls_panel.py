# GUI/src/controls_Panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox,
    QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlsPanel(QWidget):
    """
    Panel kontrol untuk:
    - Start / Stop kamera Intel RealSense
    - Setting threshold WARNING dan DANGER
    - Emit sinyal ke komponen lain (AlertPanel, dll.)
    """

    # ── Sinyal ─────────────────────────────────────────────────────────
    camera_start_requested = pyqtSignal()
    camera_stop_requested  = pyqtSignal()
    thresholds_changed     = pyqtSignal(float, float)   # (warning_m, danger_m)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_running = False
        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------ #
    #  UI Builder                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ── Judul ──────────────────────────────────────────────────────
        title = QLabel("🎛️  Controls Panel")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # ── Camera Control ─────────────────────────────────────────────
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

        # ── Threshold Settings ─────────────────────────────────────────
        thr_group = QGroupBox("⚙️  Threshold Jarak (meter)")
        thr_group.setFont(QFont("Segoe UI", 10))
        thr_layout = QVBoxLayout(thr_group)

        # WARNING threshold
        warn_row = QHBoxLayout()
        warn_row.addWidget(QLabel("⚠  WARNING  <"))
        self.spin_warning = QDoubleSpinBox()
        self.spin_warning.setRange(0.1, 10.0)
        self.spin_warning.setSingleStep(0.1)
        self.spin_warning.setValue(1.0)
        self.spin_warning.setSuffix("  m")
        self.spin_warning.setDecimals(1)
        warn_row.addWidget(self.spin_warning)
        thr_layout.addLayout(warn_row)

        # DANGER threshold
        danger_row = QHBoxLayout()
        danger_row.addWidget(QLabel("🔴 DANGER    <"))
        self.spin_danger = QDoubleSpinBox()
        self.spin_danger.setRange(0.1, 10.0)
        self.spin_danger.setSingleStep(0.1)
        self.spin_danger.setValue(0.5)
        self.spin_danger.setSuffix("  m")
        self.spin_danger.setDecimals(1)
        danger_row.addWidget(self.spin_danger)
        thr_layout.addLayout(danger_row)

        # Tombol Apply
        self.btn_apply = QPushButton("✔  Apply Threshold")
        self.btn_apply.clicked.connect(self._on_apply_threshold)
        thr_layout.addWidget(self.btn_apply)

        # Validasi info
        self.thr_info = QLabel("")
        self.thr_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thr_info.setFont(QFont("Segoe UI", 9))
        thr_layout.addWidget(self.thr_info)

        main_layout.addWidget(thr_group)
        main_layout.addStretch()

    def _apply_style(self):
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