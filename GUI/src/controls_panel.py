from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ui_config import DEPTH_MAX_M, DEPTH_MIN_M, THRESHOLD_DANGER, THRESHOLD_WARNING


class ControlsPanel(QWidget):
    camera_start_requested = pyqtSignal()
    camera_stop_requested = pyqtSignal()
    thresholds_changed = pyqtSignal(float, float)
    depth_threshold_changed = pyqtSignal(float, float)
    view_mode_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_running = False
        self._build_ui()
        self._apply_style()
        self._on_view_change(0)
        self._set_alert_info("Klik Apply untuk kirim threshold alert.")
        self._set_depth_info("Klik Apply untuk kirim threshold depth.")

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        title = QLabel("🎛️  Controls Panel")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        cam_group = QGroupBox("📷  Kamera Intel RealSense")
        cam_group.setFont(QFont("Segoe UI", 10))
        cam_layout = QVBoxLayout(cam_group)

        self.camera_status_label = QLabel("Status: Tidak Aktif")
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_status_label.setFont(QFont("Segoe UI", 10))
        cam_layout.addWidget(self.camera_status_label)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        cam_layout.addLayout(btn_row)
        main_layout.addWidget(cam_group)

        view_group = QGroupBox("📺  Pilih Tampilan")
        view_group.setFont(QFont("Segoe UI", 10))
        view_layout = QHBoxLayout(view_group)

        self.btn_view_rgb = QPushButton("RGB")
        self.btn_view_depth = QPushButton("Depth")
        self.btn_view_both = QPushButton("Overlay View")

        self.btn_view_rgb.clicked.connect(lambda: self._on_view_change(0))
        self.btn_view_depth.clicked.connect(lambda: self._on_view_change(1))
        self.btn_view_both.clicked.connect(lambda: self._on_view_change(2))

        view_layout.addWidget(self.btn_view_rgb)
        view_layout.addWidget(self.btn_view_depth)
        view_layout.addWidget(self.btn_view_both)
        main_layout.addWidget(view_group)

        alert_group = QGroupBox("⚙️  Threshold Alert (meter)")
        alert_group.setFont(QFont("Segoe UI", 10))
        alert_layout = QVBoxLayout(alert_group)

        warn_row = QHBoxLayout()
        warn_row.addWidget(QLabel("Warning:"))
        self.spin_warning = QDoubleSpinBox()
        self.spin_warning.setRange(0.2, 20.0)
        self.spin_warning.setDecimals(2)
        self.spin_warning.setSingleStep(0.1)
        self.spin_warning.setValue(THRESHOLD_WARNING)
        warn_row.addWidget(self.spin_warning)
        alert_layout.addLayout(warn_row)

        danger_row = QHBoxLayout()
        danger_row.addWidget(QLabel("Danger:"))
        self.spin_danger = QDoubleSpinBox()
        self.spin_danger.setRange(0.1, 20.0)
        self.spin_danger.setDecimals(2)
        self.spin_danger.setSingleStep(0.1)
        self.spin_danger.setValue(THRESHOLD_DANGER)
        danger_row.addWidget(self.spin_danger)
        alert_layout.addLayout(danger_row)

        self.btn_apply_alert = QPushButton("Apply Alert Threshold")
        self.btn_apply_alert.clicked.connect(self._on_apply_threshold)
        alert_layout.addWidget(self.btn_apply_alert)

        self.thr_info = QLabel()
        self.thr_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thr_info.setWordWrap(True)
        alert_layout.addWidget(self.thr_info)
        main_layout.addWidget(alert_group)

        depth_group = QGroupBox("🧭  Threshold Depth View (meter)")
        depth_group.setFont(QFont("Segoe UI", 10))
        depth_layout = QVBoxLayout(depth_group)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Depth Min:"))
        self.spin_depth_min = QDoubleSpinBox()
        self.spin_depth_min.setRange(0.1, 20.0)
        self.spin_depth_min.setDecimals(2)
        self.spin_depth_min.setSingleStep(0.1)
        self.spin_depth_min.setValue(DEPTH_MIN_M)
        min_row.addWidget(self.spin_depth_min)
        depth_layout.addLayout(min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Depth Max:"))
        self.spin_depth_max = QDoubleSpinBox()
        self.spin_depth_max.setRange(0.2, 20.0)
        self.spin_depth_max.setDecimals(2)
        self.spin_depth_max.setSingleStep(0.1)
        self.spin_depth_max.setValue(DEPTH_MAX_M)
        max_row.addWidget(self.spin_depth_max)
        depth_layout.addLayout(max_row)

        self.btn_apply_depth = QPushButton("Apply Depth Threshold")
        self.btn_apply_depth.clicked.connect(self._on_apply_depth_threshold)
        depth_layout.addWidget(self.btn_apply_depth)

        self.depth_info = QLabel()
        self.depth_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_info.setWordWrap(True)
        depth_layout.addWidget(self.depth_info)
        main_layout.addWidget(depth_group)
        main_layout.addStretch()

    def _apply_style(self):
        self.setStyleSheet(
            """
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
            """
        )

    def _on_view_change(self, mode_index):
        default_style = "background-color: #313244; font-weight: normal; color: #cdd6f4;"
        active_style = "background-color: #89b4fa; color: #1e1e2e; font-weight: bold;"

        self.btn_view_rgb.setStyleSheet(active_style if mode_index == 0 else default_style)
        self.btn_view_depth.setStyleSheet(active_style if mode_index == 1 else default_style)
        self.btn_view_both.setStyleSheet(active_style if mode_index == 2 else default_style)
        self.view_mode_changed.emit(mode_index)

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
        danger = self.spin_danger.value()

        if danger >= warning:
            self._set_alert_info("❌ DANGER harus lebih kecil dari WARNING!", is_error=True)
            return

        self._set_alert_info(f"✔ Alert: WARN={warning:.2f} m | DANGER={danger:.2f} m")
        self.thresholds_changed.emit(warning, danger)

    def _on_apply_depth_threshold(self):
        depth_min = self.spin_depth_min.value()
        depth_max = self.spin_depth_max.value()

        if depth_min >= depth_max:
            self._set_depth_info("❌ Depth Min harus lebih kecil dari Depth Max!", is_error=True)
            return

        self._set_depth_info(f"✔ Depth View: MIN={depth_min:.2f} m | MAX={depth_max:.2f} m")
        self.depth_threshold_changed.emit(depth_min, depth_max)

    def _set_alert_info(self, text: str, is_error: bool = False):
        self.thr_info.setText(text)
        self.thr_info.setStyleSheet("color: #f38ba8;" if is_error else "color: #a6e3a1;")

    def _set_depth_info(self, text: str, is_error: bool = False):
        self.depth_info.setText(text)
        self.depth_info.setStyleSheet("color: #f38ba8;" if is_error else "color: #a6e3a1;")
