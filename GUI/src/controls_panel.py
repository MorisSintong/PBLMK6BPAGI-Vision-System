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
from GUI.inc.ui_config import DEPTH_MAX_M, DEPTH_MIN_M, THRESHOLD_DANGER, THRESHOLD_WARNING
from GUI.inc.styles import (
    STATUS_ACTIVE, STATUS_INACTIVE,
    STATUS_ERROR, STATUS_SUCCESS,
    VIEW_ACTIVE, VIEW_DEFAULT
)


class ControlsPanel(QWidget):
    camera_start_requested  = pyqtSignal()
    camera_stop_requested   = pyqtSignal()
    thresholds_changed      = pyqtSignal(float, float)
    depth_threshold_changed = pyqtSignal(float, float)
    view_mode_changed       = pyqtSignal(int)
    auto_mode_changed       = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_running = False
        self._auto_mode = True
        self._build_ui()
        self._apply_style()
        self._on_view_change_auto()
        self._set_alert_info("Klik Apply untuk kirim threshold alert.")
        self._set_depth_info("Klik Apply untuk kirim threshold depth.")

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ── Judul ─────────────────────────────────────────────────────
        title = QLabel("Controls Panel")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # ── Kamera ────────────────────────────────────────────────────
        cam_group = QGroupBox("Kamera Intel RealSense")
        cam_group.setFont(QFont("Segoe UI", 10))
        cam_layout = QVBoxLayout(cam_group)

        self.camera_status_label = QLabel("Status: Tidak Aktif")
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_status_label.setFont(QFont("Segoe UI", 10))
        cam_layout.addWidget(self.camera_status_label)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        cam_layout.addLayout(btn_row)
        main_layout.addWidget(cam_group)

        # ── Pilih Tampilan ────────────────────────────────────────────
        view_group = QGroupBox("Pilih Tampilan")
        view_group.setFont(QFont("Segoe UI", 10))
        view_layout = QHBoxLayout(view_group)

        self.btn_view_auto  = QPushButton("Auto")
        self.btn_view_rgb   = QPushButton("RGB")
        self.btn_view_depth = QPushButton("Depth")

        self.btn_view_auto.clicked.connect(self._on_view_change_auto)
        self.btn_view_rgb.clicked.connect(lambda: self._on_view_change(0))
        self.btn_view_depth.clicked.connect(lambda: self._on_view_change(1))

        view_layout.addWidget(self.btn_view_auto)
        view_layout.addWidget(self.btn_view_rgb)
        view_layout.addWidget(self.btn_view_depth)
        main_layout.addWidget(view_group)

        # ── Threshold Alert ───────────────────────────────────────────
        alert_group = QGroupBox("Threshold Alert (meter)")
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

        # ── Threshold Depth View ──────────────────────────────────────
        depth_group = QGroupBox("Threshold Depth View (meter)")
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
        from GUI.inc.styles import GLOBAL_STYLESHEET
        self.setStyleSheet(GLOBAL_STYLESHEET)

    def _on_view_change_auto(self):
        self._auto_mode = True
        self.btn_view_auto.setStyleSheet(VIEW_ACTIVE)
        self.btn_view_rgb.setStyleSheet(VIEW_DEFAULT)
        self.btn_view_depth.setStyleSheet(VIEW_DEFAULT)
        self.auto_mode_changed.emit(True)

    def _on_view_change(self, mode_index):
        self._auto_mode = False
        self.btn_view_auto.setStyleSheet(VIEW_DEFAULT)
        self.btn_view_rgb.setStyleSheet(VIEW_ACTIVE   if mode_index == 0 else VIEW_DEFAULT)
        self.btn_view_depth.setStyleSheet(VIEW_ACTIVE if mode_index == 1 else VIEW_DEFAULT)
        self.auto_mode_changed.emit(False)
        self.view_mode_changed.emit(mode_index)

    def update_auto_view(self, is_dark: bool):
        """Called by MainWindow when light_mode_changed fires in auto mode.
        Switches the DepthView and updates button highlights without emitting signals."""
        if not self._auto_mode:
            return
        mode_index = 1 if is_dark else 0
        self.btn_view_rgb.setStyleSheet(VIEW_ACTIVE   if mode_index == 0 else VIEW_DEFAULT)
        self.btn_view_depth.setStyleSheet(VIEW_ACTIVE if mode_index == 1 else VIEW_DEFAULT)
        self.view_mode_changed.emit(mode_index)

    def _on_start(self):
        self._camera_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.camera_status_label.setText("Status: Aktif")
        self.camera_status_label.setStyleSheet(STATUS_ACTIVE)
        self.camera_start_requested.emit()

    def _on_stop(self):
        self._camera_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_status_label.setText("Status: Tidak Aktif")
        self.camera_status_label.setStyleSheet(STATUS_INACTIVE)
        self.camera_stop_requested.emit()

    def _on_apply_threshold(self):
        warning = self.spin_warning.value()
        danger  = self.spin_danger.value()

        if danger >= warning:
            self._set_alert_info("DANGER harus lebih kecil dari WARNING!", is_error=True)
            return

        self._set_alert_info(f"Alert: WARN={warning:.2f} m | DANGER={danger:.2f} m")
        self.thresholds_changed.emit(warning, danger)

    def _on_apply_depth_threshold(self):
        depth_min = self.spin_depth_min.value()
        depth_max = self.spin_depth_max.value()

        if depth_min >= depth_max:
            self._set_depth_info("Depth Min harus lebih kecil dari Depth Max!", is_error=True)
            return

        self._set_depth_info(f"Depth View: MIN={depth_min:.2f} m | MAX={depth_max:.2f} m")
        self.depth_threshold_changed.emit(depth_min, depth_max)

    def _set_alert_info(self, text: str, is_error: bool = False):
        self.thr_info.setText(text)
        self.thr_info.setStyleSheet(STATUS_ERROR if is_error else STATUS_SUCCESS)

    def _set_depth_info(self, text: str, is_error: bool = False):
        self.depth_info.setText(text)
        self.depth_info.setStyleSheet(STATUS_ERROR if is_error else STATUS_SUCCESS)