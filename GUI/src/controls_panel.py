from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
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

    # ── Input source signals ─────────────────────────────────────────────
    input_source_changed    = pyqtSignal(str)           # "live" or "video"
    video_file_selected     = pyqtSignal(str)            # recording directory path
    playback_start_requested = pyqtSignal()
    playback_stop_requested  = pyqtSignal()
    playback_pause_toggled   = pyqtSignal(bool)          # True = paused
    playback_speed_changed   = pyqtSignal(float)         # 0.25 - 4.0
    playback_loop_toggled    = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_running = False
        self._auto_mode = True
        self._playback_active = False
        self._playback_paused = False
        self._selected_video_dir = ""
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

        # ── Input Source ──────────────────────────────────────────────
        source_group = QGroupBox("Input Source")
        source_group.setFont(QFont("Segoe UI", 10))
        source_layout = QVBoxLayout(source_group)

        self.combo_source = QComboBox()
        self.combo_source.addItem("Live Camera", "live")
        self.combo_source.addItem("Video File", "video")
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.combo_source)

        # Video file browser (hidden initially)
        self._video_browse_widget = QWidget()
        browse_layout = QVBoxLayout(self._video_browse_widget)
        browse_layout.setContentsMargins(0, 4, 0, 0)

        browse_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_video)
        browse_row.addWidget(self.btn_browse)
        browse_layout.addLayout(browse_row)

        self.lbl_video_path = QLabel("Belum ada file dipilih")
        self.lbl_video_path.setWordWrap(True)
        self.lbl_video_path.setStyleSheet("color: #585b70; font-size: 9px;")
        browse_layout.addWidget(self.lbl_video_path)

        # Playback controls
        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("\u25b6 Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._on_play_pause)
        playback_row.addWidget(self.btn_play_pause)

        self.btn_playback_stop = QPushButton("\u25a0 Stop")
        self.btn_playback_stop.setEnabled(False)
        self.btn_playback_stop.clicked.connect(self._on_playback_stop)
        playback_row.addWidget(self.btn_playback_stop)

        self.btn_loop = QPushButton("\U0001f501")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setMaximumWidth(36)
        self.btn_loop.setToolTip("Loop playback")
        self.btn_loop.clicked.connect(self._on_loop_toggled)
        playback_row.addWidget(self.btn_loop)
        browse_layout.addLayout(playback_row)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setMinimum(1)   # 0.25x
        self.slider_speed.setMaximum(8)   # 2.0x
        self.slider_speed.setValue(4)     # 1.0x default
        self.slider_speed.setTickInterval(1)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.slider_speed)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setMinimumWidth(32)
        speed_row.addWidget(self.lbl_speed)
        browse_layout.addLayout(speed_row)

        # Progress label
        self.lbl_playback_progress = QLabel("")
        self.lbl_playback_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_playback_progress.setStyleSheet("color: #89b4fa; font-size: 10px;")
        browse_layout.addWidget(self.lbl_playback_progress)

        self._video_browse_widget.setVisible(False)
        source_layout.addWidget(self._video_browse_widget)
        main_layout.addWidget(source_group)

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

        self._cam_group = cam_group
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

    # ── Input Source handlers ─────────────────────────────────────────────

    def _on_source_changed(self, index: int):
        source = self.combo_source.currentData()
        is_video = source == "video"
        self._video_browse_widget.setVisible(is_video)
        self._cam_group.setVisible(not is_video)
        self.input_source_changed.emit(source)

    def _on_browse_video(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Recording Folder", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self._selected_video_dir = directory
            # Show just the folder name for readability
            folder_name = directory.split("/")[-1] or directory.split("\\")[-1]
            self.lbl_video_path.setText(f"\U0001f4c1 {folder_name}")
            self.lbl_video_path.setToolTip(directory)
            self.lbl_video_path.setStyleSheet("color: #a6e3a1; font-size: 9px;")
            self.btn_play_pause.setEnabled(True)
            self.video_file_selected.emit(directory)

            # Auto-start playback immediately after selecting a folder
            self._on_play_pause()

    def _on_play_pause(self):
        if not self._playback_active:
            # Start playback
            self._playback_active = True
            self._playback_paused = False
            self.btn_play_pause.setText("\u23f8 Pause")
            self.btn_playback_stop.setEnabled(True)
            self.playback_start_requested.emit()
        else:
            # Toggle pause
            self._playback_paused = not self._playback_paused
            if self._playback_paused:
                self.btn_play_pause.setText("\u25b6 Resume")
            else:
                self.btn_play_pause.setText("\u23f8 Pause")
            self.playback_pause_toggled.emit(self._playback_paused)

    def _on_playback_stop(self):
        self._playback_active = False
        self._playback_paused = False
        self.btn_play_pause.setText("\u25b6 Play")
        self.btn_play_pause.setEnabled(True)
        self.btn_playback_stop.setEnabled(False)
        self.lbl_playback_progress.setText("")
        self.playback_stop_requested.emit()

    def _on_speed_changed(self, value: int):
        # Map slider 1-8 to speed: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        speed = value * 0.25
        self.lbl_speed.setText(f"{speed:.2g}x")
        self.playback_speed_changed.emit(speed)

    def _on_loop_toggled(self):
        is_loop = self.btn_loop.isChecked()
        self.btn_loop.setStyleSheet(
            VIEW_ACTIVE if is_loop else VIEW_DEFAULT
        )
        self.playback_loop_toggled.emit(is_loop)

    def update_playback_progress(self, current: int, total: int):
        """Update the playback progress label (called from MainWindow)."""
        self.lbl_playback_progress.setText(f"Frame {current}/{total}")

    def on_playback_finished(self):
        """Reset playback controls when playback ends."""
        self._playback_active = False
        self._playback_paused = False
        self.btn_play_pause.setText("\u25b6 Play")
        self.btn_play_pause.setEnabled(True)
        self.btn_playback_stop.setEnabled(False)
        self.lbl_playback_progress.setText("Selesai")

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