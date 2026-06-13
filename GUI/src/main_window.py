import sys

from Alert_panel import AlertPanel
from camera_thread import CameraThread
from controls_panel import ControlsPanel
from depth_view import DepthView
from radar_view import RadarView
from detection_config import DetectionConfig
from frame_processor import FrameProcessor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow,
    QVBoxLayout, QWidget, QScrollArea
)
from styles import STATUS_INACTIVE


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision System PBL GUI")
        self.resize(1280, 768)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # ── Bagian Kiri: Camera View ──────────────────────────────────
        self.area_kamera = DepthView()
        main_layout.addWidget(self.area_kamera, stretch=75)

        # ── Bagian Kanan ──────────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        self.controls_panel = ControlsPanel()
        self.alert_panel    = AlertPanel()
        self.radar          = RadarView()

        # ── Wrapper untuk center radar ────────────────────────────────
        radar_wrapper = QWidget()
        radar_wrapper_layout = QHBoxLayout(radar_wrapper)
        radar_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        radar_wrapper_layout.addStretch()
        radar_wrapper_layout.addWidget(self.radar)
        radar_wrapper_layout.addStretch()

        right_layout.addWidget(self.controls_panel)
        right_layout.addWidget(self.alert_panel)
        right_layout.addWidget(radar_wrapper)
        right_layout.addStretch()

        # ── Bungkus dengan ScrollArea ─────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setStyleSheet("QScrollArea { border: none; }")

        main_layout.addWidget(scroll, stretch=25)

        # ── Vision pipeline ───────────────────────────────────────────
        config = DetectionConfig()
        self.frame_processor = FrameProcessor(config)

        # ── Kamera thread ─────────────────────────────────────────────
        self.camera_thread = CameraThread(
            camera_index=0,
            parent=self,
            processor=self.frame_processor,
        )
        self.camera_thread.frame_pair_ready.connect(self.area_kamera.update_frames)
        self.camera_thread.distance_info_ready.connect(self.alert_panel.update_info)
        self.camera_thread.error.connect(self._on_camera_error)
        self.camera_thread.set_depth_thresholds(
            self.controls_panel.spin_depth_min.value(),
            self.controls_panel.spin_depth_max.value(),
        )

        # ── Sambungkan sinyal GUI internal ────────────────────────────
        self._connect_signals()

    def _connect_signals(self):
        self.controls_panel.thresholds_changed.connect(
            self.alert_panel.set_thresholds
        )
        self.controls_panel.camera_start_requested.connect(
            self._on_camera_start
        )
        self.controls_panel.camera_stop_requested.connect(
            self._on_camera_stop
        )
        self.controls_panel.view_mode_changed.connect(
            self.area_kamera.set_view_mode
        )
        self.controls_panel.depth_threshold_changed.connect(
            self.camera_thread.set_depth_thresholds
        )

    def _on_camera_start(self):
        self.camera_thread.start_capture()

    def _on_camera_stop(self):
        self.camera_thread.stop_capture()
        self.alert_panel.update_info("Menunggu...", None)
        self.radar.clear_obstacles()

    def _on_camera_error(self, message: str):
        self.controls_panel._camera_running = False
        self.controls_panel.btn_start.setEnabled(True)
        self.controls_panel.btn_stop.setEnabled(False)
        self.controls_panel.camera_status_label.setText(f"Status: {message}")
        self.controls_panel.camera_status_label.setStyleSheet(STATUS_INACTIVE)

    def closeEvent(self, a0):
        self.camera_thread.stop_capture()
        super().closeEvent(a0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())