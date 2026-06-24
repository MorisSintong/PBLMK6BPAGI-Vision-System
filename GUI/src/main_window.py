import os
import sys
from pathlib import Path

from GUI.src.alert_panel import AlertPanel
from Vision.src.camera_thread import CameraThread
from GUI.src.controls_panel import ControlsPanel
from GUI.src.depth_view import DepthView
from GUI.src.radar_view import RadarView
from Vision.inc.detection_config import DetectionConfig
from Vision.src.frame_processor import FrameProcessor, FusionStage, YOLODetectionStage, VisualAnnotationStage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow,
    QVBoxLayout, QWidget, QScrollArea
)
from GUI.inc.styles import STATUS_INACTIVE


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

        # ── Add YOLO stage (R2) ──────────────────────────────────────
        project_root = Path(__file__).resolve().parent.parent.parent
        models_dir = project_root / "Vision" / "models"
        yolo_model_path = models_dir / "model_v3.pt"
        depth_model_path = models_dir / "modelDepth.pt"
        yolo_model_root = project_root / "yolov8n.pt"
        best_model_path = models_dir / "security_best.pt"
        
        # Pick best available RGB model
        rgb_model = None
        if yolo_model_path.exists():
            rgb_model = str(yolo_model_path)
        elif yolo_model_root.exists():
            rgb_model = str(yolo_model_root)
        elif best_model_path.exists():
            rgb_model = str(best_model_path)

        if rgb_model:
            self.frame_processor.add_stage(
                YOLODetectionStage(
                    model_path=rgb_model,
                    depth_model_path=str(depth_model_path) if depth_model_path.exists() else None,
                )
            )

        # ── Add FusionStage (R4) ────────────────────────────────────
        self.frame_processor.add_stage(FusionStage(config=config))

        # ── Add VisualAnnotationStage (R5) ──────────────────────────
        self.frame_processor.add_stage(VisualAnnotationStage(config=config))

        # ── Kamera thread ─────────────────────────────────────────────
        self.camera_thread = CameraThread(
            camera_index=0,
            parent=self,
            processor=self.frame_processor,
        )
        self.camera_thread.frame_pair_ready.connect(self.area_kamera.update_frames)
        self.camera_thread.distance_info_ready.connect(self.alert_panel.update_info)
        self.camera_thread.obstacles_ready.connect(self._on_obstacles_ready)
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
        self.controls_panel.thresholds_changed.connect(
            self.frame_processor.set_action_thresholds
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

    def _on_obstacles_ready(self, obstacles: list):
        radar_data = []
        frame_width = 640
        for obs in obstacles:
            bbox = obs.get("bbox", [0, 0, 0, 0])
            center_x = bbox[0] + bbox[2] // 2
            angle_deg = 180 * center_x / frame_width
            radar_data.append({
                "angle_deg": angle_deg,
                "distance_m": obs.get("distance_m", 0),
                "zone": obs.get("zone", "CENTER").upper(),
            })
        self.radar.update_obstacles(radar_data)

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