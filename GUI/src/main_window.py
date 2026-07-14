import os
import sys
from pathlib import Path

from GUI.src.alert_panel import AlertPanel
from Vision.src.camera_thread import CameraThread
from Vision.src.video_playback_thread import VideoPlaybackThread
from GUI.src.controls_panel import ControlsPanel
from GUI.src.depth_view import DepthView
from GUI.src.radar_view import RadarView
from Vision.inc.detection_config import DetectionConfig
from Vision.src.frame_processor import FrameProcessor, FusionStage, NavigationStage, YOLODetectionStage, VisualAnnotationStage
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
        yolo_model_path = models_dir / "ModelRGB_V4.2.pt"
        depth_model_path = models_dir / "ModelDepth_V4.pt"
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

        # ── Add NavigationStage (R1) ───────────────────────────────
        self.frame_processor.add_stage(NavigationStage(config=config))

        # ── Add VisualAnnotationStage (R5) ──────────────────────────
        self.frame_processor.add_stage(VisualAnnotationStage(config=config))

        # ── Kamera thread ─────────────────────────────────────────────
        self.camera_thread = CameraThread(
            camera_index=0,
            parent=self,
            processor=self.frame_processor,
        )
        self._connect_source_signals(self.camera_thread)
        self.camera_thread.set_depth_thresholds(
            self.controls_panel.spin_depth_min.value(),
            self.controls_panel.spin_depth_max.value(),
        )

        # ── Video playback thread (initialized on demand) ────────────
        self._playback_thread: VideoPlaybackThread | None = None
        self._selected_video_dir: str = ""
        self._input_source: str = "live"  # "live" or "video"

        # ── Sambungkan sinyal GUI internal ────────────────────────────
        self._connect_signals()

    # ── Signal wiring helpers ─────────────────────────────────────────

    def _connect_source_signals(self, source_thread):
        """Connect the common signals from a camera/playback thread to GUI."""
        source_thread.frame_pair_ready.connect(self.area_kamera.update_frames)
        source_thread.distance_info_ready.connect(self.alert_panel.update_info)
        source_thread.obstacles_ready.connect(self._on_obstacles_ready)
        source_thread.navigation_ready.connect(self.alert_panel.update_navigation)
        source_thread.navigation_ready.connect(self.radar.update_navigation)
        source_thread.light_mode_changed.connect(self._on_light_mode_changed)
        source_thread.error.connect(self._on_camera_error)

    def _disconnect_source_signals(self, source_thread):
        """Disconnect common signals from a camera/playback thread."""
        try:
            source_thread.frame_pair_ready.disconnect(self.area_kamera.update_frames)
            source_thread.distance_info_ready.disconnect(self.alert_panel.update_info)
            source_thread.obstacles_ready.disconnect(self._on_obstacles_ready)
            source_thread.navigation_ready.disconnect(self.alert_panel.update_navigation)
            source_thread.navigation_ready.disconnect(self.radar.update_navigation)
            source_thread.light_mode_changed.disconnect(self._on_light_mode_changed)
            source_thread.error.disconnect(self._on_camera_error)
        except (TypeError, RuntimeError):
            pass  # Already disconnected

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

        # Video playback signals
        self.controls_panel.input_source_changed.connect(
            self._on_input_source_changed
        )
        self.controls_panel.video_file_selected.connect(
            self._on_video_file_selected
        )
        self.controls_panel.playback_start_requested.connect(
            self._on_playback_start
        )
        self.controls_panel.playback_stop_requested.connect(
            self._on_playback_stop
        )
        self.controls_panel.playback_pause_toggled.connect(
            self._on_playback_pause_toggled
        )
        self.controls_panel.playback_speed_changed.connect(
            self._on_playback_speed_changed
        )
        self.controls_panel.playback_loop_toggled.connect(
            self._on_playback_loop_toggled
        )

    # ── Live camera handlers ──────────────────────────────────────────

    def _on_camera_start(self):
        self.camera_thread.start_capture()
        self.radar.start_animation()

    def _on_camera_stop(self):
        self.camera_thread.stop_capture()
        self.radar.stop_animation()
        self.alert_panel.update_info("Menunggu...", None)
        self.alert_panel.update_navigation({})
        self.radar.clear_obstacles()

    # ── Input source switching ────────────────────────────────────────

    def _on_input_source_changed(self, source: str):
        """Handle switching between live camera and video file input."""
        # Stop any active source first
        if self._input_source == "live":
            self.camera_thread.stop_capture()
        elif self._input_source == "video" and self._playback_thread is not None:
            self._stop_playback_thread()

        self._input_source = source
        self.alert_panel.update_info("Menunggu...", None)
        self.alert_panel.update_navigation({})
        self.radar.clear_obstacles()

    def _on_video_file_selected(self, directory: str):
        """Store the selected recording directory."""
        self._selected_video_dir = directory

    def _on_playback_start(self):
        """Start video playback from the selected recording directory."""
        if not self._selected_video_dir:
            self.controls_panel.on_playback_finished()
            return

        # Stop any existing playback thread
        self._stop_playback_thread()

        # Create and start new playback thread
        self._playback_thread = VideoPlaybackThread(
            recording_dir=self._selected_video_dir,
            parent=self,
            processor=self.frame_processor,
        )
        self._connect_source_signals(self._playback_thread)
        self.radar.start_animation()
        self._playback_thread.playback_progress.connect(
            self.controls_panel.update_playback_progress
        )
        self._playback_thread.playback_finished.connect(
            self.controls_panel.on_playback_finished
        )
        self._playback_thread.set_depth_thresholds(
            self.controls_panel.spin_depth_min.value(),
            self.controls_panel.spin_depth_max.value(),
        )
        self._playback_thread.start_playback()

    def _on_playback_stop(self):
        """Stop video playback."""
        self._stop_playback_thread()
        self.radar.stop_animation()
        self.alert_panel.update_info("Menunggu...", None)
        self.alert_panel.update_navigation({})
        self.radar.clear_obstacles()

    def _stop_playback_thread(self):
        """Safely stop and clean up the playback thread."""
        if self._playback_thread is not None:
            self._disconnect_source_signals(self._playback_thread)
            try:
                self._playback_thread.playback_progress.disconnect(
                    self.controls_panel.update_playback_progress
                )
                self._playback_thread.playback_finished.disconnect(
                    self.controls_panel.on_playback_finished
                )
            except (TypeError, RuntimeError):
                pass
            self._playback_thread.stop_playback()
            self._playback_thread = None

    def _on_playback_pause_toggled(self, paused: bool):
        """Pause or resume video playback."""
        if self._playback_thread is not None:
            self._playback_thread.set_paused(paused)

    def _on_playback_speed_changed(self, speed: float):
        """Change playback speed."""
        if self._playback_thread is not None:
            self._playback_thread.set_speed(speed)

    def _on_playback_loop_toggled(self, loop: bool):
        """Enable or disable playback loop."""
        if self._playback_thread is not None:
            self._playback_thread.set_loop(loop)

    # ── Shared handlers ───────────────────────────────────────────────

    def _on_obstacles_ready(self, obstacles: list):
        radar_data = []
        for obs in obstacles:
            bbox = obs.get("bbox", [0, 0, 0, 0])
            radar_data.append({
                "bbox": bbox,
                "distance_m": obs.get("distance_m", 0),
                "zone": obs.get("zone", "CENTER").upper(),
            })
        self.radar.update_obstacles(radar_data)

    def _on_light_mode_changed(self, is_dark: bool):
        self.controls_panel.update_auto_view(is_dark)

    def _on_camera_error(self, message: str):
        self.controls_panel._camera_running = False
        self.controls_panel.btn_start.setEnabled(True)
        self.controls_panel.btn_stop.setEnabled(False)
        self.controls_panel.camera_status_label.setText(f"Status: {message}")
        self.controls_panel.camera_status_label.setStyleSheet(STATUS_INACTIVE)

    def closeEvent(self, a0):
        self.camera_thread.stop_capture()
        self._stop_playback_thread()
        super().closeEvent(a0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())