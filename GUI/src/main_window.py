import sys
from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout
)
from depth_view     import DepthView
from Alert_panel    import AlertPanel
from controls_panel import ControlsPanel
from camera_thread  import CameraThread

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
        # Stretch ditingkatkan menjadi 80 agar lebih lebar
        main_layout.addWidget(self.area_kamera, stretch=80) 

        # ── Bagian Kanan: Controls + Alert (Tanpa Tab) ────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        self.controls_panel = ControlsPanel()
        self.alert_panel    = AlertPanel()

        right_layout.addWidget(self.controls_panel, stretch=40)
        right_layout.addWidget(self.alert_panel,    stretch=60)

        # Stretch dikurangi menjadi 20 agar panel kanan lebih ramping
        main_layout.addWidget(right_panel, stretch=20) 

        # ── Sambungkan sinyal GUI internal ────────────────────────────
        self._connect_signals()

        # ── Kamera thread ─────────────────────────────────────────────
        self.camera_thread = CameraThread(camera_index=0, parent=self)
        self.camera_thread.frame_pair_ready.connect(self.area_kamera.update_frames)
        self.camera_thread.error.connect(self._on_camera_error)

    def _connect_signals(self):
        self.controls_panel.thresholds_changed.connect(
            self.alert_panel.set_thresholds
        )
        self.controls_panel.camera_start_requested.connect(self._on_camera_start)
        self.controls_panel.camera_stop_requested.connect(self._on_camera_stop)
        self.controls_panel.view_mode_changed.connect(self.area_kamera.set_view_mode)

    def _on_camera_start(self):
        self.camera_thread.start_capture()

    def _on_camera_stop(self):
        self.camera_thread.stop_capture()

    def _on_camera_error(self, message: str):
        self.controls_panel._camera_running = False
        self.controls_panel.btn_start.setEnabled(True)
        self.controls_panel.btn_stop.setEnabled(False)
        self.controls_panel.camera_status_label.setText(f"Status: ❌ {message}")
        self.controls_panel.camera_status_label.setStyleSheet("color: #f38ba8;")

    def closeEvent(self, event):
        self.camera_thread.stop_capture()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
