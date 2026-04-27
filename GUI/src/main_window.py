import sys
from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QTabWidget
)
from depth_view     import DepthView
from radar_view     import RadarView
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
        main_layout.addWidget(self.area_kamera, stretch=65)

        # ── Bagian Kanan: Tab Widget ──────────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 6px;
                background: #1e1e2e;
            }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #45475a;
            }
        """)

        # ── Tab 1: Controls + Alert ───────────────────────────────────
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(6, 6, 6, 6)
        tab1_layout.setSpacing(6)

        self.controls_panel = ControlsPanel()
        self.alert_panel    = AlertPanel()

        tab1_layout.addWidget(self.controls_panel, stretch=40)
        tab1_layout.addWidget(self.alert_panel,    stretch=60)

        self.tab_widget.addTab(tab1, "🎛️  Controls & Alert")

        # ── Tab 2: Radar ──────────────────────────────────────────────
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(6, 6, 6, 6)
        tab2_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.radar = RadarView()
        self.radar.setFixedSize(280, 280)          # ukuran lebih compact
        tab2_layout.addWidget(self.radar)

        self.tab_widget.addTab(tab2, "📡  Radar")

        main_layout.addWidget(self.tab_widget, stretch=35)

        # ── Kamera thread ────────────────────────────────────────────────
        self.camera_thread = CameraThread(camera_index=0, parent=self)
        self.camera_thread.frame_pair_ready.connect(self.area_kamera.update_frames)
        self.camera_thread.error.connect(self._on_camera_error)

        # ── Sambungkan sinyal ─────────────────────────────────────────
        self._connect_signals()

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
