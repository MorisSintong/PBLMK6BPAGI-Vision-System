# GUI/src/Alert_panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QColor, QPalette, QFont
import collections


class AlertLevel:
    SAFE    = "SAFE"
    WARNING = "WARNING"
    DANGER  = "DANGER"


class AlertPanel(QWidget):
    """
    Panel yang menampilkan status alert berdasarkan jarak obstacle.
    Threshold WARNING dan DANGER bisa di-set dari luar (oleh ControlPanel).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Default threshold (meter) — bisa diubah via set_thresholds()
        self.threshold_warning = 1.0   # meter
        self.threshold_danger  = 0.5   # meter

        # Simpan history alert (maks 50 entri)
        self.alert_history = collections.deque(maxlen=50)

        self._build_ui()
        self._apply_style()
        self.update_alert(AlertLevel.SAFE, None)

    # ------------------------------------------------------------------ #
    #  UI Builder                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ── Judul ──────────────────────────────────────────────────────
        title = QLabel("🚨  Alert Panel")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # ── Status utama ───────────────────────────────────────────────
        self.status_frame = QFrame()
        self.status_frame.setFixedHeight(80)
        self.status_frame.setFrameShape(QFrame.Shape.StyledPanel)

        status_inner = QVBoxLayout(self.status_frame)
        self.status_label = QLabel("SAFE")
        self.status_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distance_label = QLabel("Jarak: --")
        self.distance_label.setFont(QFont("Segoe UI", 11))
        self.distance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_inner.addWidget(self.status_label)
        status_inner.addWidget(self.distance_label)
        main_layout.addWidget(self.status_frame)

        # ── Threshold info ─────────────────────────────────────────────
        self.threshold_info = QLabel()
        self.threshold_info.setFont(QFont("Segoe UI", 9))
        self.threshold_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refresh_threshold_label()
        main_layout.addWidget(self.threshold_info)

        # ── Separator ─────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(sep)

        # ── Log history ───────────────────────────────────────────────
        log_title = QLabel("📋  Log Alert")
        log_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        main_layout.addWidget(log_title)

        self.log_area = QScrollArea()
        self.log_area.setWidgetResizable(True)
        self.log_container = QWidget()
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log_layout.setSpacing(2)
        self.log_area.setWidget(self.log_container)
        main_layout.addWidget(self.log_area)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; color: #cdd6f4; }
            QScrollArea { border: none; }
            QFrame { border-radius: 8px; }
        """)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def set_thresholds(self, warning_m: float, danger_m: float):
        """Dipanggil dari ControlPanel saat user mengubah threshold."""
        self.threshold_warning = warning_m
        self.threshold_danger  = danger_m
        self._refresh_threshold_label()

    def process_distance(self, distance_m: float):
        """
        Terima jarak (meter) dari kamera/vision thread,
        tentukan level alert, lalu update UI.
        """
        if distance_m <= self.threshold_danger:
            level = AlertLevel.DANGER
        elif distance_m <= self.threshold_warning:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.SAFE

        self.update_alert(level, distance_m)

    def update_alert(self, level: str, distance_m):
        """Update tampilan status & tambah log entry."""
        colors = {
            AlertLevel.SAFE:    ("#a6e3a1", "#1e1e2e", "#2a3b2a"),
            AlertLevel.WARNING: ("#f9e2af", "#1e1e2e", "#3b3520"),
            AlertLevel.DANGER:  ("#f38ba8", "#1e1e2e", "#3b1f25"),
        }
        fg, _, bg = colors[level]

        dist_text = f"{distance_m:.2f} m" if distance_m is not None else "--"
        self.status_label.setText(level)
        self.distance_label.setText(f"Jarak: {dist_text}")
        self.status_frame.setStyleSheet(
            f"background-color: {bg}; border: 2px solid {fg}; border-radius: 8px;"
        )
        self.status_label.setStyleSheet(f"color: {fg};")

        # Tambah ke log
        if distance_m is not None:
            self._add_log_entry(level, dist_text, fg)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _refresh_threshold_label(self):
        self.threshold_info.setText(
            f"⚠ WARNING < {self.threshold_warning:.1f} m  |  "
            f"🔴 DANGER < {self.threshold_danger:.1f} m"
        )

    def _add_log_entry(self, level: str, dist_text: str, color: str):
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        entry = QLabel(f"[{timestamp}]  {level:<8}  {dist_text}")
        entry.setFont(QFont("Courier New", 9))
        entry.setStyleSheet(f"color: {color}; padding: 1px 4px;")
        self.log_layout.addWidget(entry)

        # Auto-scroll ke bawah
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )