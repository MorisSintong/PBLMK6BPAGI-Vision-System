# GUI/src/Alert_panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui_config import (
    THRESHOLD_DANGER, THRESHOLD_WARNING,
    ACTION_STOP_DISTANCE, ACTION_SLOWDOWN_DISTANCE,
    ZONE_LEFT, ZONE_CENTER, ZONE_RIGHT,
    ACTION_STOP, ACTION_SLOWDOWN,
    ACTION_TURN_RIGHT, ACTION_TURN_LEFT, ACTION_GO
)
from styles import (
    STATUS_SAFE, STATUS_WARNING, STATUS_DANGER,
    INFOBOX_DEFAULT, INFOBOX_DANGER, INFOBOX_WARNING, INFOBOX_SAFE,
    TEXT_DARK, TEXT_LIGHT, TEXT_WHITE, TEXT_BLUE, TEXT_GREEN
)


# ── Logika Rekomendasi Aksi ───────────────────────────────────────────────────
def get_action(zone: str, distance_m: float) -> str:
    if distance_m is None:
        return "--"
    if distance_m <= ACTION_STOP_DISTANCE:
        return ACTION_STOP
    elif distance_m <= ACTION_SLOWDOWN_DISTANCE:
        if zone == ZONE_CENTER:
            return ACTION_SLOWDOWN
        elif zone == ZONE_LEFT:
            return ACTION_TURN_RIGHT
        elif zone == ZONE_RIGHT:
            return ACTION_TURN_LEFT
        else:
            return ACTION_SLOWDOWN
    else:
        return ACTION_GO


# ── Style jarak dengan warna dinamis ─────────────────────────────────────────
def _distance_style(color: str) -> str:
    return (
        f"color: {color}; background: transparent; "
        f"font-family: 'Segoe UI'; font-size: 32pt; font-weight: bold;"
    )


class AlertPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threshold_warning = THRESHOLD_WARNING
        self.threshold_danger  = THRESHOLD_DANGER
        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------ #
    #  UI Builder                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)
        layout.addStretch(1)

        # ── Info Box ──────────────────────────────────────────────────
        self.info_box = QFrame()
        self.info_box.setObjectName("infoBox")
        self.info_box.setFixedHeight(200)

        box_layout = QVBoxLayout(self.info_box)
        box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_layout.setSpacing(6)

        # ── Nama Objek ────────────────────────────────────────────────
        self.lbl_object_name = QLabel("MENUNGGU...")
        self.lbl_object_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.lbl_object_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_object_name.setStyleSheet(TEXT_LIGHT)

        # ── Jarak ─────────────────────────────────────────────────────
        self.lbl_distance = QLabel("-- m")
        self.lbl_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_distance.setStyleSheet(_distance_style("#ffffff"))

        # ── Zona + Aksi ───────────────────────────────────────────────
        zone_action_row = QHBoxLayout()
        zone_action_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zone_action_row.setSpacing(8)

        self.lbl_zone = QLabel("ZONE: --")
        self.lbl_zone.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_zone.setStyleSheet(TEXT_BLUE)

        separator = QLabel("|")
        separator.setFont(QFont("Segoe UI", 10))
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setStyleSheet("color: #585b70; background: transparent;")

        self.lbl_action = QLabel("--")
        self.lbl_action.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_action.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_action.setStyleSheet(TEXT_GREEN)

        zone_action_row.addWidget(self.lbl_zone)
        zone_action_row.addWidget(separator)
        zone_action_row.addWidget(self.lbl_action)

        # ── Status Bar ────────────────────────────────────────────────
        self.lbl_status = QLabel("SAFE")
        self.lbl_status.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet(STATUS_SAFE)

        # ── Susun semua ke box ────────────────────────────────────────
        box_layout.addWidget(self.lbl_object_name)
        box_layout.addWidget(self.lbl_distance)
        box_layout.addLayout(zone_action_row)
        box_layout.addWidget(self.lbl_status)

        layout.addWidget(self.info_box)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: transparent; }
            #infoBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 15px;
            }
        """)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def set_thresholds(self, warning_m: float, danger_m: float):
        self.threshold_warning = warning_m
        self.threshold_danger  = danger_m

    def update_info(self, object_name: str, distance_m: float, zone: str = ZONE_CENTER):
        """
        Dipanggil oleh CameraThread / FrameProcessor setiap ada data baru.

        Parameters:
            object_name : nama objek, contoh 'Person', 'Vehicle'
            distance_m  : jarak dalam meter
            zone        : posisi objek — 'LEFT', 'CENTER', atau 'RIGHT'
        """

        # ── Reset jika tidak ada data ─────────────────────────────────
        if distance_m is None:
            self.lbl_object_name.setText("MENUNGGU...")
            self.lbl_distance.setText("-- m")
            self.lbl_distance.setStyleSheet(_distance_style("#ffffff"))
            self.lbl_zone.setText("ZONE: --")
            self.lbl_action.setText("--")
            self.lbl_status.setText("SAFE")
            self.lbl_status.setStyleSheet(STATUS_SAFE)
            self.info_box.setStyleSheet(INFOBOX_DEFAULT)
            self.lbl_object_name.setStyleSheet(TEXT_LIGHT)
            self.lbl_zone.setStyleSheet(TEXT_BLUE)
            self.lbl_action.setStyleSheet(TEXT_GREEN)
            return

        # ── Update teks ───────────────────────────────────────────────
        self.lbl_object_name.setText(object_name.upper())
        self.lbl_distance.setText(f"{distance_m:.1f} m")
        self.lbl_zone.setText(f"ZONE: {zone.upper()}")
        self.lbl_action.setText(get_action(zone, distance_m))

        # ── Tentukan status & warna ───────────────────────────────────
        if distance_m <= self.threshold_danger:
            self.info_box.setStyleSheet(INFOBOX_DANGER)
            self.lbl_status.setText("DANGER")
            self.lbl_status.setStyleSheet(STATUS_DANGER)
            self.lbl_distance.setStyleSheet(_distance_style("#1e1e2e"))
            self.lbl_object_name.setStyleSheet(TEXT_DARK)
            self.lbl_zone.setStyleSheet(TEXT_DARK)
            self.lbl_action.setStyleSheet(TEXT_DARK)

        elif distance_m <= self.threshold_warning:
            self.info_box.setStyleSheet(INFOBOX_WARNING)
            self.lbl_status.setText("WARNING")
            self.lbl_status.setStyleSheet(STATUS_WARNING)
            self.lbl_distance.setStyleSheet(_distance_style("#1e1e2e"))
            self.lbl_object_name.setStyleSheet(TEXT_DARK)
            self.lbl_zone.setStyleSheet(TEXT_DARK)
            self.lbl_action.setStyleSheet(TEXT_DARK)

        else:
            self.info_box.setStyleSheet(INFOBOX_SAFE)
            self.lbl_status.setText("SAFE")
            self.lbl_status.setStyleSheet(STATUS_SAFE)
            self.lbl_distance.setStyleSheet(_distance_style("#ffffff"))
            self.lbl_object_name.setStyleSheet(TEXT_LIGHT)
            self.lbl_zone.setStyleSheet(TEXT_BLUE)
            self.lbl_action.setStyleSheet(TEXT_GREEN)