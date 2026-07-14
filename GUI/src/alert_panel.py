# GUI/src/Alert_panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from GUI.inc.ui_config import (
    THRESHOLD_DANGER, THRESHOLD_WARNING,
    ACTION_STOP_DISTANCE, ACTION_SLOWDOWN_DISTANCE,
    ZONE_LEFT, ZONE_CENTER, ZONE_RIGHT,
    ACTION_STOP, ACTION_SLOWDOWN,
    ACTION_TURN_RIGHT, ACTION_TURN_LEFT, ACTION_GO
)
from GUI.inc.styles import (
    STATUS_SAFE, STATUS_WARNING, STATUS_DANGER,
    INFOBOX_DEFAULT, INFOBOX_DANGER, INFOBOX_WARNING, INFOBOX_SAFE,
    TEXT_DARK, TEXT_LIGHT, TEXT_WHITE, TEXT_BLUE, TEXT_GREEN
)


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


def _distance_style(color: str) -> str:
    return (
        f"color: {color}; background: transparent; "
        f"font-family: 'Segoe UI'; font-size: 32pt; font-weight: bold;"
    )


# Pre-computed style strings (avoid f-string allocation per frame)
_STYLE_DANGER = {
    "infobox": INFOBOX_DANGER,
    "status": STATUS_DANGER,
    "distance": _distance_style("#1e1e2e"),
    "object": TEXT_DARK,
    "zone": TEXT_DARK,
    "action": TEXT_DARK,
}
_STYLE_WARNING = {
    "infobox": INFOBOX_WARNING,
    "status": STATUS_WARNING,
    "distance": _distance_style("#1e1e2e"),
    "object": TEXT_DARK,
    "zone": TEXT_DARK,
    "action": TEXT_DARK,
}
_STYLE_SAFE = {
    "infobox": INFOBOX_SAFE,
    "status": STATUS_SAFE,
    "distance": _distance_style("#ffffff"),
    "object": TEXT_LIGHT,
    "zone": TEXT_BLUE,
    "action": TEXT_GREEN,
}
_STYLE_WAITING = {
    "infobox": INFOBOX_DEFAULT,
    "status": STATUS_SAFE,
    "distance": _distance_style("#ffffff"),
    "object": TEXT_LIGHT,
    "zone": TEXT_BLUE,
    "action": TEXT_GREEN,
}


class AlertPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threshold_warning = THRESHOLD_WARNING
        self.threshold_danger  = THRESHOLD_DANGER
        self._prev_status = "init"
        # Navigation state guards (avoid per-frame setStyleSheet + setText churn)
        self._prev_nav_status = ""
        self._prev_nav_steer = 0.0
        self._prev_nav_speed = -1.0
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)
        layout.addStretch(1)

        self.info_box = QFrame()
        self.info_box.setObjectName("infoBox")
        self.info_box.setFixedHeight(200)

        box_layout = QVBoxLayout(self.info_box)
        box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_layout.setSpacing(6)

        self.lbl_object_name = QLabel("MENUNGGU...")
        self.lbl_object_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.lbl_object_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_object_name.setStyleSheet(TEXT_LIGHT)

        self.lbl_distance = QLabel("-- m")
        self.lbl_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_distance.setStyleSheet(_distance_style("#ffffff"))

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

        self.lbl_status = QLabel("SAFE")
        self.lbl_status.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet(STATUS_SAFE)

        box_layout.addWidget(self.lbl_object_name)
        box_layout.addWidget(self.lbl_distance)
        box_layout.addLayout(zone_action_row)
        box_layout.addWidget(self.lbl_status)

        layout.addWidget(self.info_box)

        # ── Navigation Panel ───────────────────────────────────────
        self.nav_box = QFrame()
        self.nav_box.setObjectName("navBox")
        nav_layout = QVBoxLayout(self.nav_box)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.setSpacing(4)

        self.lbl_nav_title = QLabel("NAVIGATION")
        self.lbl_nav_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_nav_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_title.setStyleSheet("color: #585b70; background: transparent;")

        self.lbl_nav_status = QLabel("IDLE")
        self.lbl_nav_status.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_nav_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_status.setStyleSheet("color: #50f050; background: transparent;")

        self.lbl_nav_steer = QLabel("STEER: 0 deg")
        self.lbl_nav_steer.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_nav_steer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_steer.setStyleSheet("color: #89b4fa; background: transparent;")

        self.lbl_nav_speed = QLabel("SPEED: 0%")
        self.lbl_nav_speed.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_nav_speed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_speed.setStyleSheet("color: #89b4fa; background: transparent;")

        nav_layout.addWidget(self.lbl_nav_title)
        nav_layout.addWidget(self.lbl_nav_status)
        nav_layout.addWidget(self.lbl_nav_steer)
        nav_layout.addWidget(self.lbl_nav_speed)

        layout.addWidget(self.nav_box)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: transparent; }
            #infoBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 15px;
            }
            #navBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 15px;
            }
        """)

    def set_thresholds(self, warning_m: float, danger_m: float):
        self.threshold_warning = warning_m
        self.threshold_danger  = danger_m

    def _apply_status_style(self, style: dict):
        """Apply a full set of stylesheets. Only call when status changes."""
        self.info_box.setStyleSheet(style["infobox"])
        self.lbl_status.setStyleSheet(style["status"])
        self.lbl_distance.setStyleSheet(style["distance"])
        self.lbl_object_name.setStyleSheet(style["object"])
        self.lbl_zone.setStyleSheet(style["zone"])
        self.lbl_action.setStyleSheet(style["action"])

    def update_info(self, object_name: str, distance_m: float, zone: str = ZONE_CENTER):
        if distance_m is None:
            if self._prev_status != "waiting":
                self._prev_status = "waiting"
                self._apply_status_style(_STYLE_WAITING)
            if self.lbl_object_name.text() != "MENUNGGU...":
                self.lbl_object_name.setText("MENUNGGU...")
            if self.lbl_distance.text() != "-- m":
                self.lbl_distance.setText("-- m")
            if self.lbl_zone.text() != "ZONE: --":
                self.lbl_zone.setText("ZONE: --")
            if self.lbl_action.text() != "--":
                self.lbl_action.setText("--")
            if self.lbl_status.text() != "SAFE":
                self.lbl_status.setText("SAFE")
            return

        obj_text = object_name.upper()
        dist_text = f"{distance_m:.1f} m"
        zone_text = f"ZONE: {zone.upper()}"
        action_text = get_action(zone, distance_m)

        if self.lbl_object_name.text() != obj_text:
            self.lbl_object_name.setText(obj_text)
        if self.lbl_distance.text() != dist_text:
            self.lbl_distance.setText(dist_text)
        if self.lbl_zone.text() != zone_text:
            self.lbl_zone.setText(zone_text)
        if self.lbl_action.text() != action_text:
            self.lbl_action.setText(action_text)

        if distance_m <= self.threshold_danger:
            if self._prev_status != "danger":
                self._prev_status = "danger"
                self._apply_status_style(_STYLE_DANGER)
            if self.lbl_status.text() != "DANGER":
                self.lbl_status.setText("DANGER")
        elif distance_m <= self.threshold_warning:
            if self._prev_status != "warning":
                self._prev_status = "warning"
                self._apply_status_style(_STYLE_WARNING)
            if self.lbl_status.text() != "WARNING":
                self.lbl_status.setText("WARNING")
        else:
            if self._prev_status != "safe":
                self._prev_status = "safe"
                self._apply_status_style(_STYLE_SAFE)
            if self.lbl_status.text() != "SAFE":
                self.lbl_status.setText("SAFE")

    def update_navigation(self, nav_data: dict):
        """Update navigation display from NavigationStage output."""
        if not nav_data:
            return

        status = nav_data.get("status", "IDLE")
        steer = nav_data.get("steering_angle_deg", 0.0)
        speed = nav_data.get("speed", 0.0)

        # Guard: skip if nothing meaningful changed (avoids setText+setStyleSheet per frame)
        if (status == self._prev_nav_status
                and abs(steer - self._prev_nav_steer) < 0.5
                and abs(speed - self._prev_nav_speed) < 0.01):
            return
        self._prev_nav_status = status
        self._prev_nav_steer = steer
        self._prev_nav_speed = speed

        if self.lbl_nav_status.text() != status:
            self.lbl_nav_status.setText(status)
        steer_text = f"STEER: {steer:+.0f} deg"
        if self.lbl_nav_steer.text() != steer_text:
            self.lbl_nav_steer.setText(steer_text)
        speed_text = f"SPEED: {speed:.0%}"
        if self.lbl_nav_speed.text() != speed_text:
            self.lbl_nav_speed.setText(speed_text)

        if status == "STOPPED" or status == "BLOCKED":
            self.lbl_nav_status.setStyleSheet("color: #f38ba8; background: transparent;")
        elif status == "AVOIDING":
            self.lbl_nav_status.setStyleSheet("color: #fab387; background: transparent;")
        else:
            self.lbl_nav_status.setStyleSheet("color: #50f050; background: transparent;")
