"""
Vision/inc/detection_config.py — Centralized detection thresholds.

Single source of truth for distance thresholds used across the pipeline.
Values are mutated by GUI sliders via set_danger_distance(); the pipeline
re-reads them per frame so changes apply immediately.

Thresholds (metres):
  min_distance     — closer than this is treated as camera noise / too close
  max_distance     — farther than this is ignored
  danger_distance  — closer than this triggers DANGER status + STOP action
  warning_distance — closer than this triggers WARN status + SLOWDOWN
"""

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)


class DetectionConfig:
    def __init__(self):
        # Jarak threshold dalam meter
        self.min_distance = 0.3   # jarak minimum (terlalu dekat, diabaikan)
        self.max_distance = 5.0   # jarak maksimum yang dipantau
        self.danger_distance = 1.5  # jarak bahaya = obstacle terdeteksi
        self.warning_distance = 3.0  # jarak warning = obstacle mendekati

    def set_danger_distance(self, distance: float):
        """Ubah jarak bahaya, dipanggil dari GUI"""
        if self.min_distance <= distance <= self.max_distance:
            self.danger_distance = distance
            logger.info(f"Danger distance diubah ke: {distance} meter")
        else:
            logger.error(f"Jarak harus antara {self.min_distance} - {self.max_distance} meter")

    def get_danger_distance(self):
        return self.danger_distance