"""
Vision/src/obstacle_detector.py — Depth-based obstacle detection.

Role 3 (Long) — Depth Specialist.

Detects obstacle regions in a depth frame using a distance mask + morphological
filtering + contour extraction. Each obstacle is classified into a horizontal
zone (left / center / right) and assigned a 5th-percentile distance to suppress
outliers from reflective surfaces.

Performance:
  - Reusable float32 buffer (no allocation per frame)
  - Cached morphological kernel
  - np.partition for O(n) percentile instead of O(n log n)

Thread safety:
  - last_detections is guarded by a lock for cross-thread read/write.
"""

import threading
from typing import Optional

import cv2
import numpy as np

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)


class ObstacleDetector:
    def __init__(
        self,
        max_distance_m=5.0,
        min_distance_m=0.3,
        min_area=800,
        max_area_ratio=0.40,
    ):
        """
        max_distance_m  : jarak maksimal obstacle yang dianggap penting
        min_distance_m  : jarak minimal valid agar noise kamera diabaikan
        min_area        : luas minimum objek agar tidak mendeteksi noise kecil
        max_area_ratio  : rasio maks area kontur terhadap frame (0.0-1.0).
                          Kontur lebih besar dari ini dianggap "seluruh scene",
                          bukan obstacle diskret, dan akan diskip.
        """
        self.max_distance_m = max_distance_m
        self.min_distance_m = min_distance_m
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

        # Thread safety for last_detections
        self._detections_lock = threading.Lock()
        self._last_detections: list = []

        # Reusable buffer for float32 depth conversion (avoids allocation per frame)
        self._depth_buffer: Optional[np.ndarray] = None

        # Cache morphological kernel (avoids per-frame allocation)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    @property
    def last_detections(self) -> list:
        """Thread-safe getter for last detections."""
        with self._detections_lock:
            return self._last_detections.copy()

    @last_detections.setter
    def last_detections(self, value: list) -> None:
        """Thread-safe setter for last detections."""
        with self._detections_lock:
            self._last_detections = value

    def detect(
        self,
        color_frame,
        depth_frame,
        depth_scale=0.001,
        danger_threshold=1.0,
        warning_threshold=3.0,
        depth_m=None,
    ):
        """
        Mengembalikan frame asli dan daftar obstacle (list of dict)
        sesuai kontrak dengan Role 4 (Sensor Fusion).

        Args:
            depth_m: Pre-computed float32 depth in meters (optional, avoids
                redundant conversion). If None, computed from depth_frame.
        """
        if color_frame is None or depth_frame is None:
            return color_frame, []

        height, width = depth_frame.shape[:2]

        # Use pre-computed depth_m if provided, else convert (legacy fallback)
        if depth_m is not None:
            depth_meter = depth_m
        else:
            # Reuse buffer for float32 conversion (avoids ~1.2MB allocation per frame)
            if self._depth_buffer is None or self._depth_buffer.shape != depth_frame.shape:
                self._depth_buffer = np.empty_like(depth_frame, dtype=np.float32)
            np.multiply(depth_frame, depth_scale, out=self._depth_buffer, casting="unsafe")
            depth_meter = self._depth_buffer

        # Mask area yang dianggap obstacle dalam rentang jarak
        obstacle_mask = (
            (depth_meter >= self.min_distance_m) & (depth_meter <= self.max_distance_m)
        ).astype(np.uint8) * 255

        # Mengurangi noise dengan optimized native structure (cached kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, self._morph_kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # Cari contour obstacle
        contours, _ = cv2.findContours(
            obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles_list = []
        frame_area = height * width
        max_area = frame_area * self.max_area_ratio
        zone_width = width // 3

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + (w // 2)
            if center_x < zone_width:
                zone_str = "left"
            elif center_x < zone_width * 2:
                zone_str = "center"
            else:
                zone_str = "right"

            object_depth = depth_meter[y : y + h, x : x + w]
            valid_depth = object_depth[
                (object_depth >= self.min_distance_m)
                & (object_depth <= self.max_distance_m)
            ]

            if valid_depth.size > 0:
                # np.partition is O(n) vs np.percentile O(n log n)
                k = max(0, int(len(valid_depth) * 0.05))
                distance = float(np.partition(valid_depth, k)[k])

                obstacles_list.append(
                    {
                        "object_class": "obstacle",
                        "bbox": [x, y, w, h],
                        "distance_m": distance,
                        "zone": zone_str,
                        "area_px": int(area),
                    }
                )

        with self._detections_lock:
            self._last_detections = obstacles_list

        return color_frame, obstacles_list
