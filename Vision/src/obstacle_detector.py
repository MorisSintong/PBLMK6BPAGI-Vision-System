# Vision/src/obstacle_detector.py

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
    ):
        """
        Mengembalikan frame yang sudah dianotasi dan daftar obstacle (list of dict)
        sesuai kontrak dengan Role 4 (Sensor Fusion).
        """
        if color_frame is None or depth_frame is None:
            return color_frame, []

        annotated_frame = color_frame.copy()
        height, width = depth_frame.shape[:2]

        # Reuse buffer for float32 conversion (avoids ~1.2MB allocation per frame)
        if self._depth_buffer is None or self._depth_buffer.shape != depth_frame.shape:
            self._depth_buffer = np.empty_like(depth_frame, dtype=np.float32)
        np.multiply(depth_frame, depth_scale, out=self._depth_buffer, casting="unsafe")
        depth_meter = self._depth_buffer

        # Pembagian 3 Zona (HUD Ticks instead of full lines)
        zone_width = width // 3
        tick_len = 25
        tick_color = (200, 200, 200)
        
        # Kiri boundary
        cv2.line(annotated_frame, (zone_width, 0), (zone_width, tick_len), tick_color, 2)
        cv2.line(annotated_frame, (zone_width, height - tick_len), (zone_width, height), tick_color, 2)
        
        # Kanan boundary
        cv2.line(annotated_frame, (zone_width * 2, 0), (zone_width * 2, tick_len), tick_color, 2)
        cv2.line(annotated_frame, (zone_width * 2, height - tick_len), (zone_width * 2, height), tick_color, 2)

        # Mask area yang dianggap obstacle dalam rentang jarak
        obstacle_mask = (
            (depth_meter >= self.min_distance_m) & (depth_meter <= self.max_distance_m)
        ).astype(np.uint8) * 255

        # Mengurangi noise dengan optimized native structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        # Cari contour obstacle
        contours, _ = cv2.findContours(
            obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles_list = []
        global_status = "SAFE"
        frame_area = height * width
        max_area = frame_area * self.max_area_ratio

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if area > max_area:
                continue  # Skip scene-wide contours (bukan obstacle diskret)

            x, y, w, h = cv2.boundingRect(contour)

            # Tentukan zone berdasarkan posisi tengah objek
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
                # Menggunakan 5th percentile agar lebih stabil
                distance = float(np.percentile(valid_depth, 5))

                # Hitung prioritas
                priority = round(1 / max(distance, 0.01), 2)

                # Format data SESUAI KONTRAK dengan FrameData
                obstacles_list.append(
                    {
                        "object_class": "obstacle",
                        "bbox": [x, y, w, h],
                        "distance_m": distance,
                        "zone": zone_str,
                        "priority": priority,
                        "area_px": int(area),
                    }
                )

        # Thread-safe assignment
        with self._detections_lock:
            self._last_detections = obstacles_list

        return color_frame, obstacles_list
