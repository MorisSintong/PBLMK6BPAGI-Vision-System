# Vision/src/obstacle_detector.py

import threading
from typing import Optional

import cv2
import numpy as np

from Vision.src.logging_config import get_logger

logger = get_logger(__name__)


class ObstacleDetector:
    def __init__(
        self,
        max_distance_m=5.0,
        min_distance_m=0.3,
        min_area=800,
    ):
        """
        max_distance_m : jarak maksimal obstacle yang dianggap penting
        min_distance_m : jarak minimal valid agar noise kamera diabaikan
        min_area       : luas minimum objek agar tidak mendeteksi noise kecil
        """
        self.max_distance_m = max_distance_m
        self.min_distance_m = min_distance_m
        self.min_area = min_area

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

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

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
            # Buat threshold berdasarkan nilai min di ROI
            min_val = np.min(roi_depth)
            threshold_m = min_val + 0.30  # Toleransi 30cm

            # Masking area obstacle
            obstacle_mask = (roi_depth <= threshold_m).astype(np.uint8) * 255

            # Cari kontur menggunakan OpenCV
            contours, _ = cv2.findContours(
                obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Ambil kontur terbesar berdasarkan area
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > self.min_area:
                    # Geser koordinat X sesuai offset ROI
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    x += x1

                    distance = float(min_val)

                    # Tentukan Priority & Status
                    if distance < danger_threshold:
                        priority = 1.0
                        global_status = "DANGER"
                    elif distance < warning_threshold:
                        priority = 2.0
                        if global_status != "DANGER":
                            global_status = "WARN"
                    else:
                        priority = 3.0

                    # Translasi zona (L/C/R) ke string untuk GUI
                    zone_str = {"L": "left", "C": "center", "R": "right"}.get(
                        name, "center"
                    )

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

        return rgb_frame, obstacles_list
