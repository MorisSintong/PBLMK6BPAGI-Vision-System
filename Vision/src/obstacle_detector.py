# Vision/src/obstacle_detector.py

import threading
from typing import Optional

import cv2
import numpy as np

from logging_config import get_logger

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

        # Mengurangi noise
        kernel = np.ones((5, 5), np.uint8)
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
                # Menggunakan 5th percentile agar lebih stabil
                distance = float(np.percentile(valid_depth, 5))

                # Hitung prioritas
                priority = round(1 / max(distance, 0.01), 2)

                # Tentukan warna dan status (Warna HUD Premium)
                if distance < danger_threshold:
                    color = (60, 60, 255)  # Soft Red
                    global_status = "DANGER"
                elif distance < warning_threshold:
                    color = (0, 165, 255)  # Amber
                    if global_status != "DANGER":
                        global_status = "WARN"
                else:
                    color = (50, 205, 50)  # Lime Green

                # Label yang lebih minimalis: [L] 1.50m
                label = f"[{zone_str.upper()[0]}] {distance:.2f}m"

                # Gambar Bounding Box (HUD Corner Brackets)
                bracket_len = max(5, min(w, h, 40) // 4)
                thick = 3
                # Top Left
                cv2.line(annotated_frame, (x, y), (x + bracket_len, y), color, thick)
                cv2.line(annotated_frame, (x, y), (x, y + bracket_len), color, thick)
                # Top Right
                cv2.line(annotated_frame, (x + w, y), (x + w - bracket_len, y), color, thick)
                cv2.line(annotated_frame, (x + w, y), (x + w, y + bracket_len), color, thick)
                # Bottom Left
                cv2.line(annotated_frame, (x, y + h), (x + bracket_len, y + h), color, thick)
                cv2.line(annotated_frame, (x, y + h), (x, y + h - bracket_len), color, thick)
                # Bottom Right
                cv2.line(annotated_frame, (x + w, y + h), (x + w - bracket_len, y + h), color, thick)
                cv2.line(annotated_frame, (x + w, y + h), (x + w, y + h - bracket_len), color, thick)

                # Teks dengan Background Plate
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
                text_y = y - 8
                
                # Gambar background (dark slate)
                cv2.rectangle(annotated_frame, (x, text_y - text_h - 4), (x + text_w, text_y + 4), (30, 30, 30), -1)
                
                # Teks dengan Anti-Aliasing
                cv2.putText(
                    annotated_frame,
                    label,
                    (x, text_y),
                    font,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
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

        # Tampilkan Status Global di Pojok dengan HUD Panel
        if global_status == "DANGER":
            status_color = (60, 60, 255)
        elif global_status == "WARN":
            status_color = (0, 165, 255)
        else:
            status_color = (50, 205, 50)

        status_text = f"SYS: {global_status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(status_text, font, scale, thickness)
        sx, sy = 25, 45
        
        # Background panel + outline
        cv2.rectangle(annotated_frame, (sx - 8, sy - text_h - 10), (sx + text_w + 8, sy + 10), (30, 30, 30), -1)
        cv2.rectangle(annotated_frame, (sx - 8, sy - text_h - 10), (sx + text_w + 8, sy + 10), status_color, 1)

        cv2.putText(
            annotated_frame,
            status_text,
            (sx, sy),
            font,
            scale,
            status_color,
            thickness,
            cv2.LINE_AA,
        )

        # Thread-safe assignment
        with self._detections_lock:
            self._last_detections = obstacles_list

        return annotated_frame, obstacles_list
