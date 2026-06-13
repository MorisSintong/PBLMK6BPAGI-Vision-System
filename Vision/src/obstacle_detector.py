# Vision/src/obstacle_detector.py

import cv2
import numpy as np


class ObstacleDetector:
    def __init__(
        self,
        max_distance_m=5.0,
        min_distance_m=0.3,
        min_area=800,
        roi_ratio=0.7,
    ):
        """
        max_distance_m : jarak maksimal obstacle yang dianggap penting
        min_distance_m : jarak minimal valid agar noise kamera diabaikan
        min_area       : luas minimum objek agar tidak mendeteksi noise kecil
        roi_ratio      : rasio area ROI (region of interest)
        """
        self.max_distance_m = max_distance_m
        self.min_distance_m = min_distance_m
        self.min_area = min_area
        self.roi_ratio = roi_ratio
        self.last_detections: list = []

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
        depth_meter = depth_frame.astype(np.float32) * depth_scale

        # Pembagian 3 Zona
        zone_width = width // 3
        cv2.line(
            annotated_frame, (zone_width, 0), (zone_width, height), (255, 255, 255), 2
        )
        cv2.line(
            annotated_frame,
            (zone_width * 2, 0),
            (zone_width * 2, height),
            (255, 255, 255),
            2,
        )

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
                priority = round(1 / distance, 2)

                # Tentukan warna dan status
                if distance < danger_threshold:
                    color = (0, 0, 255)  # Merah (DANGER)
                    global_status = "DANGER"
                elif distance < warning_threshold:
                    color = (0, 255, 255)  # Kuning (WARN)
                    if global_status != "DANGER":
                        global_status = "WARN"
                else:
                    color = (0, 255, 0)  # Hijau (SAFE)

                label = f"{zone_str.upper()}: {distance:.2f} m | P:{priority}"

                # Gambar Bounding Box dan Teks
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
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

        # Tampilkan Status Global di Pojok
        if global_status == "DANGER":
            status_color = (0, 0, 255)
        elif global_status == "WARN":
            status_color = (0, 255, 255)
        else:
            status_color = (0, 255, 0)

        cv2.putText(
            annotated_frame,
            f"STATUS: {global_status}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        self.last_detections = obstacles_list
        return annotated_frame, obstacles_list
