# Vision/src/obstacle_detector.py

import cv2
import numpy as np


class ObstacleDetector:
    def __init__(
        self,
        max_distance_m=2.0,
        min_distance_m=0.04,
        min_area=800,
        roi_ratio=0.7
    ):
        """
        max_distance_m : jarak maksimal obstacle yang dianggap penting
        min_distance_m : jarak minimal valid agar noise kamera diabaikan
        min_area       : luas minimum objek agar tidak mendeteksi noise kecil
        roi_ratio      : area tengah kamera yang dipakai untuk deteksi
        """
        self.max_distance_m = max_distance_m
        self.min_distance_m = min_distance_m
        self.min_area = min_area
        self.roi_ratio = roi_ratio

    def detect(self, color_frame, depth_frame, depth_scale=0.001):
        """
        color_frame : frame RGB/BGR dari kamera
        depth_frame : frame depth dari kamera
        depth_scale : konversi depth ke meter
                      biasanya 0.001 jika depth dalam milimeter
        """

        if color_frame is None or depth_frame is None:
            return color_frame, False, None

        annotated_frame = color_frame.copy()

        height, width = depth_frame.shape[:2]

        # Membuat Region of Interest (ROI) di bagian tengah kamera
        roi_w = int(width * self.roi_ratio)
        roi_h = int(height * self.roi_ratio)

        x1 = (width - roi_w) // 2
        y1 = (height - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        depth_roi = depth_frame[y1:y2, x1:x2]

        # Konversi depth menjadi meter
        depth_meter = depth_roi.astype(np.float32) * depth_scale

        # Mask area yang dianggap obstacle
        obstacle_mask = (
            (depth_meter >= self.min_distance_m) &
            (depth_meter <= self.max_distance_m)
        ).astype(np.uint8) * 255

        # Mengurangi noise
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        # Cari contour obstacle
        contours, _ = cv2.findContours(
            obstacle_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        obstacle_detected = False
        closest_distance = None

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            obstacle_detected = True

            x, y, w, h = cv2.boundingRect(contour)

            # Posisi bounding box dikembalikan ke koordinat frame asli
            real_x = x + x1
            real_y = y + y1

            object_depth = depth_meter[y:y+h, x:x+w]
            valid_depth = object_depth[
                (object_depth >= self.min_distance_m) &
                (object_depth <= self.max_distance_m)
            ]

            if valid_depth.size > 0:
                distance = float(np.min(valid_depth))

                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance

                label = f"Obstacle: {distance:.2f} m"
            else:
                label = "Obstacle"

            cv2.rectangle(
                annotated_frame,
                (real_x, real_y),
                (real_x + w, real_y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (real_x, real_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Gambar area ROI
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        if obstacle_detected:
            status_text = "OBSTACLE DETECTED"
            status_color = (0, 0, 255)
        else:
            status_text = "CLEAR"
            status_color = (0, 255, 0)

        cv2.putText(
            annotated_frame,
            status_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        return annotated_frame, obstacle_detected, closest_distance