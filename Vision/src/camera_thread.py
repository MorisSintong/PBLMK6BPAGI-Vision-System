import os
import sys
import threading
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

try:
    from ui_config import DEPTH_MAX_M, DEPTH_MIN_M
except ImportError:
    DEPTH_MAX_M = 5.0
    DEPTH_MIN_M = 0.3

if TYPE_CHECKING:
    from frame_processor import FrameProcessor

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None


class CameraThread(QThread):
    frame_pair_ready = pyqtSignal(object, object)
    distance_info_ready = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(
        self, camera_index=0, parent=None, processor: "Optional[FrameProcessor]" = None
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._capture = None
        self._pipeline = None
        self._align = None
        
        # Konfigurasi Jarak
        self._depth_scale = 0.001
        self._depth_min_m = DEPTH_MIN_M
        self._depth_max_m = DEPTH_MAX_M
        
        # Threshold Zona Bahaya
        self.danger_threshold = 1.0  
        self.warning_threshold = 3.0 
        
        self._threshold_lock = threading.Lock()
        self._processor = processor

    def start_capture(self):
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_capture(self):
        self._running = False
        if self.isRunning():
            self.wait(1000)

    def run(self):
        if self._start_realsense():
            self._run_realsense_loop()
        else:
            self._capture = self._open_camera()
            if self._capture is None:
                self.error.emit("Kamera gagal dibuka. Tutup aplikasi lain.")
                self._release_resources()
                return
            self._run_webcam_loop()

        self._release_resources()

    def _start_realsense(self):
        if rs is None:
            return False

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            profile = self._pipeline.start(config)
        except RuntimeError:
            self._pipeline = None
            return False

        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # --- INISIALISASI SEMUA FILTER (TERMASUK DECIMATION) ---
        self._decimation_filter = rs.decimation_filter()
        self._decimation_filter.set_option(rs.option.filter_magnitude, 2)
        
        self._spatial_filter = rs.spatial_filter()
        self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        self._temporal_filter = rs.temporal_filter()
        self._hole_filling_filter = rs.hole_filling_filter()
        # -------------------------------------------------------

        return True

    def _run_realsense_loop(self):
        read_failures = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                read_failures += 1
                if read_failures >= 10:
                    self.error.emit("Gagal membaca stream RealSense.")
                    break
                self.msleep(30)
                continue

            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # --- APLIKASI FILTER SECARA BERURUTAN ---
            depth_frame = self._decimation_filter.process(depth_frame)
            depth_frame = self._spatial_filter.process(depth_frame)
            depth_frame = self._temporal_filter.process(depth_frame)
            depth_frame = self._hole_filling_filter.process(depth_frame)

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Kembalikan ukuran depth map ke 640x480 karena decimation mengecilkannya
            if depth_raw.shape != color_bgr.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Eksekusi Deteksi Multi-Zona Langsung di Sini
            annotated_frame, depth_colormap, global_status, center_dist = self._process_multi_zone(color_bgr, depth_raw)

            # Emit hasil ke GUI / Console
            rgb_pixmap = self._bgr_to_qpixmap(annotated_frame)
            depth_pixmap = self._bgr_to_qpixmap(depth_colormap)
            
            self.frame_pair_ready.emit(rgb_pixmap, depth_pixmap)
            self.distance_info_ready.emit(global_status, center_dist)

    def _process_multi_zone(self, color_bgr, depth_raw):
        annotated_frame = color_bgr.copy()
        height, width = depth_raw.shape[:2]
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        
        # Buat colormap hitam kosong
        depth_colormap = np.zeros((height, width, 3), dtype=np.uint8)
        
        with self._threshold_lock:
            valid_mask = (depth_m >= self._depth_min_m) & (depth_m <= self._depth_max_m)
            
        danger_mask = valid_mask & (depth_m < self.danger_threshold)
        warning_mask = valid_mask & (depth_m >= self.danger_threshold) & (depth_m < self.warning_threshold)
        safe_mask = valid_mask & (depth_m >= self.warning_threshold)
        
        # Warnai sesuai zona bahaya (Merah, Kuning, Hijau dalam BGR)
        depth_colormap[danger_mask] = (0, 0, 255)
        depth_colormap[warning_mask] = (0, 255, 255)
        depth_colormap[safe_mask] = (0, 255, 0)
        
        # Pembagian 3 Zona Lebar
        zone_width = width // 3
        zones_coords = {
            "KIRI": (0, zone_width),
            "TENGAH": (zone_width, zone_width * 2),
            "KANAN": (zone_width * 2, width)
        }
        
        center_dist = None
        global_status = "SAFE"
        
        for zone_name, (x_start, x_end) in zones_coords.items():
            # Gambar garis vertikal pembatas zona
            cv2.line(annotated_frame, (x_end, 0), (x_end, height), (255, 255, 255), 2)
            
            zone_depth = depth_m[:, x_start:x_end]
            zone_valid = valid_mask[:, x_start:x_end]
            
            if np.any(zone_valid):
                valid_pixels = zone_depth[zone_valid]
                if valid_pixels.size > 50:
                    min_dist = float(np.percentile(valid_pixels, 5))
                    
                    if min_dist < self.danger_threshold:
                        text_color, status = (0, 0, 255), "DANGER"
                        global_status = "DANGER"
                    elif min_dist < self.warning_threshold:
                        text_color, status = (0, 255, 255), "WARN"
                        if global_status != "DANGER": global_status = "WARN"
                    else:
                        text_color, status = (0, 255, 0), "SAFE"
                        
                    # Tulis jarak dan status per zona
                    cv2.putText(annotated_frame, f"{zone_name}: {min_dist:.2f}m", (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    cv2.putText(annotated_frame, status, (x_start + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    if zone_name == "TENGAH":
                        center_dist = min_dist
                else:
                    cv2.putText(annotated_frame, f"{zone_name}: CLEAR", (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            else:
                cv2.putText(annotated_frame, f"{zone_name}: CLEAR", (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
        return annotated_frame, depth_colormap, global_status, center_dist

    def _run_webcam_loop(self):
        # (Sisa loop webcam legacy)
        while self._running:
            ok, frame_bgr = self._capture.read()
            if not ok: continue
            rgb_pixmap = self._bgr_to_qpixmap(frame_bgr)
            self.frame_pair_ready.emit(rgb_pixmap, None)
            self.distance_info_ready.emit("Depth Tidak Tersedia", None)

    def _open_camera(self):
        if os.name == "nt": backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else: backend_candidates = [cv2.CAP_ANY]
        for backend in backend_candidates:
            capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened(): return capture
        return None

    def _bgr_to_qpixmap(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _release_resources(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


# =====================================================================
# BLOK TESTING STANDALONE
# =====================================================================
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout

    app = QApplication(sys.argv)

    test_window = QWidget()
    test_window.setWindowTitle("Test Standalone Camera Thread (3 Zones Unified)")
    test_window.resize(1280, 480)
    layout = QHBoxLayout()

    label_rgb = QLabel("Menunggu kamera RGB...")
    label_depth = QLabel("Menunggu kamera Depth...")
    layout.addWidget(label_rgb)
    layout.addWidget(label_depth)
    test_window.setLayout(layout)

    def update_ui(rgb_pixmap, depth_pixmap):
        if rgb_pixmap is not None: label_rgb.setPixmap(rgb_pixmap)
        if depth_pixmap is not None: label_depth.setPixmap(depth_pixmap)

    def print_status(label, dist):
        jarak_str = f"{dist:.2f}" if dist else "N/A"
        print(f"Status Global: {label} | Jarak Tengah: {jarak_str} meter", end="\r")

    tester_thread = CameraThread()
    tester_thread.frame_pair_ready.connect(update_ui)
    tester_thread.distance_info_ready.connect(print_status)

    test_window.show()
    tester_thread.start_capture()
    sys.exit(app.exec())