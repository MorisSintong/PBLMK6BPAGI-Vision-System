import cv2
import os
import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from ui_config import DEPTH_MAX_M, DEPTH_MIN_M

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class CameraThread(QThread):
    frame_pair_ready = pyqtSignal(object, object)
    distance_info_ready = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._capture = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._depth_min_m = DEPTH_MIN_M
        self._depth_max_m = DEPTH_MAX_M
        self._threshold_lock = threading.Lock()

    def start_capture(self):
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_capture(self):
        self._running = False
        if self.isRunning():
            self.wait(1000)

    def set_depth_thresholds(self, depth_min_m: float, depth_max_m: float):
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return

        with self._threshold_lock:
            self._depth_min_m = depth_min_m
            self._depth_max_m = depth_max_m

    def run(self):
        if self._start_realsense():
            self._run_realsense_loop()
        else:
            self._capture = self._open_camera()
            if self._capture is None:
                self.error.emit("Kamera gagal dibuka. Tutup aplikasi lain yang sedang memakai kamera.")
                self._release_resources()
                return
            self._run_webcam_loop()

        self._release_resources()

    def _run_realsense_loop(self):
        read_failures = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                read_failures += 1
                if read_failures >= 10:
                    self.error.emit("Gagal membaca stream RealSense secara stabil.")
                    break
                self.msleep(30)
                continue

            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                read_failures += 1
                if read_failures >= 10:
                    self.error.emit("Frame RealSense tidak lengkap.")
                    break
                self.msleep(30)
                continue

            read_failures = 0
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_colormap = self._depth_to_colormap(depth_raw)
            distance_m = self._estimate_distance(depth_raw)

            rgb_pixmap = self._bgr_to_qpixmap(color_bgr)
            depth_pixmap = self._bgr_to_qpixmap(depth_colormap)
            self.frame_pair_ready.emit(rgb_pixmap, depth_pixmap)
            self.distance_info_ready.emit("Objek Terdeteksi", distance_m)

    def _run_webcam_loop(self):
        read_failures = 0
        while self._running:
            ok, frame_bgr = self._capture.read()
            if not ok:
                read_failures += 1
                if read_failures >= 10:
                    self.error.emit("Gagal membaca frame kamera secara stabil.")
                    break
                self.msleep(30)
                continue

            read_failures = 0
            rgb_pixmap = self._bgr_to_qpixmap(frame_bgr)
            self.frame_pair_ready.emit(rgb_pixmap, None)
            self.distance_info_ready.emit("Depth Tidak Tersedia", None)

    def _open_camera(self):
        if os.name == "nt":
            backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backend_candidates = [cv2.CAP_ANY]

        for backend in backend_candidates:
            capture = cv2.VideoCapture(self.camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)

            ok, _ = capture.read()
            if ok:
                return capture

            capture.release()

        return None

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
        return True

    def _depth_to_colormap(self, depth_raw):
        with self._threshold_lock:
            depth_min_m = self._depth_min_m
            depth_max_m = self._depth_max_m

        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        valid_mask = (depth_m >= depth_min_m) & (depth_m <= depth_max_m)

        normalized = np.zeros_like(depth_m, dtype=np.float32)
        normalized[valid_mask] = (depth_m[valid_mask] - depth_min_m) / (depth_max_m - depth_min_m)

        depth_8u = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
        depth_colormap[~valid_mask] = (0, 0, 0)
        return depth_colormap

    def _estimate_distance(self, depth_raw):
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        height, width = depth_m.shape

        roi_h = int(height * 0.5)
        roi_w = int(width * 0.5)
        y1 = (height - roi_h) // 2
        x1 = (width - roi_w) // 2
        roi = depth_m[y1 : y1 + roi_h, x1 : x1 + roi_w]

        valid = roi[(roi > 0.10) & (roi < 10.0)]
        if valid.size < 50:
            return None

        return float(np.percentile(valid, 5))

    def _bgr_to_qpixmap(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    def _release_resources(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None
