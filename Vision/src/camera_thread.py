import os
import sys
import threading
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from logging_config import get_logger

logger = get_logger(__name__)

try:
    from ui_config import DEPTH_MAX_M, DEPTH_MIN_M, DISPLAY_FPS
except ImportError:
    DEPTH_MAX_M = 5.0
    DEPTH_MIN_M = 0.3
    DISPLAY_FPS = 30

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
        self._depth_scale = 0.001
        self._depth_min_m = DEPTH_MIN_M
        self._depth_max_m = DEPTH_MAX_M
        self._threshold_lock = threading.Lock()
        self._frame_delay_ms = max(1, 1000 // DISPLAY_FPS)
        
        # Fitur Moris yang dikembalikan
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

    # Fitur Moris yang dikembalikan
    def set_depth_thresholds(self, depth_min_m: float, depth_max_m: float):
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return

        with self._threshold_lock:
            self._depth_min_m = depth_min_m
            self._depth_max_m = depth_max_m

        if self._processor is not None:
            self._processor.set_depth_thresholds(depth_min_m, depth_max_m)

    def run(self):
        if self._start_realsense():
            logger.info("RealSense camera started successfully")
            self._run_realsense_loop()
        else:
            self._capture = self._open_camera()
            if self._capture is None:
                error_msg = "Kamera gagal dibuka. Tutup aplikasi lain yang sedang memakai kamera."
                logger.error(error_msg)
                self.error.emit(error_msg)
                self._release_resources()
                return
            logger.info("Webcam fallback activated")
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

        # --- INISIALISASI FILTER KAMERA DEPTH (Tetap dipertahankan) ---
        self._decimation_filter = rs.decimation_filter()
        self._decimation_filter.set_option(rs.option.filter_magnitude, 2)
        
        self._spatial_filter = rs.spatial_filter()
        self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        self._temporal_filter = rs.temporal_filter()
        self._hole_filling_filter = rs.hole_filling_filter()
        # --------------------------------------------------------------

        return True

    def _run_realsense_loop(self):
        read_failures = 0  # Fitur Moris yang dikembalikan
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                read_failures += 1
                if read_failures >= 10:
                    error_msg = "Gagal membaca stream RealSense secara stabil."
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    break
                self.msleep(30)
                continue

            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                read_failures += 1
                if read_failures >= 10:
                    error_msg = "Frame RealSense tidak lengkap."
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    break
                self.msleep(30)
                continue

            # --- APLIKASI FILTER KAMERA DEPTH ---
            depth_frame = self._decimation_filter.process(depth_frame)
            depth_frame = self._spatial_filter.process(depth_frame)
            depth_frame = self._temporal_filter.process(depth_frame)
            depth_frame = self._hole_filling_filter.process(depth_frame)
            # ------------------------------------

            read_failures = 0
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Kembalikan ukuran depth map ke 640x480 karena decimation mengecilkannya
            if depth_raw.shape != color_bgr.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Fitur Moris yang dikembalikan: Pipeline Integration
            if self._processor is not None:
                result = self._processor.process(color_bgr, depth_raw, self._depth_scale)
                rgb_pixmap = self._bgr_to_qimage(result.rgb_frame)
                depth_pixmap = self._bgr_to_qimage(
                    result.depth_colormap if result.depth_colormap is not None else np.zeros_like(color_bgr)
                )

                if result.obstacles:
                    dist = result.obstacles[0].get("distance_m")
                    label = "Objek Terdeteksi"
                else:
                    dist = None
                    label = "Clear"
            else:
                # Mode fallback jika processor tidak ada
                rgb_pixmap = self._bgr_to_qimage(color_bgr)
                depth_pixmap = None
                label = "Clear"
                dist = None

            self.frame_pair_ready.emit(rgb_pixmap, depth_pixmap)
            self.distance_info_ready.emit(label, dist)
            self.msleep(self._frame_delay_ms)

    def _run_webcam_loop(self):
        read_failures = 0
        while self._running:
            ok, frame_bgr = self._capture.read()
            if not ok:
                read_failures += 1
                if read_failures >= 10:
                    error_msg = "Gagal membaca frame kamera secara stabil."
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    break
                self.msleep(30)
                continue

            read_failures = 0

            if self._processor is not None:
                result = self._processor.process(frame_bgr, None)
                rgb_pixmap = self._bgr_to_qimage(result.rgb_frame)
            else:
                rgb_pixmap = self._bgr_to_qimage(frame_bgr)

            self.frame_pair_ready.emit(rgb_pixmap, None)
            self.distance_info_ready.emit("Depth Tidak Tersedia", None)
            self.msleep(self._frame_delay_ms)

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

    def _bgr_to_qimage(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        return QImage(
            frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()

    def _release_resources(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None