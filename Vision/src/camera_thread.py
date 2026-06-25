import os
import sys
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)

try:
    from ui_config import DEPTH_MAX_M, DEPTH_MIN_M, DISPLAY_FPS
except ImportError:
    DEPTH_MAX_M = 5.0
    DEPTH_MIN_M = 0.3
    DISPLAY_FPS = 30

try:
    from Vision.inc.camera_config import CameraConfig
    _cam_config = CameraConfig()
    ENABLE_DECIMATION = _cam_config.enable_decimation
except (ImportError, Exception):
    _cam_config = None
    ENABLE_DECIMATION = False

if TYPE_CHECKING:
    from frame_processor import FrameProcessor

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None


class CameraThread(QThread):
    frame_pair_ready = pyqtSignal(QImage, QImage)
    distance_info_ready = pyqtSignal(str, object, str)
    obstacles_ready = pyqtSignal(list)
    navigation_ready = pyqtSignal(dict)
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

        self._processor = processor

        # Cached empty depth QImage (avoid np.zeros_like allocation per frame)
        self._empty_depth_qimage: Optional[QImage] = None

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

        if ENABLE_DECIMATION and _cam_config:
            self._decimation_filter = rs.decimation_filter()
            self._decimation_filter.set_option(rs.option.filter_magnitude, _cam_config.decimation_magnitude)
        else:
            self._decimation_filter = None

        self._spatial_filter = rs.spatial_filter()
        if _cam_config:
            self._spatial_filter.set_option(rs.option.filter_magnitude, _cam_config.spatial_magnitude)
            self._spatial_filter.set_option(rs.option.filter_smooth_alpha, _cam_config.spatial_smooth_alpha)
            self._spatial_filter.set_option(rs.option.filter_smooth_delta, _cam_config.spatial_smooth_delta)

        self._temporal_filter = rs.temporal_filter()
        self._hole_filling_filter = rs.hole_filling_filter()

        return True

    def _realsense_acquisition_worker(self):
        read_failures = 0
        while self._acq_running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                read_failures += 1
                if read_failures >= 10:
                    error_msg = "Gagal membaca stream RealSense secara stabil."
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    break
                time.sleep(0.03)
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
                time.sleep(0.03)
                continue

            depth_raw_unfiltered = np.asanyarray(depth_frame.get_data())

            if self._decimation_filter is not None:
                depth_frame = self._decimation_filter.process(depth_frame)
            depth_frame = self._spatial_filter.process(depth_frame)
            depth_frame = self._temporal_filter.process(depth_frame)
            depth_frame = self._hole_filling_filter.process(depth_frame)

            read_failures = 0
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            if self._decimation_filter is not None and depth_raw.shape != color_bgr.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                depth_raw_unfiltered = cv2.resize(depth_raw_unfiltered, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put((color_bgr, depth_raw, depth_raw_unfiltered))

    def _run_realsense_loop(self):
        self._frame_queue = queue.Queue(maxsize=2)
        self._acq_running = True
        self._acq_thread = threading.Thread(target=self._realsense_acquisition_worker, daemon=True)
        self._acq_thread.start()

        while self._running:
            try:
                frames_data = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            color_bgr, depth_raw, depth_raw_unfiltered = frames_data

            result = None
            zone = "center"
            if self._processor is not None:
                result = self._processor.process(color_bgr, depth_raw, self._depth_scale, depth_raw_unfiltered)
                rgb_qimg = self._bgr_to_qimage(result.rgb_frame, is_depth=False)

                if result.depth_colormap is not None:
                    depth_qimg = self._bgr_to_qimage(result.depth_colormap, is_depth=True)
                else:
                    depth_qimg = self._get_empty_depth_qimage(color_bgr.shape[:2])

                if result.fused_output:
                    dist = result.fused_output[0].get("distance_m")
                    obj_class = result.fused_output[0].get("object_class", "obstacle")
                    zone = result.fused_output[0].get("zone", "center")
                    label = f"Terdeteksi: {obj_class}"
                elif result.obstacles:
                    dist = result.obstacles[0].get("distance_m")
                    zone = result.obstacles[0].get("zone", "center")
                    label = "Objek Terdeteksi"
                else:
                    dist = None
                    label = "Clear"
            else:
                rgb_qimg = self._bgr_to_qimage(color_bgr, is_depth=False)
                depth_qimg = self._get_empty_depth_qimage(color_bgr.shape[:2])
                label = "Clear"
                dist = None

            self.frame_pair_ready.emit(rgb_qimg, depth_qimg)
            self.distance_info_ready.emit(label, dist, zone)

            final_obstacles = []
            nav_data = {}
            if self._processor is not None and result is not None:
                final_obstacles = result.fused_output if result.fused_output else result.obstacles
                nav_data = result.navigation or {}
            self.obstacles_ready.emit(final_obstacles)
            if nav_data:
                self.navigation_ready.emit(nav_data)

        self._acq_running = False
        self._acq_thread.join(timeout=1.0)

    def _webcam_acquisition_worker(self):
        read_failures = 0
        while self._acq_running:
            ok, frame_bgr = self._capture.read()
            if not ok:
                read_failures += 1
                if read_failures >= 10:
                    error_msg = "Gagal membaca frame kamera secara stabil."
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    break
                time.sleep(0.03)
                continue

            read_failures = 0
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame_bgr)

    def _run_webcam_loop(self):
        self._frame_queue = queue.Queue(maxsize=2)
        self._acq_running = True
        self._acq_thread = threading.Thread(target=self._webcam_acquisition_worker, daemon=True)
        self._acq_thread.start()

        while self._running:
            try:
                frame_bgr = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._processor is not None:
                result = self._processor.process(frame_bgr, None, 0.001, None)
                rgb_qimg = self._bgr_to_qimage(result.rgb_frame, is_depth=False)
            else:
                rgb_qimg = self._bgr_to_qimage(frame_bgr, is_depth=False)

            self.frame_pair_ready.emit(rgb_qimg, QImage())
            self.distance_info_ready.emit("Depth Tidak Tersedia", None, "center")
            self.obstacles_ready.emit([])
            if result is not None and result.navigation:
                self.navigation_ready.emit(result.navigation)

        self._acq_running = False
        self._acq_thread.join(timeout=1.0)

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

    def _get_empty_depth_qimage(self, shape) -> QImage:
        """Return a cached empty depth QImage. Avoids np.zeros_like per frame."""
        h, w = shape[:2]
        if self._empty_depth_qimage is None or self._empty_depth_qimage.width() != w or self._empty_depth_qimage.height() != h:
            zeros = np.zeros((h, w, 3), dtype=np.uint8)
            self._empty_depth_qimage = QImage(
                zeros.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
            )
        return self._empty_depth_qimage

    def _bgr_to_qimage(self, frame_bgr, is_depth=False):
        height, width = frame_bgr.shape[:2]
        channels = frame_bgr.shape[2] if frame_bgr.ndim == 3 else 1
        if channels == 3:
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            bytes_per_line = 3 * width
            return QImage(
                frame_rgb.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            return QImage(
                frame_bgr.tobytes(), width, height, width, QImage.Format.Format_Grayscale8
            )

    def _release_resources(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None
