import cv2
import os
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class CameraThread(QThread):
    frame_pair_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._capture = None
        self._pipeline = None
        self._align = None

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
            depth_8u = cv2.convertScaleAbs(depth_raw, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)

            rgb_pixmap = self._bgr_to_qpixmap(color_bgr)
            depth_pixmap = self._bgr_to_qpixmap(depth_colormap)
            self.frame_pair_ready.emit(rgb_pixmap, depth_pixmap)

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
            self._pipeline.start(config)
        except RuntimeError:
            self._pipeline = None
            return False

        self._align = rs.align(rs.stream.color)
        return True

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
