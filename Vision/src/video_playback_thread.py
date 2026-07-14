"""
Vision/src/video_playback_thread.py — QThread for replaying recorded RGB + Depth.

Reads a recording directory (created by VideoRecorder) and feeds frame pairs
through the FrameProcessor pipeline, emitting the same signals as CameraThread.

This allows the full vision pipeline (YOLO, depth processing, fusion, navigation,
visual annotation) to run on pre-recorded data — no live camera needed.

Supported depth formats:
    - Stacked .npy: depth.npy with shape (N, H, W) uint16 — fast, preferred
    - Individual .npy: depth/frame_00000.npy, depth/frame_00001.npy, ... — legacy
    - RGB-only: no depth files — depth pipeline stages are skipped
"""

import glob
import json
import os
import time
import threading
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from Vision.src.frame_processor import FrameProcessor


class VideoPlaybackThread(QThread):
    """Replays a recorded RGB + Depth session through the vision pipeline.

    Emits the same signals as CameraThread so all downstream GUI widgets
    (DepthView, AlertPanel, RadarView) work identically.
    """

    # ── Signals (identical to CameraThread) ──────────────────────────────────
    frame_pair_ready = pyqtSignal(QImage, QImage)
    distance_info_ready = pyqtSignal(str, object, str)
    obstacles_ready = pyqtSignal(list)
    navigation_ready = pyqtSignal(dict)
    light_mode_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    # ── Playback-specific signals ────────────────────────────────────────────
    playback_progress = pyqtSignal(int, int)       # (current_frame, total_frames)
    playback_finished = pyqtSignal()

    def __init__(
        self,
        recording_dir: str,
        parent=None,
        processor: "Optional[FrameProcessor]" = None,
    ) -> None:
        super().__init__(parent)
        self._recording_dir = recording_dir
        self._processor = processor
        self._running = False

        # Playback controls
        self._paused = False
        self._pause_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._speed = 1.0  # 0.25x - 4.0x
        self._loop = False

        # Recording data (loaded on start)
        self._metadata = {}
        self._depth_scale = 0.001
        self._fps = 30
        self._total_frames = 0
        self._rgb_capture = None
        self._has_depth = False
        self._has_raw_depth = False

        # Depth data — stacked arrays in memory (fast access)
        self._depth_stack: Optional[np.ndarray] = None       # (N, H, W) uint16
        self._depth_raw_stack: Optional[np.ndarray] = None   # (N, H, W) uint16

        # Legacy: individual .npy files (fallback if stacked not found)
        self._depth_files: List[str] = []
        self._depth_raw_files: List[str] = []

        # Depth thresholds (matching CameraThread interface)
        self._depth_min_m = 0.3
        self._depth_max_m = 5.0
        self._threshold_lock = threading.Lock()

        # Cached empty depth QImage
        self._empty_depth_qimage: Optional[QImage] = None

        # Separate reusable buffers for RGB and depth BGR→RGB conversion.
        # Two buffers are needed because both rgb_qimg and depth_qimg are
        # created in the same frame iteration — sharing a single buffer
        # would cause the second cv2.cvtColor to overwrite the first,
        # making both QImages point to the same (depth) data.
        self._rgb_qimg_buffer: Optional[np.ndarray] = None
        self._depth_qimg_buffer: Optional[np.ndarray] = None

    # ── Public control methods ───────────────────────────────────────────────

    def start_playback(self) -> None:
        """Start playback (equivalent to CameraThread.start_capture)."""
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_playback(self) -> None:
        """Stop playback (equivalent to CameraThread.stop_capture)."""
        self._running = False
        # Unpause so the thread can exit
        self._pause_event.set()
        if self.isRunning():
            self.wait(2000)

    # Aliases for CameraThread compatibility
    start_capture = start_playback
    stop_capture = stop_playback

    def set_paused(self, paused: bool) -> None:
        """Pause or resume playback."""
        self._paused = paused
        if paused:
            self._pause_event.clear()
        else:
            self._pause_event.set()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.set_paused(not self._paused)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (e.g. 0.5, 1.0, 2.0)."""
        self._speed = max(0.25, min(4.0, speed))

    def set_loop(self, loop: bool) -> None:
        """Enable or disable looping."""
        self._loop = loop

    def set_depth_thresholds(self, depth_min_m: float, depth_max_m: float) -> None:
        """Update depth thresholds (CameraThread compatibility)."""
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return
        with self._threshold_lock:
            self._depth_min_m = depth_min_m
            self._depth_max_m = depth_max_m
        if self._processor is not None:
            self._processor.set_depth_thresholds(depth_min_m, depth_max_m)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def recording_dir(self) -> str:
        return self._recording_dir

    # ── Thread execution ─────────────────────────────────────────────────────

    def _load_recording(self) -> bool:
        """Load recording metadata and validate file structure."""
        meta_path = os.path.join(self._recording_dir, "metadata.json")
        if not os.path.exists(meta_path):
            self.error.emit(f"metadata.json tidak ditemukan di: {self._recording_dir}")
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._depth_scale = self._metadata.get("depth_scale", 0.001)
        self._fps = self._metadata.get("fps", 30)
        self._total_frames = self._metadata.get("frame_count", 0)

        # Open RGB video
        rgb_path = os.path.join(self._recording_dir, "rgb.avi")
        if not os.path.exists(rgb_path):
            self.error.emit(f"rgb.avi tidak ditemukan di: {self._recording_dir}")
            return False

        self._rgb_capture = cv2.VideoCapture(rgb_path)
        if not self._rgb_capture.isOpened():
            self.error.emit("Gagal membuka rgb.avi")
            return False

        # If total_frames is 0 in metadata, get from video
        if self._total_frames == 0:
            self._total_frames = int(
                self._rgb_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

        # Load depth data — prefer stacked .npy (fast), fallback to individual files
        self._depth_stack = None
        self._depth_raw_stack = None
        self._depth_files = []
        self._depth_raw_files = []

        self._has_depth = self._metadata.get("has_depth", False)
        self._has_raw_depth = self._metadata.get("has_raw_depth", False)

        depth_stack_path = os.path.join(self._recording_dir, "depth.npy")
        depth_raw_stack_path = os.path.join(self._recording_dir, "depth_raw.npy")

        if os.path.exists(depth_stack_path):
            # Stacked format — load all frames into memory at once
            try:
                self._depth_stack = np.load(depth_stack_path)
                self._has_depth = True
                logger.info(f"Loaded depth.npy: shape={self._depth_stack.shape}")
            except Exception as e:
                logger.warning(f"Failed to load depth.npy: {e}")

        if os.path.exists(depth_raw_stack_path):
            try:
                self._depth_raw_stack = np.load(depth_raw_stack_path)
                self._has_raw_depth = True
                logger.info(f"Loaded depth_raw.npy: shape={self._depth_raw_stack.shape}")
            except Exception as e:
                logger.warning(f"Failed to load depth_raw.npy: {e}")

        # Fallback: individual .npy files (legacy format)
        if self._depth_stack is None:
            depth_dir = os.path.join(self._recording_dir, "depth")
            if os.path.isdir(depth_dir):
                self._depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
                self._has_depth = len(self._depth_files) > 0

        if self._depth_raw_stack is None:
            depth_raw_dir = os.path.join(self._recording_dir, "depth_raw")
            if os.path.isdir(depth_raw_dir):
                self._depth_raw_files = sorted(glob.glob(os.path.join(depth_raw_dir, "*.npy")))
                self._has_raw_depth = len(self._depth_raw_files) > 0

        logger.info(
            f"Recording loaded: {self._recording_dir}\n"
            f"  Frames: {self._total_frames} | FPS: {self._fps} | "
            f"Depth scale: {self._depth_scale}\n"
            f"  Has depth: {self._has_depth} | Has raw depth: {self._has_raw_depth}\n"
            f"  Depth format: {'stacked' if self._depth_stack is not None else 'individual' if self._depth_files else 'none'}"
        )
        return True

    def _get_depth_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get filtered depth frame for the given index."""
        if self._depth_stack is not None:
            if frame_idx < self._depth_stack.shape[0]:
                return self._depth_stack[frame_idx]
            return None
        if frame_idx < len(self._depth_files):
            try:
                return np.load(self._depth_files[frame_idx])
            except Exception as e:
                logger.warning(f"Failed to load depth frame {frame_idx}: {e}")
        return None

    def _get_depth_raw_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get unfiltered depth frame for the given index."""
        if self._depth_raw_stack is not None:
            if frame_idx < self._depth_raw_stack.shape[0]:
                return self._depth_raw_stack[frame_idx]
            return None
        if frame_idx < len(self._depth_raw_files):
            try:
                return np.load(self._depth_raw_files[frame_idx])
            except Exception as e:
                logger.warning(f"Failed to load raw depth frame {frame_idx}: {e}")
        return None

    def run(self) -> None:
        """Main playback loop — reads frames and runs them through pipeline."""
        if not self._load_recording():
            return

        frame_idx = 0
        target_delay = 1.0 / self._fps if self._fps > 0 else 1.0 / 30

        while self._running:
            # Wait if paused
            self._pause_event.wait()
            if not self._running:
                break

            # Read RGB frame
            ok, color_bgr = self._rgb_capture.read()
            if not ok:
                if self._loop:
                    self._rgb_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                else:
                    logger.info("Playback finished")
                    self.playback_finished.emit()
                    break

            # Load depth frames (fast: from stacked array or individual files)
            depth_filtered = self._get_depth_frame(frame_idx)
            depth_raw = self._get_depth_raw_frame(frame_idx)

            # Process through pipeline (same as CameraThread)
            t0 = time.perf_counter()

            result = None
            zone = "center"
            if self._processor is not None:
                result = self._processor.process(
                    color_bgr, depth_filtered, self._depth_scale, depth_raw
                )
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

            # Emit signals (identical to CameraThread)
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
            if self._processor is not None and result is not None:
                is_dark = result.metadata.get("is_dark", False)
                self.light_mode_changed.emit(is_dark)

            # Emit progress
            self.playback_progress.emit(frame_idx + 1, self._total_frames)

            frame_idx += 1

            # Frame rate control (adjusted by speed multiplier)
            process_time = time.perf_counter() - t0
            adjusted_delay = target_delay / self._speed
            sleep_time = adjusted_delay - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        self._release_resources()

    # ── Image conversion (identical to CameraThread) ─────────────────────────

    def _bgr_to_qimage(self, frame_bgr: np.ndarray, is_depth: bool = False) -> QImage:
        height, width = frame_bgr.shape[:2]
        channels = frame_bgr.shape[2] if frame_bgr.ndim == 3 else 1
        if channels == 3:
            # Use a separate reusable buffer per is_depth flag.
            # Safe because callers immediately convert QImage → QPixmap
            # (which copies the data) before the next frame reuses the buffer.
            if is_depth:
                buf = self._depth_qimg_buffer
                if buf is None or buf.shape != frame_bgr.shape:
                    buf = np.empty_like(frame_bgr)
                    self._depth_qimg_buffer = buf
            else:
                buf = self._rgb_qimg_buffer
                if buf is None or buf.shape != frame_bgr.shape:
                    buf = np.empty_like(frame_bgr)
                    self._rgb_qimg_buffer = buf
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=buf)
            bytes_per_line = 3 * width
            return QImage(
                buf.data, width, height, bytes_per_line,
                QImage.Format.Format_RGB888,
            )
        else:
            return QImage(
                frame_bgr.tobytes(), width, height, width,
                QImage.Format.Format_Grayscale8,
            )

    def _get_empty_depth_qimage(self, shape) -> QImage:
        """Return a cached empty depth QImage."""
        h, w = shape[:2]
        if (self._empty_depth_qimage is None
                or self._empty_depth_qimage.width() != w
                or self._empty_depth_qimage.height() != h):
            zeros = np.zeros((h, w, 3), dtype=np.uint8)
            self._empty_depth_qimage = QImage(
                zeros.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888,
            )
        return self._empty_depth_qimage

    def _release_resources(self) -> None:
        if self._rgb_capture is not None:
            self._rgb_capture.release()
            self._rgb_capture = None
        self._depth_stack = None
        self._depth_raw_stack = None
        logger.info("Playback resources released")
