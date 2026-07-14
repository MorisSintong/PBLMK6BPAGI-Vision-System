"""
Vision/src/video_recorder.py — Standalone RealSense RGB + Depth recorder.

Records synchronized RGB and Depth streams from an Intel RealSense camera
and saves them to disk for later playback through the vision pipeline.

Output structure:
    recording_YYYYMMDD_HHMMSS/
    ├── rgb.avi                  # RGB video (MJPG codec)
    ├── depth/                   # Filtered depth frames
    │   ├── frame_00000.npy
    │   ├── frame_00001.npy
    │   └── ...
    ├── depth_raw/               # Unfiltered depth frames
    │   ├── frame_00000.npy
    │   ├── frame_00001.npy
    │   └── ...
    └── metadata.json            # Recording metadata

Usage:
    python -m Vision.src.video_recorder
    python -m Vision.src.video_recorder --output data/recordings --duration 60
"""

import os
import sys
import json
import time
import argparse
import datetime
from typing import Optional

import cv2
import numpy as np

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None

try:
    from Vision.inc.camera_config import CameraConfig
    _cam_config = CameraConfig()
except (ImportError, Exception):
    _cam_config = None


class VideoRecorder:
    """Records synchronized RGB + Depth streams from Intel RealSense.

    All data is written to disk in real-time (no in-memory buffering) so
    recordings of any length are safe — no risk of running out of memory.

    Supports two modes:
    - CLI blocking: call record() which runs a loop until user presses 'q'.
    - GUI non-blocking: call start_recording(), then record_frame() per frame,
      then stop_recording() to finalize.
    """

    def __init__(self, save_dir: str = "data/recordings") -> None:
        self._save_dir = save_dir
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001

        # Filters — matching CameraThread configuration
        self._spatial_filter = None
        self._temporal_filter = None
        self._hole_filling_filter = None
        self._decimation_filter = None

        # Recording state
        self._recording = False
        self._frame_count = 0
        self._rgb_writer = None
        self._recording_dir = ""
        self._depth_dir = ""
        self._depth_raw_dir = ""
        self._start_time = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def start_pipeline(self) -> bool:
        """Initialize and start the RealSense pipeline."""
        if rs is None:
            logger.error("pyrealsense2 is not installed")
            return False

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            profile = self._pipeline.start(config)
        except RuntimeError as e:
            logger.error(f"Failed to start RealSense pipeline: {e}")
            self._pipeline = None
            return False

        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._setup_filters()

        logger.info(f"RealSense started | depth_scale={self._depth_scale:.6f}")
        return True

    def start_recording(self, output_dir: Optional[str] = None) -> str:
        """Start a non-blocking recording session (GUI mode).

        Frames are written to disk in real-time via record_frame().
        Call stop_recording() to finalize metadata.

        Returns:
            Path to the recording directory.
        """
        if output_dir is not None:
            self._save_dir = output_dir

        rec_dir = self._create_recording_dirs()

        # Setup RGB video writer
        rgb_path = os.path.join(rec_dir, "rgb.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 30.0, (640, 480))

        if not self._rgb_writer.isOpened():
            logger.error(f"MJPG codec failed, falling back to XVID")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 30.0, (640, 480))

        self._frame_count = 0
        self._recording = True
        self._start_time = time.time()

        logger.info(f"Non-blocking recording started -> {rec_dir}")
        return rec_dir

    def record_frame(
        self,
        rgb_bgr: np.ndarray,
        depth_filtered: Optional[np.ndarray] = None,
        depth_raw: Optional[np.ndarray] = None,
    ) -> None:
        """Record a single frame pair (called from CameraThread per frame).

        Writes all data to disk immediately — no in-memory buffering.

        Args:
            rgb_bgr: BGR frame from camera (uint8, HxWx3).
            depth_filtered: Filtered depth frame (uint16, HxW) or None.
            depth_raw: Unfiltered depth frame (uint16, HxW) or None.
        """
        if not self._recording or self._rgb_writer is None:
            return

        # Write RGB to .avi in real-time
        self._rgb_writer.write(rgb_bgr)

        # Write depth frames to individual .npy files in real-time
        frame_name = f"frame_{self._frame_count:05d}.npy"

        if depth_filtered is not None:
            np.save(os.path.join(self._depth_dir, frame_name), depth_filtered)

        if depth_raw is not None:
            np.save(os.path.join(self._depth_raw_dir, frame_name), depth_raw)

        self._frame_count += 1

        # Log progress every 300 frames (~10s)
        if self._frame_count % 300 == 0:
            elapsed = time.time() - self._start_time
            logger.info(
                f"Recording: {self._frame_count} frames | {elapsed:.1f}s"
            )

    def stop_recording(self) -> str:
        """Stop recording and write metadata.

        Returns:
            Path to the recording directory.
        """
        if not self._recording:
            return self._recording_dir

        self._recording = False
        duration_s = time.time() - self._start_time

        # Release video writer
        if self._rgb_writer is not None:
            self._rgb_writer.release()
            self._rgb_writer = None

        # Count actual depth files
        depth_count = len([f for f in os.listdir(self._depth_dir) if f.endswith(".npy")]) if os.path.isdir(self._depth_dir) else 0
        depth_raw_count = len([f for f in os.listdir(self._depth_raw_dir) if f.endswith(".npy")]) if os.path.isdir(self._depth_raw_dir) else 0

        # Save metadata
        metadata = {
            "depth_scale": self._depth_scale,
            "width": 640,
            "height": 480,
            "fps": 30,
            "frame_count": self._frame_count,
            "duration_s": round(duration_s, 2),
            "timestamp": datetime.datetime.now().isoformat(),
            "has_depth": depth_count > 0,
            "has_filtered_depth": depth_count > 0,
            "has_raw_depth": depth_raw_count > 0,
            "depth_format": "individual_npy",
        }
        meta_path = os.path.join(self._recording_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Recording saved -> {self._recording_dir}\n"
            f"  Frames: {self._frame_count}\n"
            f"  Duration: {duration_s:.1f}s\n"
            f"  Depth files: {depth_count} | Raw depth files: {depth_raw_count}\n"
            f"  Depth scale: {self._depth_scale}"
        )
        return self._recording_dir

    def stop_pipeline(self) -> None:
        """Stop the RealSense pipeline."""
        if self._recording:
            self.stop_recording()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None
        logger.info("RealSense pipeline stopped")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    # ── CLI blocking mode ───────────────────────────────────────────────────

    def record(
        self,
        max_duration_s: float = 0,
        show_preview: bool = True,
    ) -> str:
        """Record RGB + Depth to disk (blocking CLI mode).

        Args:
            max_duration_s: Maximum recording duration in seconds (0 = unlimited).
            show_preview: Whether to show a live preview window.

        Returns:
            Path to the recording directory.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not started. Call start_pipeline() first.")

        rec_dir = self._create_recording_dirs()

        # Setup RGB video writer
        rgb_path = os.path.join(rec_dir, "rgb.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 30.0, (640, 480))

        if not self._rgb_writer.isOpened():
            logger.error("MJPG codec failed, falling back to XVID")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 30.0, (640, 480))

        self._frame_count = 0
        self._recording = True
        self._start_time = time.time()

        logger.info(f"Recording started -> {rec_dir}")
        logger.info("Press 'q' in preview window to stop (or Ctrl+C in terminal)")

        try:
            while self._recording:
                # Check duration limit
                elapsed = time.time() - self._start_time
                if max_duration_s > 0 and elapsed >= max_duration_s:
                    logger.info(f"Max duration reached ({max_duration_s}s)")
                    break

                # Grab frames
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    logger.warning("Frame timeout, retrying...")
                    continue

                aligned = self._align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Get numpy arrays
                color_bgr = np.asanyarray(color_frame.get_data())
                depth_raw_unfiltered = np.asanyarray(depth_frame.get_data())

                # Apply filters (same as CameraThread)
                filtered_depth_frame = self._filter_depth(depth_frame)
                depth_raw_filtered = np.asanyarray(filtered_depth_frame.get_data())

                # Resize if decimation changed resolution
                if (self._decimation_filter is not None
                        and depth_raw_filtered.shape != color_bgr.shape[:2]):
                    depth_raw_filtered = cv2.resize(
                        depth_raw_filtered,
                        (color_bgr.shape[1], color_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    depth_raw_unfiltered = cv2.resize(
                        depth_raw_unfiltered,
                        (color_bgr.shape[1], color_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # Write frame to disk in real-time
                self.record_frame(color_bgr, depth_raw_filtered, depth_raw_unfiltered)

                # Show preview
                if show_preview:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_raw_filtered, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    combined = np.hstack((color_bgr, depth_colormap))

                    info_text = (
                        f"REC | Frame: {self._frame_count} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    cv2.putText(
                        combined, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    cv2.circle(combined, (combined.shape[1] - 30, 25), 10, (0, 0, 255), -1)

                    cv2.imshow("RealSense Recorder - RGB | Depth  [Press Q to stop]", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        finally:
            self._recording = False
            if show_preview:
                cv2.destroyAllWindows()

            # Write metadata
            self.stop_recording()

        return rec_dir

    # ── Internal helpers ────────────────────────────────────────────────────

    def _setup_filters(self) -> None:
        """Configure depth post-processing filters (matches CameraThread)."""
        if rs is None:
            return

        enable_decimation = False
        if _cam_config:
            enable_decimation = _cam_config.enable_decimation

        if enable_decimation and _cam_config:
            self._decimation_filter = rs.decimation_filter()
            self._decimation_filter.set_option(
                rs.option.filter_magnitude, _cam_config.decimation_magnitude
            )
        else:
            self._decimation_filter = None

        self._spatial_filter = rs.spatial_filter()
        if _cam_config:
            self._spatial_filter.set_option(
                rs.option.filter_magnitude, _cam_config.spatial_magnitude
            )
            self._spatial_filter.set_option(
                rs.option.filter_smooth_alpha, _cam_config.spatial_smooth_alpha
            )
            self._spatial_filter.set_option(
                rs.option.filter_smooth_delta, _cam_config.spatial_smooth_delta
            )

        self._temporal_filter = rs.temporal_filter()
        self._hole_filling_filter = rs.hole_filling_filter()

    def _filter_depth(self, depth_frame):
        """Apply depth post-processing pipeline (matches CameraThread)."""
        if self._decimation_filter is not None:
            depth_frame = self._decimation_filter.process(depth_frame)
        depth_frame = self._spatial_filter.process(depth_frame)
        depth_frame = self._temporal_filter.process(depth_frame)
        depth_frame = self._hole_filling_filter.process(depth_frame)
        return depth_frame

    def _create_recording_dirs(self) -> str:
        """Create timestamped recording directory with depth subdirectories."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_dir = os.path.join(self._save_dir, f"recording_{timestamp}")

        self._recording_dir = rec_dir
        self._depth_dir = os.path.join(rec_dir, "depth")
        self._depth_raw_dir = os.path.join(rec_dir, "depth_raw")

        os.makedirs(self._depth_dir, exist_ok=True)
        os.makedirs(self._depth_raw_dir, exist_ok=True)

        # Write initial metadata.json to ensure it always exists even if process is killed
        initial_metadata = {
            "depth_scale": self._depth_scale,
            "width": 640,
            "height": 480,
            "fps": 30,
            "frame_count": 0,
            "duration_s": 0.0,
            "timestamp": datetime.datetime.now().isoformat(),
            "has_depth": True,
            "has_filtered_depth": True,
            "has_raw_depth": True,
            "depth_format": "individual_npy",
        }
        meta_path = os.path.join(rec_dir, "metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(initial_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write initial metadata: {e}")

        return rec_dir


def main():
    parser = argparse.ArgumentParser(
        description="Record RGB + Depth video from Intel RealSense"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/recordings",
        help="Output directory for recordings (default: data/recordings)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=0,
        help="Max recording duration in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window",
    )
    args = parser.parse_args()

    if rs is None:
        logger.error("pyrealsense2 is not installed. Install with: pip install pyrealsense2")
        sys.exit(1)

    recorder = VideoRecorder(save_dir=args.output)

    if not recorder.start_pipeline():
        logger.error("Failed to start RealSense. Is the camera connected?")
        sys.exit(1)

    try:
        rec_dir = recorder.record(
            max_duration_s=args.duration,
            show_preview=not args.no_preview,
        )
        print(f"\nRecording saved to: {rec_dir}")
    finally:
        recorder.stop_pipeline()


if __name__ == "__main__":
    main()
