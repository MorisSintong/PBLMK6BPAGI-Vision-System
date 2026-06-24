import pyrealsense2 as rs  # type: ignore[import-untyped]
import numpy as np
import cv2
import os
import datetime

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)


class FrameRecorder:
    # Class-level flag to prevent multiple pipelines on same device
    _active_pipeline_count = 0

    def __init__(self, save_path="data/recordings"):
        self.pipeline = rs.pipeline()  # type: ignore[union-attr]
        self.config = rs.config()  # type: ignore[union-attr]

        # Enable stream RGB dan Depth dari RealSense
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.pipeline_started = False

    def start(self):
        if FrameRecorder._active_pipeline_count > 0:
            logger.error(
                "RealSense pipeline sudah aktif di device lain. "
                "Hentikan CameraThread terlebih dahulu sebelum recording."
            )
            return False

        try:
            self.pipeline.start(self.config)
            self.pipeline_started = True
            FrameRecorder._active_pipeline_count += 1
            logger.info("RealSense camera aktif. Tekan 'q' untuk berhenti.")
            return True
        except Exception as e:
            logger.error(f"RealSense tidak terdeteksi: {e}")
            return False

    def get_frames(self):
        """
        Return: (color_frame, depth_frame) sebagai numpy array,
        atau (None, None) jika gagal.
        """
        if not self.pipeline_started:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            logger.error(f"Error reading frames: {e}")
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def stop(self):
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
            FrameRecorder._active_pipeline_count = max(0, FrameRecorder._active_pipeline_count - 1)
        logger.info("RealSense dimatikan.")


if __name__ == "__main__":
    recorder = FrameRecorder()
    if recorder.start():
        try:
            while True:
                color_image, depth_image = recorder.get_frames()
                if color_image is not None:
                    # Visualisasi depth sebagai colormap
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    combined = np.hstack((color_image, depth_colormap))
                    cv2.imshow('RealSense - Color | Depth', combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            recorder.stop()
            cv2.destroyAllWindows()