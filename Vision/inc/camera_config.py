"""
Vision/inc/camera_config.py — Camera configuration parameters.

Centralizes all camera-related settings for RealSense and webcam fallback.
"""


class CameraConfig:
    """Camera configuration for Intel RealSense D455 and webcam fallback."""

    def __init__(self):
        # RealSense D455 settings
        self.realsense_width = 640
        self.realsense_height = 480
        self.realsense_fps = 30
        self.realsense_color_format = "bgr8"
        self.realsense_depth_format = "z16"

        # Webcam fallback settings
        self.webcam_width = 1280
        self.webcam_height = 720
        self.webcam_fps = 30
        self.webcam_index = 0

        # Depth filter settings (RealSense SDK)
        self.enable_decimation = False  # Disabled by default to preserve 640x480 resolution
        self.decimation_magnitude = 2
        self.spatial_magnitude = 2
        self.spatial_smooth_alpha = 0.5
        self.spatial_smooth_delta = 20

        # Camera backend (Windows: DirectShow, Linux: Auto)
        self.prefer_directshow = True

    def get_realsense_resolution(self):
        """Return RealSense resolution as (width, height) tuple."""
        return (self.realsense_width, self.realsense_height)

    def get_webcam_resolution(self):
        """Return webcam resolution as (width, height) tuple."""
        return (self.webcam_width, self.webcam_height)
