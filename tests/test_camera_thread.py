"""
Tests for CameraThread.
Tests thread lifecycle, signal definitions, and helper methods.
No camera hardware required.
"""

import os
import sys

import numpy as np
import pytest

# No sys.path insertion needed
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from Vision.src.camera_thread import CameraThread


def test_instantiation():
    thread = CameraThread(camera_index=0)
    assert thread.camera_index == 0
    assert thread._running is False
    assert thread._capture is None
    assert thread._pipeline is None


def test_custom_camera_index():
    thread = CameraThread(camera_index=2)
    assert thread.camera_index == 2


def test_depth_thresholds_valid():
    thread = CameraThread()
    thread.set_depth_thresholds(0.5, 4.0)
    assert thread._depth_min_m == 0.5
    assert thread._depth_max_m == 4.0


def test_depth_thresholds_invalid_zero():
    thread = CameraThread()
    thread.set_depth_thresholds(0, 4.0)
    assert thread._depth_min_m == 0.3
    assert thread._depth_max_m == 5.0


def test_depth_thresholds_invalid_negative():
    thread = CameraThread()
    thread.set_depth_thresholds(-1.0, 4.0)
    assert thread._depth_min_m == 0.3
    assert thread._depth_max_m == 5.0


def test_depth_thresholds_min_greater_than_max():
    thread = CameraThread()
    thread.set_depth_thresholds(5.0, 1.0)
    assert thread._depth_min_m == 0.3
    assert thread._depth_max_m == 5.0


def test_depth_thresholds_equal():
    thread = CameraThread()
    thread.set_depth_thresholds(2.0, 2.0)
    assert thread._depth_min_m == 0.3
    assert thread._depth_max_m == 5.0


def test_bgr_to_qimage():
    thread = CameraThread()
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    qimage = thread._bgr_to_qimage(bgr)
    assert qimage.width() == 640
    assert qimage.height() == 480
    assert not qimage.isNull()


def test_bgr_to_qimage_small():
    thread = CameraThread()
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    qimage = thread._bgr_to_qimage(bgr)
    assert qimage.width() == 200
    assert qimage.height() == 100


def test_stop_capture_when_not_running():
    thread = CameraThread()
    thread.stop_capture()
    assert thread._running is False


def test_start_capture_sets_running():
    thread = CameraThread()
    thread._running = False
    thread.start_capture()
    assert thread._running is True
    thread._running = False
    if thread.isRunning():
        thread.wait(500)


def test_release_resources_no_capture():
    thread = CameraThread()
    thread._release_resources()
    assert thread._pipeline is None
    assert thread._capture is None


def test_signals_defined():
    thread = CameraThread()
    assert hasattr(thread, "frame_pair_ready")
    assert hasattr(thread, "distance_info_ready")
    assert hasattr(thread, "error")


def test_processor_integration():
    from detection_config import DetectionConfig
    from frame_processor import FrameProcessor

    config = DetectionConfig()
    processor = FrameProcessor(config)
    thread = CameraThread(processor=processor)
    assert thread._processor is processor


def test_processor_none():
    thread = CameraThread(processor=None)
    assert thread._processor is None


if __name__ == "__main__":
    print("=== CameraThread Tests ===\n")
    tests = [
        ("instantiation", test_instantiation),
        ("custom camera index", test_custom_camera_index),
        ("depth thresholds valid", test_depth_thresholds_valid),
        ("depth thresholds invalid zero", test_depth_thresholds_invalid_zero),
        ("depth thresholds invalid negative", test_depth_thresholds_invalid_negative),
        ("depth thresholds min > max", test_depth_thresholds_min_greater_than_max),
        ("depth thresholds equal", test_depth_thresholds_equal),
        ("bgr to qimage", test_bgr_to_qimage),
        ("bgr to qimage small", test_bgr_to_qimage_small),
        ("stop capture when not running", test_stop_capture_when_not_running),
        ("start capture sets running", test_start_capture_sets_running),
        ("release resources no capture", test_release_resources_no_capture),
        ("signals defined", test_signals_defined),
        ("processor integration", test_processor_integration),
        ("processor none", test_processor_none),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)
