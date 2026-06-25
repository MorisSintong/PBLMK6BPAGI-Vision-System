"""
Tests for CameraThread.
Tests thread lifecycle, signal definitions, helper methods, and QImage conversion.
No camera hardware required.
"""

import os
import sys

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage

app = QApplication.instance() or QApplication(sys.argv)

from Vision.src.camera_thread import CameraThread


# ═══════════════════════════════════════════════════════════════════════════════
# Instantiation & Properties
# ═══════════════════════════════════════════════════════════════════════════════

def test_instantiation():
    thread = CameraThread(camera_index=0)
    assert thread.camera_index == 0
    assert thread._running is False
    assert thread._capture is None
    assert thread._pipeline is None


def test_custom_camera_index():
    thread = CameraThread(camera_index=2)
    assert thread.camera_index == 2


def test_processor_stored():
    """Processor should be stored on the thread."""
    thread = CameraThread()
    assert thread._processor is None


# ═══════════════════════════════════════════════════════════════════════════════
# Depth Threshold Validation
# ═══════════════════════════════════════════════════════════════════════════════

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


def test_depth_thresholds_propagates_to_processor():
    """set_depth_thresholds should call processor.set_depth_thresholds."""
    from Vision.inc.detection_config import DetectionConfig
    from Vision.src.frame_processor import FrameProcessor
    processor = FrameProcessor(DetectionConfig())
    thread = CameraThread(processor=processor)
    thread.set_depth_thresholds(0.5, 4.0)
    depth_stage = processor.get_stage("DepthProcessingStage")
    assert depth_stage._depth_min_m == 0.5
    assert depth_stage._depth_max_m == 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# BGR → QImage Conversion
# ═══════════════════════════════════════════════════════════════════════════════

def test_bgr_to_qimage():
    """Standard 480x640 BGR frame → valid QImage."""
    thread = CameraThread()
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    qimage = thread._bgr_to_qimage(bgr)
    assert qimage.width() == 640
    assert qimage.height() == 480
    assert not qimage.isNull()


def test_bgr_to_qimage_small():
    """Small frame → correct dimensions."""
    thread = CameraThread()
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    qimage = thread._bgr_to_qimage(bgr)
    assert qimage.width() == 200
    assert qimage.height() == 100


def test_bgr_to_qimage_pixel_integrity():
    """BGR→RGB swap: pixel (10,20) BGR=[10,20,30] → RGB=[30,20,10]."""
    thread = CameraThread()
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[10, 20] = [10, 20, 30]  # B=10, G=20, R=30
    qimage = thread._bgr_to_qimage(bgr)
    # QImage stores in RGB888 format — extract pixel and verify channels swapped
    # QPixmap.fromImage would display correctly; check via raw pixel
    ptr = qimage.constBits()
    ptr.setsize(qimage.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((100, 100, 3))
    # arr[10, 20] should be RGB = [30, 20, 10]
    assert arr[10, 20, 0] == 30  # R
    assert arr[10, 20, 1] == 20  # G
    assert arr[10, 20, 2] == 10  # B


def test_bgr_to_qimage_grayscale():
    """Single-channel frame → Grayscale8 QImage."""
    thread = CameraThread()
    gray = np.full((100, 200), 128, dtype=np.uint8)
    qimage = thread._bgr_to_qimage(gray)
    assert qimage.width() == 200
    assert qimage.height() == 100
    assert not qimage.isNull()


def test_bgr_to_qimage_preserves_dimensions():
    """Various frame sizes → correct QImage dimensions."""
    thread = CameraThread()
    for h, w in [(1, 1), (10, 10), (240, 320), (480, 640), (720, 1280)]:
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        qimage = thread._bgr_to_qimage(bgr)
        assert qimage.width() == w
        assert qimage.height() == h


# ═══════════════════════════════════════════════════════════════════════════════
# Empty Depth QImage Cache
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_empty_depth_qimage_cached():
    """Repeated calls with same shape → same QImage object."""
    thread = CameraThread()
    q1 = thread._get_empty_depth_qimage((480, 640))
    q2 = thread._get_empty_depth_qimage((480, 640))
    assert q1 is q2, "Empty depth QImage should be cached"


def test_get_empty_depth_qimage_different_shape():
    """Different shape → new QImage object."""
    thread = CameraThread()
    q1 = thread._get_empty_depth_qimage((480, 640))
    q2 = thread._get_empty_depth_qimage((240, 320))
    assert q1 is not q2
    assert q2.width() == 320
    assert q2.height() == 240


def test_get_empty_depth_qimage_valid():
    """Cached empty QImage should be valid (not null)."""
    thread = CameraThread()
    q = thread._get_empty_depth_qimage((480, 640))
    assert not q.isNull()
    assert q.width() == 640
    assert q.height() == 480


# ═══════════════════════════════════════════════════════════════════════════════
# Thread Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def test_signals_defined():
    thread = CameraThread()
    assert hasattr(thread, "frame_pair_ready")
    assert hasattr(thread, "distance_info_ready")
    assert hasattr(thread, "error")


def test_obstacles_ready_signal_defined():
    thread = CameraThread()
    assert hasattr(thread, "obstacles_ready")


# ═══════════════════════════════════════════════════════════════════════════════
# Processor Integration
# ═══════════════════════════════════════════════════════════════════════════════

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
        ("processor stored", test_processor_stored),
        ("depth thresholds valid", test_depth_thresholds_valid),
        ("depth thresholds invalid zero", test_depth_thresholds_invalid_zero),
        ("depth thresholds invalid negative", test_depth_thresholds_invalid_negative),
        ("depth thresholds min > max", test_depth_thresholds_min_greater_than_max),
        ("depth thresholds equal", test_depth_thresholds_equal),
        ("depth thresholds propagate to processor", test_depth_thresholds_propagates_to_processor),
        ("bgr to qimage", test_bgr_to_qimage),
        ("bgr to qimage small", test_bgr_to_qimage_small),
        ("bgr to qimage pixel integrity", test_bgr_to_qimage_pixel_integrity),
        ("bgr to qimage grayscale", test_bgr_to_qimage_grayscale),
        ("bgr to qimage preserves dimensions", test_bgr_to_qimage_preserves_dimensions),
        ("get empty depth qimage cached", test_get_empty_depth_qimage_cached),
        ("get empty depth qimage different shape", test_get_empty_depth_qimage_different_shape),
        ("get empty depth qimage valid", test_get_empty_depth_qimage_valid),
        ("stop capture when not running", test_stop_capture_when_not_running),
        ("start capture sets running", test_start_capture_sets_running),
        ("release resources no capture", test_release_resources_no_capture),
        ("signals defined", test_signals_defined),
        ("obstacles ready signal defined", test_obstacles_ready_signal_defined),
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
