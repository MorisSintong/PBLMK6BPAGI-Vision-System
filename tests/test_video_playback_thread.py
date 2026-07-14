"""
Tests for VideoPlaybackThread.
Creates mock recording directories with metadata.json, rgb.avi, and depth.npy
to test loading and playback logic without RealSense hardware.
"""

import json
import os
import sys
import shutil
import tempfile

import numpy as np
import pytest
import cv2
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage

app = QApplication.instance() or QApplication(sys.argv)

from Vision.src.video_playback_thread import VideoPlaybackThread


def _create_mock_recording(
    base_dir,
    num_frames=10,
    width=640,
    height=480,
    fps=30,
    depth_scale=0.001,
    with_depth=True,
    with_raw_depth=True,
    format="stacked",
):
    """Helper to create a mock recording directory."""
    rec_dir = os.path.join(base_dir, "recording_20250101_120000")
    os.makedirs(rec_dir, exist_ok=True)

    # Create RGB video
    rgb_path = os.path.join(rec_dir, "rgb.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(rgb_path, fourcc, float(fps), (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Create depth data
    if format == "stacked":
        if with_depth:
            depth = np.random.randint(100, 5000, (num_frames, height, width), dtype=np.uint16)
            np.save(os.path.join(rec_dir, "depth.npy"), depth)
        if with_raw_depth:
            depth_raw = np.random.randint(100, 5000, (num_frames, height, width), dtype=np.uint16)
            np.save(os.path.join(rec_dir, "depth_raw.npy"), depth_raw)
    elif format == "individual":
        if with_depth:
            depth_dir = os.path.join(rec_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)
            for i in range(num_frames):
                depth = np.random.randint(100, 5000, (height, width), dtype=np.uint16)
                np.save(os.path.join(depth_dir, f"frame_{i:05d}.npy"), depth)
        if with_raw_depth:
            depth_raw_dir = os.path.join(rec_dir, "depth_raw")
            os.makedirs(depth_raw_dir, exist_ok=True)
            for i in range(num_frames):
                depth_raw = np.random.randint(100, 5000, (height, width), dtype=np.uint16)
                np.save(os.path.join(depth_raw_dir, f"frame_{i:05d}.npy"), depth_raw)

    # Create metadata
    metadata = {
        "depth_scale": depth_scale,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": num_frames,
        "duration_s": round(num_frames / fps, 2),
        "timestamp": "2025-01-01T12:00:00",
        "has_depth": with_depth,
        "has_raw_depth": with_raw_depth,
        "depth_format": format,
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return rec_dir


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_playback_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_recording_stacked(tmp_dir):
    """Recording with stacked depth format (preferred)."""
    return _create_mock_recording(tmp_dir, format="stacked")


@pytest.fixture
def mock_recording_individual(tmp_dir):
    """Recording with individual .npy files (legacy format)."""
    return _create_mock_recording(tmp_dir, format="individual")


@pytest.fixture
def mock_recording_rgb_only(tmp_dir):
    """Recording with RGB only, no depth."""
    return _create_mock_recording(tmp_dir, with_depth=False, with_raw_depth=False)


@pytest.fixture
def playback_thread(mock_recording_stacked):
    """VideoPlaybackThread with a valid stacked recording."""
    return VideoPlaybackThread(recording_dir=mock_recording_stacked)


# ═══════════════════════════════════════════════════════════════════════════════
# Instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def test_instantiation(playback_thread):
    assert playback_thread._running is False
    assert playback_thread._paused is False
    assert playback_thread._speed == 1.0
    assert playback_thread._loop is False


def test_initial_properties(playback_thread):
    assert playback_thread.total_frames == 0
    assert playback_thread.is_paused is False
    assert playback_thread.recording_dir != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Controls
# ═══════════════════════════════════════════════════════════════════════════════

def test_set_paused(playback_thread):
    playback_thread.set_paused(True)
    assert playback_thread.is_paused is True
    assert not playback_thread._pause_event.is_set()

    playback_thread.set_paused(False)
    assert playback_thread.is_paused is False
    assert playback_thread._pause_event.is_set()


def test_toggle_pause(playback_thread):
    assert playback_thread.is_paused is False
    playback_thread.toggle_pause()
    assert playback_thread.is_paused is True
    playback_thread.toggle_pause()
    assert playback_thread.is_paused is False


def test_set_speed(playback_thread):
    playback_thread.set_speed(2.0)
    assert playback_thread._speed == 2.0

    playback_thread.set_speed(0.1)  # Below minimum
    assert playback_thread._speed == 0.25

    playback_thread.set_speed(10.0)  # Above maximum
    assert playback_thread._speed == 4.0


def test_set_loop(playback_thread):
    playback_thread.set_loop(True)
    assert playback_thread._loop is True
    playback_thread.set_loop(False)
    assert playback_thread._loop is False


def test_stop_playback_when_not_running(playback_thread):
    """stop_playback is safe to call when not running."""
    playback_thread.stop_playback()  # Should not crash
    assert playback_thread._running is False


# ═══════════════════════════════════════════════════════════════════════════════
# Loading — Stacked format
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_recording_stacked(playback_thread):
    """Load a recording with stacked depth format."""
    result = playback_thread._load_recording()
    assert result is True
    assert playback_thread._total_frames == 10
    assert playback_thread._fps == 30
    assert playback_thread._depth_scale == 0.001
    assert playback_thread._depth_stack is not None
    assert playback_thread._depth_stack.shape == (10, 480, 640)
    assert playback_thread._depth_raw_stack is not None
    assert playback_thread._has_depth is True
    assert playback_thread._has_raw_depth is True


def test_load_recording_stacked_has_rgb(playback_thread):
    playback_thread._load_recording()
    assert playback_thread._rgb_capture is not None
    assert playback_thread._rgb_capture.isOpened()
    playback_thread._release_resources()


# ═══════════════════════════════════════════════════════════════════════════════
# Loading — Individual files (legacy)
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_recording_individual(mock_recording_individual):
    """Load a recording with individual .npy files (legacy format)."""
    thread = VideoPlaybackThread(recording_dir=mock_recording_individual)
    result = thread._load_recording()
    assert result is True
    assert thread._depth_stack is None  # Not stacked
    assert len(thread._depth_files) == 10
    assert len(thread._depth_raw_files) == 10
    assert thread._has_depth is True
    thread._release_resources()


def test_get_depth_frame_individual(mock_recording_individual):
    """_get_depth_frame loads individual .npy files correctly."""
    thread = VideoPlaybackThread(recording_dir=mock_recording_individual)
    thread._load_recording()
    frame = thread._get_depth_frame(0)
    assert frame is not None
    assert frame.shape == (480, 640)
    assert frame.dtype == np.uint16
    thread._release_resources()


# ═══════════════════════════════════════════════════════════════════════════════
# Loading — RGB only (no depth)
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_recording_rgb_only(mock_recording_rgb_only):
    """Load a recording with RGB only, no depth."""
    thread = VideoPlaybackThread(recording_dir=mock_recording_rgb_only)
    result = thread._load_recording()
    assert result is True
    assert thread._has_depth is False
    assert thread._has_raw_depth is False
    assert thread._depth_stack is None
    assert thread._depth_raw_stack is None
    assert len(thread._depth_files) == 0
    thread._release_resources()


def test_get_depth_frame_rgb_only(mock_recording_rgb_only):
    """_get_depth_frame returns None when no depth available."""
    thread = VideoPlaybackThread(recording_dir=mock_recording_rgb_only)
    thread._load_recording()
    assert thread._get_depth_frame(0) is None
    assert thread._get_depth_raw_frame(0) is None
    thread._release_resources()


# ═══════════════════════════════════════════════════════════════════════════════
# Depth frame access
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_depth_frame_stacked(playback_thread):
    playback_thread._load_recording()
    frame = playback_thread._get_depth_frame(0)
    assert frame is not None
    assert frame.shape == (480, 640)
    assert frame.dtype == np.uint16


def test_get_depth_frame_out_of_range(playback_thread):
    playback_thread._load_recording()
    assert playback_thread._get_depth_frame(999) is None


def test_get_depth_raw_frame_stacked(playback_thread):
    playback_thread._load_recording()
    frame = playback_thread._get_depth_raw_frame(5)
    assert frame is not None
    assert frame.shape == (480, 640)
    playback_thread._release_resources()


# ═══════════════════════════════════════════════════════════════════════════════
# Loading failures
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_missing_metadata(tmp_dir):
    """_load_recording returns False if metadata.json is missing."""
    rec_dir = os.path.join(tmp_dir, "empty_recording")
    os.makedirs(rec_dir, exist_ok=True)
    thread = VideoPlaybackThread(recording_dir=rec_dir)
    result = thread._load_recording()
    assert result is False


def test_load_missing_rgb(tmp_dir):
    """_load_recording returns False if rgb.avi is missing."""
    rec_dir = os.path.join(tmp_dir, "no_rgb")
    os.makedirs(rec_dir, exist_ok=True)
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump({"fps": 30, "frame_count": 10, "depth_scale": 0.001}, f)
    thread = VideoPlaybackThread(recording_dir=rec_dir)
    result = thread._load_recording()
    assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# Signal definitions
# ═══════════════════════════════════════════════════════════════════════════════

def test_signals_defined(playback_thread):
    assert hasattr(playback_thread, "frame_pair_ready")
    assert hasattr(playback_thread, "distance_info_ready")
    assert hasattr(playback_thread, "obstacles_ready")
    assert hasattr(playback_thread, "navigation_ready")
    assert hasattr(playback_thread, "light_mode_changed")
    assert hasattr(playback_thread, "error")
    assert hasattr(playback_thread, "playback_progress")
    assert hasattr(playback_thread, "playback_finished")


# ═══════════════════════════════════════════════════════════════════════════════
# BGR to QImage conversion
# ═══════════════════════════════════════════════════════════════════════════════

def test_bgr_to_qimage(playback_thread):
    bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    qimg = playback_thread._bgr_to_qimage(bgr)
    assert qimg.width() == 640
    assert qimg.height() == 480
    assert not qimg.isNull()


def test_bgr_to_qimage_grayscale(playback_thread):
    gray = np.full((480, 640), 128, dtype=np.uint8)
    qimg = playback_thread._bgr_to_qimage(gray)
    assert qimg.width() == 640
    assert qimg.height() == 480


def test_empty_depth_qimage_cached(playback_thread):
    q1 = playback_thread._get_empty_depth_qimage((480, 640))
    q2 = playback_thread._get_empty_depth_qimage((480, 640))
    assert q1 is q2


def test_empty_depth_qimage_different_size(playback_thread):
    q1 = playback_thread._get_empty_depth_qimage((480, 640))
    q2 = playback_thread._get_empty_depth_qimage((240, 320))
    assert q1 is not q2
    assert q2.width() == 320
    assert q2.height() == 240


# ═══════════════════════════════════════════════════════════════════════════════
# Depth threshold propagation
# ═══════════════════════════════════════════════════════════════════════════════

def test_set_depth_thresholds(playback_thread):
    playback_thread.set_depth_thresholds(0.5, 4.0)
    assert playback_thread._depth_min_m == 0.5
    assert playback_thread._depth_max_m == 4.0


def test_set_depth_thresholds_invalid(playback_thread):
    playback_thread.set_depth_thresholds(0.5, 4.0)
    playback_thread.set_depth_thresholds(5.0, 1.0)  # Invalid
    assert playback_thread._depth_min_m == 0.5  # Unchanged
    assert playback_thread._depth_max_m == 4.0


def test_set_depth_thresholds_propagates_to_processor(playback_thread, mock_recording_stacked):
    """set_depth_thresholds propagates to processor's depth stage."""
    from Vision.inc.detection_config import DetectionConfig
    from Vision.src.frame_processor import FrameProcessor

    config = DetectionConfig()
    processor = FrameProcessor(config)
    thread = VideoPlaybackThread(
        recording_dir=mock_recording_stacked, processor=processor
    )
    thread.set_depth_thresholds(0.5, 4.0)
    depth_stage = processor.get_stage("DepthProcessingStage")
    assert depth_stage._depth_min_m == 0.5
    assert depth_stage._depth_max_m == 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# Resource cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def test_release_resources(playback_thread):
    playback_thread._load_recording()
    assert playback_thread._rgb_capture is not None
    playback_thread._release_resources()
    assert playback_thread._rgb_capture is None
    assert playback_thread._depth_stack is None
    assert playback_thread._depth_raw_stack is None


def test_release_resources_without_loading(playback_thread):
    """release_resources is safe even if nothing was loaded."""
    playback_thread._release_resources()  # Should not crash
