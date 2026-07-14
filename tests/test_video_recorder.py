"""
Tests for VideoRecorder.
Tests non-blocking API, metadata, and depth saving without RealSense hardware.
"""

import json
import os
import sys
import shutil
import tempfile

import numpy as np
import pytest
import cv2

from Vision.src.video_recorder import VideoRecorder


@pytest.fixture
def tmp_rec_dir():
    """Create a temporary directory for recording output."""
    d = tempfile.mkdtemp(prefix="test_recording_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def recorder(tmp_rec_dir):
    """Create a VideoRecorder instance (no RealSense required)."""
    return VideoRecorder(save_dir=tmp_rec_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Instantiation
# ═══════════════════════════════════════════════════════════════════════════════

def test_instantiation():
    """VideoRecorder can be instantiated without pyrealsense2."""
    rec = VideoRecorder()
    assert rec.is_recording is False
    assert rec.frame_count == 0
    assert rec.depth_scale == 0.001


def test_instantiation_custom_dir():
    rec = VideoRecorder(save_dir="/tmp/test_recordings")
    assert rec._save_dir == "/tmp/test_recordings"


def test_start_pipeline_fails_without_realsense():
    """start_pipeline returns False when pyrealsense2 is not available."""
    import Vision.src.video_recorder as mod
    original = mod.rs
    mod.rs = None
    try:
        rec = VideoRecorder()
        assert rec.start_pipeline() is False
    finally:
        mod.rs = original


# ═══════════════════════════════════════════════════════════════════════════════
# Non-blocking API: start_recording / record_frame / stop_recording
# ═══════════════════════════════════════════════════════════════════════════════

def test_start_recording_creates_dir(recorder, tmp_rec_dir):
    """start_recording creates a timestamped subdirectory."""
    rec_dir = recorder.start_recording()
    assert os.path.isdir(rec_dir)
    assert rec_dir.startswith(tmp_rec_dir)
    assert recorder.is_recording is True
    recorder.stop_recording()


def test_start_recording_custom_dir():
    """start_recording with custom output_dir."""
    with tempfile.TemporaryDirectory() as d:
        rec = VideoRecorder()
        rec_dir = rec.start_recording(output_dir=d)
        assert rec_dir.startswith(d)
        assert os.path.isdir(rec_dir)
        rec.stop_recording()


def test_record_frame_buffers_rgb(recorder):
    """record_frame writes RGB to video writer."""
    rec_dir = recorder.start_recording()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(frame)
    assert recorder.frame_count == 1
    recorder.stop_recording()
    assert os.path.exists(os.path.join(rec_dir, "rgb.avi"))


def test_record_frame_buffers_depth(recorder):
    """record_frame writes depth frames directly to disk."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    recorder.record_frame(rgb, depth_filtered=depth, depth_raw=depth)
    assert recorder.frame_count == 1
    
    # Check files exist on disk
    filtered_path = os.path.join(rec_dir, "depth", "frame_00000.npy")
    raw_path = os.path.join(rec_dir, "depth_raw", "frame_00000.npy")
    assert os.path.exists(filtered_path)
    assert os.path.exists(raw_path)
    recorder.stop_recording()


def test_record_frame_no_depth(recorder):
    """record_frame works with RGB only (no depth)."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(rgb)
    assert recorder.frame_count == 1
    assert len(os.listdir(os.path.join(rec_dir, "depth"))) == 0
    assert len(os.listdir(os.path.join(rec_dir, "depth_raw"))) == 0
    recorder.stop_recording()


def test_record_frame_ignored_when_not_recording(recorder):
    """record_frame does nothing if not recording."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(frame)  # Should not crash
    assert recorder.frame_count == 0


def test_stop_recording_returns_dir(recorder):
    """stop_recording returns the recording directory path."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(rgb)
    result = recorder.stop_recording()
    assert result == rec_dir
    assert recorder.is_recording is False


def test_stop_recording_saves_rgb_avi(recorder):
    """stop_recording saves rgb.avi video file."""
    rec_dir = recorder.start_recording()
    for _ in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recorder.record_frame(frame)
    recorder.stop_recording()
    rgb_path = os.path.join(rec_dir, "rgb.avi")
    assert os.path.exists(rgb_path)
    assert os.path.getsize(rgb_path) > 0


def test_stop_recording_saves_depth_npy(recorder):
    """stop_recording saves depth frames as individual .npy files in subdirectories."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    for _ in range(3):
        recorder.record_frame(rgb, depth_filtered=depth, depth_raw=depth)
    recorder.stop_recording()

    for idx in range(3):
        depth_path = os.path.join(rec_dir, "depth", f"frame_{idx:05d}.npy")
        depth_raw_path = os.path.join(rec_dir, "depth_raw", f"frame_{idx:05d}.npy")
        assert os.path.exists(depth_path)
        assert os.path.exists(depth_raw_path)

        depth_arr = np.load(depth_path)
        assert depth_arr.shape == (480, 640)
        assert depth_arr.dtype == np.uint16


def test_stop_recording_saves_metadata(recorder):
    """stop_recording saves metadata.json with correct fields."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 500, dtype=np.uint16)
    recorder.record_frame(rgb, depth_filtered=depth, depth_raw=depth)
    recorder.record_frame(rgb, depth_filtered=depth, depth_raw=depth)
    recorder.stop_recording()

    meta_path = os.path.join(rec_dir, "metadata.json")
    assert os.path.exists(meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["frame_count"] == 2
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert meta["fps"] == 30
    assert meta["has_depth"] is True
    assert meta["has_raw_depth"] is True
    assert meta["depth_format"] == "individual_npy"
    assert "depth_scale" in meta
    assert "timestamp" in meta


def test_stop_recording_no_depth_saves_no_depth_files(recorder):
    """stop_recording with RGB-only does not save depth files."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(rgb)
    recorder.stop_recording()

    assert len(os.listdir(os.path.join(rec_dir, "depth"))) == 0
    assert len(os.listdir(os.path.join(rec_dir, "depth_raw"))) == 0

    with open(os.path.join(rec_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta["has_depth"] is False
    assert meta["has_raw_depth"] is False


def test_stop_recording_called_twice_is_safe(recorder):
    """Calling stop_recording twice does not crash."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    recorder.record_frame(rgb)
    recorder.stop_recording()
    recorder.stop_recording()  # Should not crash


def test_start_stop_multiple_sessions(recorder):
    """Multiple start/stop recording sessions work independently."""
    for i in range(3):
        rec_dir = recorder.start_recording()
        rgb = np.full((480, 640, 3), i * 50, dtype=np.uint8)
        recorder.record_frame(rgb)
        recorder.stop_recording()
        assert os.path.exists(os.path.join(rec_dir, "rgb.avi"))


def test_record_frame_clones_depth(recorder):
    """record_frame copies depth arrays to disk immediately (not reference)."""
    rec_dir = recorder.start_recording()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    recorder.record_frame(rgb, depth_filtered=depth, depth_raw=depth)
    # Modify original — saved frame should be unaffected
    depth[:] = 0
    recorder.stop_recording()
    
    saved_arr = np.load(os.path.join(rec_dir, "depth", "frame_00000.npy"))
    assert saved_arr[0, 0] == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# RGB video validity
# ═══════════════════════════════════════════════════════════════════════════════

def test_rgb_avi_is_readable_video(recorder):
    """Saved rgb.avi can be opened and read by OpenCV."""
    rec_dir = recorder.start_recording()
    for i in range(10):
        frame = np.full((480, 640, 3), i * 25, dtype=np.uint8)
        recorder.record_frame(frame)
    recorder.stop_recording()

    cap = cv2.VideoCapture(os.path.join(rec_dir, "rgb.avi"))
    assert cap.isOpened()
    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1
    cap.release()
    assert frame_count == 10


def test_rgb_avi_frame_count_matches(recorder):
    """Number of readable frames matches frame_count in metadata."""
    rec_dir = recorder.start_recording()
    for _ in range(7):
        recorder.record_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    recorder.stop_recording()

    cap = cv2.VideoCapture(os.path.join(rec_dir, "rgb.avi"))
    actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert actual == 7
