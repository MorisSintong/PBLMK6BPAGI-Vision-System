"""
Tests for ObstacleDetector.
Uses synthetic frames — no camera hardware required.
"""

import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["Vision/src", "Vision/inc"]:
    p = os.path.join(BASE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from obstacle_detector import ObstacleDetector


def make_color_frame(h=480, w=640):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def make_depth_frame(h=480, w=640, default_m=5.0):
    depth_m = np.full((h, w), default_m, dtype=np.float32)
    return (depth_m / 0.001).astype(np.uint16)


def test_instantiation():
    det = ObstacleDetector()
    assert det.max_distance_m == 5.0
    assert det.min_distance_m == 0.3
    assert det.min_area == 800
    assert det.last_detections == []


def test_custom_params():
    det = ObstacleDetector(max_distance_m=3.0, min_distance_m=0.5, min_area=500)
    assert det.max_distance_m == 3.0
    assert det.min_distance_m == 0.5
    assert det.min_area == 500


def test_detect_no_obstacles():
    det = ObstacleDetector()
    color = make_color_frame()
    depth = make_depth_frame(default_m=10.0)
    annotated, obstacles = det.detect(color, depth)
    assert annotated is not None
    assert obstacles == []


def test_detect_none_inputs():
    det = ObstacleDetector()
    annotated, obstacles = det.detect(None, None)
    assert annotated is None
    assert obstacles == []


def test_detect_with_obstacle():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth = make_depth_frame(default_m=10.0)
    depth_raw = (depth / 0.001).astype(np.uint16)
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 1.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    annotated, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    obs = obstacles[0]
    assert "bbox" in obs
    assert "distance_m" in obs
    assert "zone" in obs
    assert "area_px" in obs
    assert obs["zone"] in ("left", "center", "right")
    assert 0.3 <= obs["distance_m"] <= 5.0


def test_zone_center():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "center"


def test_zone_left():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 20:100] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "left"


def test_zone_right():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 540:620] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "right"


def test_min_area_filter():
    det = ObstacleDetector(min_area=50000)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:210, 280:290] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert obstacles == []


def test_priority_no_division_by_zero():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 0.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    annotated, obstacles = det.detect(color, depth_raw)
    for obs in obstacles:
        assert isinstance(obs["priority"], float)
        assert obs["priority"] <= 100.0


def test_last_detections_updated():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert det.last_detections == obstacles


def test_annotated_frame_has_status():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 0.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    annotated, _ = det.detect(color, depth_raw)
    assert annotated.shape == color.shape


def test_output_format_contract():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 2.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    obs = obstacles[0]
    assert isinstance(obs["bbox"], list)
    assert len(obs["bbox"]) == 4
    assert isinstance(obs["distance_m"], float)
    assert isinstance(obs["zone"], str)
    assert isinstance(obs["priority"], float)
    assert isinstance(obs["area_px"], int)


if __name__ == "__main__":
    print("=== ObstacleDetector Tests ===\n")
    tests = [
        ("instantiation", test_instantiation),
        ("custom params", test_custom_params),
        ("detect no obstacles", test_detect_no_obstacles),
        ("detect none inputs", test_detect_none_inputs),
        ("detect with obstacle", test_detect_with_obstacle),
        ("zone center", test_zone_center),
        ("zone left", test_zone_left),
        ("zone right", test_zone_right),
        ("min area filter", test_min_area_filter),
        ("priority no division by zero", test_priority_no_division_by_zero),
        ("last detections updated", test_last_detections_updated),
        ("annotated frame has status", test_annotated_frame_has_status),
        ("output format contract", test_output_format_contract),
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
