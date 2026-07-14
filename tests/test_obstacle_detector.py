"""
Tests for ObstacleDetector.
Uses synthetic frames — no camera hardware required.
"""

import threading

import numpy as np
import pytest

from Vision.src.obstacle_detector import ObstacleDetector


def make_color_frame(h=480, w=640):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def make_depth_frame(h=480, w=640, default_m=5.0):
    depth_m = np.full((h, w), default_m, dtype=np.float32)
    return (depth_m / 0.001).astype(np.uint16)


def make_depth_with_obstacle(h=480, w=640, default_m=10.0,
                              obs_m=2.0, obs_y=(200, 300), obs_x=(280, 360)):
    """Create a depth frame with a single obstacle region."""
    depth_m = np.full((h, w), default_m, dtype=np.float32)
    depth_m[obs_y[0]:obs_y[1], obs_x[0]:obs_x[1]] = obs_m
    return (depth_m / 0.001).astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════════════════
# Instantiation & Parameters
# ═══════════════════════════════════════════════════════════════════════════════

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


def test_max_area_ratio_param():
    det = ObstacleDetector(max_area_ratio=0.5)
    assert det.max_area_ratio == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases: None / Empty Inputs
# ═══════════════════════════════════════════════════════════════════════════════

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


def test_detect_none_color_only():
    det = ObstacleDetector()
    depth = make_depth_frame()
    annotated, obstacles = det.detect(None, depth)
    assert annotated is None
    assert obstacles == []


def test_detect_none_depth_only():
    det = ObstacleDetector()
    color = make_color_frame()
    annotated, obstacles = det.detect(color, None)
    assert obstacles == []


def test_detect_all_zero_depth():
    """All-zero depth frame → no valid obstacles (below min_distance)."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth = np.zeros((480, 640), dtype=np.uint16)
    _, obstacles = det.detect(color, depth)
    assert obstacles == []


# ═══════════════════════════════════════════════════════════════════════════════
# Obstacle Detection & Zones
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_with_obstacle():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=1.5)
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
    depth_raw = make_depth_with_obstacle(obs_m=2.0, obs_x=(280, 360))
    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "center"


def test_zone_left():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0, obs_x=(20, 100))
    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "left"


def test_zone_right():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0, obs_x=(540, 620))
    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert obstacles[0]["zone"] == "right"


def test_multiple_obstacles_different_zones():
    """Multiple obstacles in different zones should all be detected."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 20:100] = 1.5      # left
    depth_m[200:300, 280:360] = 1.5     # center
    depth_m[200:300, 540:620] = 1.5     # right
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    _, obstacles = det.detect(color, depth_raw)
    zones = {obs["zone"] for obs in obstacles}
    assert "left" in zones
    assert "center" in zones
    assert "right" in zones


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering: min_area, max_area_ratio, distance bounds
# ═══════════════════════════════════════════════════════════════════════════════

def test_min_area_filter():
    """Obstacle smaller than min_area → filtered out."""
    det = ObstacleDetector(min_area=50000)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0, obs_y=(200, 210), obs_x=(280, 290))
    _, obstacles = det.detect(color, depth_raw)
    assert obstacles == []


def test_max_area_ratio_filter():
    """Obstacle covering > max_area_ratio of frame → filtered as scene-wide."""
    det = ObstacleDetector(min_area=100, max_area_ratio=0.10)
    color = make_color_frame()
    # Obstacle covers 50% of frame
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[0:480, 0:320] = 1.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    _, obstacles = det.detect(color, depth_raw)
    # With max_area_ratio=0.10, 50% coverage should be filtered
    assert all(obs["area_px"] <= 0.10 * 480 * 640 for obs in obstacles)


def test_obstacle_beyond_max_distance():
    """Obstacle beyond max_distance_m → not detected."""
    det = ObstacleDetector(max_distance_m=1.0, min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(default_m=10.0, obs_m=3.0)
    _, obstacles = det.detect(color, depth_raw)
    assert obstacles == []


def test_obstacle_within_distance_bounds():
    """Obstacle within [min_distance, max_distance] → detected."""
    det = ObstacleDetector(max_distance_m=5.0, min_distance_m=0.3, min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(default_m=10.0, obs_m=2.0)
    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    assert all(0.3 <= obs["distance_m"] <= 5.0 for obs in obstacles)


# ═══════════════════════════════════════════════════════════════════════════════
# Priority & Distance Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def test_priority_no_division_by_zero():
    """Distance = 0 should not cause infinity/nan issues (priority removed in v4)."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 0.0
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    _, obstacles = det.detect(color, depth_raw)
    # priority is no longer computed in ObstacleDetector (FusionStage computes its own)
    for obs in obstacles:
        assert "priority" not in obs
        assert isinstance(obs["distance_m"], float)


def test_distance_inverse_to_depth():
    """Closer obstacle → smaller distance value."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_m_far = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m_far[200:300, 280:360] = 3.0
    _, obstacles_far = det.detect(color, (depth_m_far / 0.001).astype(np.uint16))

    depth_m_near = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m_near[200:300, 280:360] = 0.5
    _, obstacles_near = det.detect(color, (depth_m_near / 0.001).astype(np.uint16))

    if obstacles_far and obstacles_near:
        assert obstacles_near[0]["distance_m"] < obstacles_far[0]["distance_m"]


def test_distance_accuracy():
    """Detected distance should be close to synthetic depth value."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(default_m=10.0, obs_m=1.5)
    _, obstacles = det.detect(color, depth_raw)
    if obstacles:
        # 5th percentile of a uniform 1.5m region should be ~1.5m
        assert abs(obstacles[0]["distance_m"] - 1.5) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# Frame Handling (no copy regression test)
# ═══════════════════════════════════════════════════════════════════════════════

def test_last_detections_updated():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0)
    _, obstacles = det.detect(color, depth_raw)
    assert det.last_detections == obstacles


def test_detect_returns_frame():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=0.5)
    frame, _ = det.detect(color, depth_raw)
    assert frame.shape == color.shape


def test_detect_returns_same_frame_object():
    """Detector should NOT copy the color frame — returns the same object."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=0.5)
    frame, _ = det.detect(color, depth_raw)
    assert frame is color, "detect() should return the same color_frame object (no copy)"


def test_detect_does_not_modify_color_frame():
    """Detector should not draw on or modify the color frame."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    original = color.copy()
    depth_raw = make_depth_with_obstacle(obs_m=0.5)
    det.detect(color, depth_raw)
    assert np.array_equal(color, original), "color frame should be unmodified"


# ═══════════════════════════════════════════════════════════════════════════════
# Output Format Contract
# ═══════════════════════════════════════════════════════════════════════════════

def test_output_format_contract():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0)
    _, obstacles = det.detect(color, depth_raw)
    assert len(obstacles) >= 1
    obs = obstacles[0]
    assert isinstance(obs["bbox"], list)
    assert len(obs["bbox"]) == 4
    assert isinstance(obs["distance_m"], float)
    assert isinstance(obs["zone"], str)
    assert isinstance(obs["area_px"], int)
    assert "object_class" in obs
    assert obs["object_class"] == "obstacle"
    # priority is no longer in ObstacleDetector output (FusionStage computes its own)
    assert "priority" not in obs


def test_bbox_format_xywh():
    """Obstacle bbox should be [x, y, w, h], not [x1, y1, x2, y2]."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0, obs_y=(200, 300), obs_x=(280, 360))
    _, obstacles = det.detect(color, depth_raw)
    if obstacles:
        x, y, w, h = obstacles[0]["bbox"]
        # w and h should be positive, and x+w should be <= frame width
        assert w > 0
        assert h > 0
        assert x + w <= 640
        assert y + h <= 480


# ═══════════════════════════════════════════════════════════════════════════════
# Buffer Reuse & Shape Changes
# ═══════════════════════════════════════════════════════════════════════════════

def test_depth_buffer_reuse():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0)
    det.detect(color, depth_raw)
    buf1 = det._depth_buffer
    det.detect(color, depth_raw)
    buf2 = det._depth_buffer
    assert buf1 is buf2, "Depth buffer should be reused across calls"


def test_depth_buffer_resize_on_shape_change():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame(480, 640)
    depth_raw = np.full((480, 640), 5000, dtype=np.uint16)
    det.detect(color, depth_raw)
    buf1 = det._depth_buffer
    color_small = make_color_frame(240, 320)
    depth_raw_small = np.full((240, 320), 5000, dtype=np.uint16)
    det.detect(color_small, depth_raw_small)
    buf2 = det._depth_buffer
    assert buf1 is not buf2, "Buffer should realloc on shape change"
    assert buf2.shape == (240, 320)


def test_different_depth_scale():
    """Detector should work with non-default depth_scale."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    # depth_scale = 0.002 (each raw unit = 2mm)
    depth_m = np.full((480, 640), 10.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 1.5
    depth_raw = (depth_m / 0.002).astype(np.uint16)
    _, obstacles = det.detect(color, depth_raw, depth_scale=0.002)
    if obstacles:
        assert abs(obstacles[0]["distance_m"] - 1.5) < 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# Thread Safety
# ═══════════════════════════════════════════════════════════════════════════════

def test_thread_safety_last_detections():
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0)

    errors = []

    def writer():
        try:
            for _ in range(50):
                det.detect(color, depth_raw)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                _ = det.last_detections
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Last Detections Property
# ═══════════════════════════════════════════════════════════════════════════════

def test_last_detections_returns_copy():
    """last_detections getter should return a copy, not the internal list."""
    det = ObstacleDetector(min_area=100)
    color = make_color_frame()
    depth_raw = make_depth_with_obstacle(obs_m=2.0)
    det.detect(color, depth_raw)
    d1 = det.last_detections
    d1.clear()  # Mutate the copy
    d2 = det.last_detections
    assert len(d2) > 0, "Mutating the returned list should not affect internal state"


if __name__ == "__main__":
    import sys
    print("=== ObstacleDetector Tests ===\n")
    tests = [
        ("instantiation", test_instantiation),
        ("custom params", test_custom_params),
        ("max_area_ratio param", test_max_area_ratio_param),
        ("detect no obstacles", test_detect_no_obstacles),
        ("detect none inputs", test_detect_none_inputs),
        ("detect none color only", test_detect_none_color_only),
        ("detect none depth only", test_detect_none_depth_only),
        ("detect all zero depth", test_detect_all_zero_depth),
        ("detect with obstacle", test_detect_with_obstacle),
        ("zone center", test_zone_center),
        ("zone left", test_zone_left),
        ("zone right", test_zone_right),
        ("multiple obstacles different zones", test_multiple_obstacles_different_zones),
        ("min area filter", test_min_area_filter),
        ("max area ratio filter", test_max_area_ratio_filter),
        ("obstacle beyond max distance", test_obstacle_beyond_max_distance),
        ("obstacle within distance bounds", test_obstacle_within_distance_bounds),
        ("priority no division by zero", test_priority_no_division_by_zero),
        ("priority inverse to distance", test_priority_inverse_to_distance),
        ("distance accuracy", test_distance_accuracy),
        ("last detections updated", test_last_detections_updated),
        ("detect returns frame", test_detect_returns_frame),
        ("detect returns same frame object", test_detect_returns_same_frame_object),
        ("detect does not modify color frame", test_detect_does_not_modify_color_frame),
        ("output format contract", test_output_format_contract),
        ("bbox format xywh", test_bbox_format_xywh),
        ("depth buffer reuse", test_depth_buffer_reuse),
        ("depth buffer resize on shape change", test_depth_buffer_resize_on_shape_change),
        ("different depth scale", test_different_depth_scale),
        ("thread safety last_detections", test_thread_safety_last_detections),
        ("last detections returns copy", test_last_detections_returns_copy),
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
