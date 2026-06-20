"""
Standalone test for FrameProcessor pipeline.
Runs without camera hardware — uses synthetic frames.
"""

import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["Vision/src", "Vision/inc", "GUI/inc"]:
    p = os.path.join(BASE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from detection_config import DetectionConfig
from frame_processor import (
    DepthProcessingStage,
    FrameData,
    FrameProcessor,
    FusionStage,
    PipelineStage,
    YOLODetectionStage,
)


def make_synthetic_frames():
    h, w = 480, 640
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth_m = np.full((h, w), 3.0, dtype=np.float32)
    depth_m[h // 2 - 60 : h // 2 + 60, w // 2 - 60 : w // 2 + 60] = 1.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    return rgb, depth_raw


def test_imports():
    assert FrameProcessor is not None
    assert FrameData is not None
    assert PipelineStage is not None
    print("  PASS  test_imports")


def test_instantiation():
    p = FrameProcessor(DetectionConfig())
    assert len(p.stages) == 1
    assert p.stages[0].name == "DepthProcessingStage"
    print("  PASS  test_instantiation")


def test_process_with_depth():
    p = FrameProcessor(DetectionConfig())
    rgb, depth = make_synthetic_frames()
    r = p.process(rgb, depth)
    assert r.has_depth()
    assert r.depth_colormap is not None
    assert r.depth_colormap.shape == (480, 640, 3)
    assert "timestamp" in r.metadata
    print("  PASS  test_process_with_depth")


def test_process_without_depth():
    p = FrameProcessor(DetectionConfig())
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    r = p.process(rgb)
    assert r.depth_frame is None
    assert not r.has_depth()
    print("  PASS  test_process_without_depth")


def test_stage_management():
    p = FrameProcessor(DetectionConfig())
    yolo = YOLODetectionStage()
    p.add_stage(yolo)
    assert len(p.stages) == 2
    assert p.get_stage("YOLODetectionStage") is yolo
    p.set_stage_enabled("YOLODetectionStage", False)
    assert not yolo.enabled
    p.set_stage_enabled("YOLODetectionStage", True)
    assert yolo.enabled
    p.remove_stage("YOLODetectionStage")
    assert len(p.stages) == 1
    print("  PASS  test_stage_management")


def test_threshold_update():
    p = FrameProcessor(DetectionConfig())
    p.set_depth_thresholds(0.5, 4.0)
    print("  PASS  test_threshold_update")


def test_latency_report():
    p = FrameProcessor(DetectionConfig())
    rgb, depth = make_synthetic_frames()
    p.process(rgb, depth)
    report = p.get_latency_report()
    assert "DepthProcessingStage" in report
    assert report["DepthProcessingStage"] > 0
    print(f"  PASS  test_latency_report ({report['DepthProcessingStage']:.2f} ms)")


def test_custom_stage():
    class MultiplyStage(PipelineStage):
        def process(self, data):
            data.rgb_frame = np.clip(
                data.rgb_frame.astype(np.int32) * 2, 0, 255
            ).astype(np.uint8)
            data.metadata["x2"] = True
            return data

    p = FrameProcessor(DetectionConfig())
    p.set_stage_enabled("DepthProcessingStage", False)
    rgb, depth = make_synthetic_frames()
    p.add_stage(MultiplyStage("M"))
    r = p.process(rgb, depth)
    assert r.metadata.get("x2")
    assert r.rgb_frame.mean() == 255.0
    print("  PASS  test_custom_stage")


# ═══════════════════════════════════════════════════════════════════════════════
# FusionStage Tests
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import List

@dataclass
class MockDetection:
    """Mock Detection dataclass matching yolowrapper.Detection format."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


def _make_frame_data(detections=None, obstacles=None):
    """Helper to create FrameData with mock detections and obstacles."""
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.detections = detections or []
    data.obstacles = obstacles or []
    return data


def test_fusion_matching():
    """YOLO box fully covers depth blob → class should be assigned."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 2.0, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    assert len(result.fused_output) == 1
    assert result.fused_output[0]["object_class"] == "person"
    print("  PASS  test_fusion_matching")


def test_fusion_no_match():
    """No YOLO overlap → falls back to 'obstacle'."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[500, 500, 600, 600])
    obs = {"bbox": [10, 10, 50, 50], "distance_m": 2.0, "zone": "left", "area_px": 2500, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    assert result.fused_output[0]["object_class"] == "obstacle"
    print("  PASS  test_fusion_no_match")


def test_fusion_priority_person_close():
    """Person < danger_distance → priority 0 (STOP)."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 0.8, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 0
    assert result.fused_output[0]["action"] == "STOP"
    print("  PASS  test_fusion_priority_person_close")


def test_fusion_priority_obstacle_close():
    """Non-person obstacle < danger_distance → priority 1."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=1, class_name="chair", confidence=0.8, bbox=[100, 100, 300, 400])
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 0.5, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 1
    assert result.fused_output[0]["action"] is None
    print("  PASS  test_fusion_priority_obstacle_close")


def test_fusion_empty_inputs():
    """Empty detections and obstacles → empty fused_output."""
    fusion = FusionStage()
    data = _make_frame_data(detections=[], obstacles=[])
    result = fusion.process(data)
    assert result.fused_output == []
    print("  PASS  test_fusion_empty_inputs")


def test_fusion_bbox_format_xyxy():
    """Output bbox should be [x1, y1, x2, y2], not [x, y, w, h]."""
    fusion = FusionStage()
    obs = {"bbox": [100, 200, 50, 60], "distance_m": 1.5, "zone": "right", "area_px": 3000, "priority": 1.0}
    data = _make_frame_data(detections=[], obstacles=[obs])
    result = fusion.process(data)
    bbox = result.fused_output[0]["bbox"]
    # xyxy: [x1, y1, x1+w, y1+h] = [100, 200, 150, 260]
    assert bbox == [100, 200, 150, 260], f"Expected [100, 200, 150, 260], got {bbox}"
    print("  PASS  test_fusion_bbox_format_xyxy")


def test_fusion_overlap_ratio_with_area_px():
    """When area_px is provided, it should be used instead of AABB area."""
    fusion = FusionStage()
    # Depth blob: L-shape, bbox [0,0,10,10] (AABB=100), but contour area is 60
    # YOLO box fully covers: [0,0,10,10]
    # With AABB: ratio = 100/100 = 1.0
    # With area_px: ratio = 60/60 = 1.0 (same here since YOLO covers all)
    # More interesting: YOLO partially covers [0,0,5,10] (50px intersection)
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 5, 10])
    obs = {"bbox": [0, 0, 10, 10], "distance_m": 1.5, "zone": "center", "area_px": 60, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    # intersection=50, area_px=60 → ratio=0.833 > 0.5 → should match
    assert result.fused_output[0]["object_class"] == "person"
    print("  PASS  test_fusion_overlap_ratio_with_area_px")


def test_fusion_config_thresholds():
    """Priority should use DetectionConfig.danger_distance, not hardcoded 1.0."""
    config = DetectionConfig()
    config.danger_distance = 2.0  # Non-default
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    # Distance 1.5m: above old hardcoded 1.0, below new config 2.0
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 1.5, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    # With danger_distance=2.0, person at 1.5m should be priority 0
    assert result.fused_output[0]["priority"] == 0, f"Expected 0, got {result.fused_output[0]['priority']}"
    assert result.fused_output[0]["action"] == "STOP"
    print("  PASS  test_fusion_config_thresholds")


if __name__ == "__main__":
    print("=== FrameProcessor Tests ===\n")
    tests = [
        ("imports", test_imports),
        ("instantiation", test_instantiation),
        ("process with depth", test_process_with_depth),
        ("process without depth", test_process_without_depth),
        ("stage management", test_stage_management),
        ("threshold update", test_threshold_update),
        ("latency report", test_latency_report),
        ("custom stage", test_custom_stage),
        ("fusion matching", test_fusion_matching),
        ("fusion no match", test_fusion_no_match),
        ("fusion priority person close", test_fusion_priority_person_close),
        ("fusion priority obstacle close", test_fusion_priority_obstacle_close),
        ("fusion empty inputs", test_fusion_empty_inputs),
        ("fusion bbox format xyxy", test_fusion_bbox_format_xyxy),
        ("fusion overlap with area_px", test_fusion_overlap_ratio_with_area_px),
        ("fusion config thresholds", test_fusion_config_thresholds),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback

            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)
