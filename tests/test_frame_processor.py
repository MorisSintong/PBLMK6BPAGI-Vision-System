"""
Standalone test for FrameProcessor pipeline.
Runs without camera hardware — uses synthetic frames.
"""

import os
import sys
import time

import numpy as np

from Vision.inc.detection_config import DetectionConfig
from Vision.src.frame_processor import (
    DepthProcessingStage,
    FrameData,
    FrameProcessor,
    FusionStage,
    NavigationStage,
    PipelineStage,
    YOLODetectionStage,
    VisualAnnotationStage
)


def make_synthetic_frames():
    h, w = 480, 640
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth_m = np.full((h, w), 3.0, dtype=np.float32)
    depth_m[h // 2 - 60 : h // 2 + 60, w // 2 - 60 : w // 2 + 60] = 1.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    return rgb, depth_raw


# ═══════════════════════════════════════════════════════════════════════════════
# FrameData Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_frame_data_defaults():
    """FrameData should have sensible defaults."""
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8))
    assert data.depth_frame is None
    assert data.depth_frame_raw is None
    assert data.depth_colormap is None
    assert data.depth_colormap_raw is None
    assert data.depth_scale == 0.001
    assert data.obstacles == []
    assert data.detections == []
    assert data.fused_output == []
    assert data.metadata == {}
    assert data.errors == []


def test_frame_data_has_depth():
    assert FrameData(rgb_frame=np.zeros((10, 10, 3), dtype=np.uint8),
                     depth_frame=np.zeros((10, 10), dtype=np.uint16)).has_depth()
    assert not FrameData(rgb_frame=np.zeros((10, 10, 3), dtype=np.uint8)).has_depth()


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineStage Base Class Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_stage_disabled_skips_processing():
    """Disabled stage should pass data through unchanged."""
    class MutateStage(PipelineStage):
        def process(self, data):
            data.metadata["mutated"] = True
            return data

    stage = MutateStage("Mutate")
    stage.enabled = False
    data = FrameData(rgb_frame=np.zeros((10, 10, 3), dtype=np.uint8))
    result = stage._measure(data)
    assert "mutated" not in result.metadata


def test_pipeline_stage_measure_sets_latency():
    """_measure should record latency > 0."""
    class SleepStage(PipelineStage):
        def process(self, data):
            time.sleep(0.005)
            return data

    stage = SleepStage("Sleep")
    data = FrameData(rgb_frame=np.zeros((10, 10, 3), dtype=np.uint8))
    stage._measure(data)
    assert stage.last_latency_ms > 0


def test_pipeline_stage_exception_captured():
    """Exception in process() should not crash; error added to data.errors."""
    class CrashStage(PipelineStage):
        def process(self, data):
            raise ValueError("boom")

    stage = CrashStage("Crash")
    data = FrameData(rgb_frame=np.zeros((10, 10, 3), dtype=np.uint8))
    result = stage._measure(data)
    assert len(result.errors) == 1
    assert "Crash" in result.errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# FrameProcessor Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_imports():
    assert FrameProcessor is not None
    assert FrameData is not None
    assert PipelineStage is not None


def test_instantiation():
    p = FrameProcessor(DetectionConfig())
    assert len(p.stages) == 1
    assert p.stages[0].name == "DepthProcessingStage"


def test_process_with_depth():
    p = FrameProcessor(DetectionConfig())
    rgb, depth = make_synthetic_frames()
    r = p.process(rgb, depth)
    assert r.has_depth()
    assert r.depth_colormap is not None
    assert r.depth_colormap.shape == (480, 640, 3)
    assert r.depth_colormap.dtype == np.uint8
    assert "timestamp" in r.metadata


def test_process_without_depth():
    p = FrameProcessor(DetectionConfig())
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    r = p.process(rgb)
    assert r.depth_frame is None
    assert not r.has_depth()
    assert r.depth_colormap is None


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


def test_get_stage_nonexistent():
    p = FrameProcessor(DetectionConfig())
    assert p.get_stage("NonExistent") is None


def test_remove_stage_nonexistent():
    p = FrameProcessor(DetectionConfig())
    assert p.remove_stage("NonExistent") is False


def test_threshold_update():
    p = FrameProcessor(DetectionConfig())
    p.set_depth_thresholds(0.5, 4.0)
    print("  PASS  test_threshold_update")


def test_set_action_thresholds_propagates():
    """set_action_thresholds should update DepthStage, FusionStage, and VisualAnnotationStage."""
    p = FrameProcessor(DetectionConfig())
    p.add_stage(FusionStage(config=DetectionConfig()))
    p.add_stage(VisualAnnotationStage(config=DetectionConfig()))
    p.set_action_thresholds(warning=2.5, danger=0.8)
    depth_stage = p.get_stage("DepthProcessingStage")
    fusion_stage = p.get_stage("FusionStage")
    annot_stage = p.get_stage("VisualAnnotationStage")
    assert depth_stage.warning_threshold == 2.5
    assert depth_stage.danger_threshold == 0.8
    assert annot_stage._config.warning_distance == 2.5
    assert annot_stage._config.danger_distance == 0.8


def test_latency_report():
    p = FrameProcessor(DetectionConfig())
    rgb, depth = make_synthetic_frames()
    p.process(rgb, depth)
    report = p.get_latency_report()
    assert "DepthProcessingStage" in report
    assert report["DepthProcessingStage"] > 0


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


# ═══════════════════════════════════════════════════════════════════════════════
# DepthProcessingStage Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_depth_colormap_colors():
    """LUT colormap: red=danger, yellow=warning, green=safe, black=invalid."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config, depth_min_m=0.3, depth_max_m=5.0)
    stage.danger_threshold = 1.0
    stage.warning_threshold = 3.0
    stage._build_depth_lut()

    h, w = 10, 10
    depth_m = np.zeros((h, w), dtype=np.float32)
    depth_m[0:3, :] = 0.5    # danger → red
    depth_m[3:6, :] = 2.0    # warning → yellow
    depth_m[6:10, :] = 4.0   # safe → green
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    data = FrameData(rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = stage.process(data)

    assert result.depth_colormap is not None
    # Danger zone should have red channel = 255
    assert result.depth_colormap[0, 0, 2] == 255  # BGR red
    # Warning zone should have green+yellow channels
    assert result.depth_colormap[3, 0, 1] == 255  # G
    assert result.depth_colormap[3, 0, 2] == 255  # R
    # Safe zone should have green channel
    assert result.depth_colormap[6, 0, 1] == 255  # G


def test_depth_colormap_raw_generated():
    """depth_colormap_raw should be set when depth_frame_raw is provided."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config)
    h, w = 10, 10
    depth_raw_filtered = np.full((h, w), 2000, dtype=np.uint16)
    depth_raw_unfiltered = np.full((h, w), 2000, dtype=np.uint16)
    data = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw_filtered,
        depth_frame_raw=depth_raw_unfiltered,
        depth_scale=0.001,
    )
    result = stage.process(data)
    assert result.depth_colormap is not None
    assert result.depth_colormap_raw is not None
    assert result.depth_colormap_raw.shape == (h, w, 3)


def test_depth_colormap_raw_not_set_without_raw():
    """depth_colormap_raw should remain None when depth_frame_raw is None."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config)
    h, w = 10, 10
    depth_raw = np.full((h, w), 2000, dtype=np.uint16)
    data = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw,
        depth_scale=0.001,
    )
    result = stage.process(data)
    assert result.depth_colormap is not None
    assert result.depth_colormap_raw is None


def test_depth_colormap_out_of_range_is_black():
    """Depth values beyond max_distance should map to black (not green)."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config, depth_max_m=5.0)
    stage._build_depth_lut()
    h, w = 10, 10
    # depth_m = 10.0 (beyond max_m=5.0) → should be black
    depth_m = np.full((h, w), 10.0, dtype=np.float32)
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    data = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw,
        depth_scale=0.001,
    )
    result = stage.process(data)
    cm = result.depth_colormap
    # All pixels should be black (0,0,0) since depth > max_m
    assert cm[0, 0, 0] == 0 and cm[0, 0, 1] == 0 and cm[0, 0, 2] == 0


def test_depth_colormap_zero_depth_is_black():
    """Depth = 0 (invalid/no data) should map to black."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config, depth_min_m=0.3)
    stage._build_depth_lut()
    h, w = 10, 10
    depth_raw = np.zeros((h, w), dtype=np.uint16)  # depth = 0
    data = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw,
        depth_scale=0.001,
    )
    result = stage.process(data)
    cm = result.depth_colormap
    assert cm[0, 0, 0] == 0 and cm[0, 0, 1] == 0 and cm[0, 0, 2] == 0


def test_depth_lut_rebuild_on_threshold_change():
    """LUT should rebuild when danger/warning thresholds change."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config)
    stage.danger_threshold = 1.0
    stage._build_depth_lut()
    lut_before = stage._depth_lut.copy()

    stage.set_action_thresholds(warning=2.5, danger=0.5)
    lut_after = stage._depth_lut

    # LUT should differ because danger threshold changed
    assert not np.array_equal(lut_before, lut_after)


def test_depth_stage_populates_obstacles():
    """DepthProcessingStage should fill data.obstacles after processing."""
    config = DetectionConfig()
    stage = DepthProcessingStage(config)
    rgb, depth = make_synthetic_frames()
    data = FrameData(rgb_frame=rgb, depth_frame=depth, depth_scale=0.001)
    result = stage.process(data)
    assert isinstance(result.obstacles, list)


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


def _make_frame_data(detections=None, obstacles=None, depth_at=None, depth_scale=0.001):
    """Helper to create FrameData with mock detections and obstacles.

    Args:
        detections: List of MockDetection objects.
        obstacles: List of depth obstacle dicts (bbox in [x,y,w,h] format).
        depth_at: Dict mapping (x1,y1,x2,y2) regions to distance in meters.
                  e.g. {(100,100,300,400): 2.0} sets depth within that bbox.
                  Background defaults to 10.0m (out of range).
        depth_scale: Depth scale factor (default 0.001).
    """
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.detections = detections or []
    data.obstacles = obstacles or []

    if depth_at is not None:
        depth_m = np.full((480, 640), 10.0, dtype=np.float32)  # Background = out of range
        for (x1, y1, x2, y2), dist_m in depth_at.items():
            depth_m[y1:y2, x1:x2] = dist_m
        data.depth_frame = (depth_m / depth_scale).astype(np.uint16)
        data.depth_scale = depth_scale

    return data


def test_fusion_matching():
    """YOLO detection with depth data → class and distance should be assigned."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 2.0, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs], depth_at={(100, 100, 300, 400): 2.0})
    result = fusion.process(data)
    assert len(result.fused_output) >= 1
    assert result.fused_output[0]["object_class"] == "person"


def test_fusion_no_match():
    """Depth-only obstacle with no YOLO match → stays 'obstacle'."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[500, 400, 600, 470])
    obs = {"bbox": [10, 10, 50, 50], "distance_m": 1.0, "zone": "left", "area_px": 2500, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs])
    result = fusion.process(data)
    obs_results = [r for r in result.fused_output if r["object_class"] == "obstacle"]
    assert len(obs_results) >= 1


def test_fusion_priority_person_close():
    """Person < danger_distance → priority 0 (STOP)."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 0.8})
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 0
    assert result.fused_output[0]["action"] == "STOP"


def test_fusion_priority_obstacle_close():
    """Non-person obstacle < danger_distance → priority 1."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=1, class_name="chair", confidence=0.8, bbox=[100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 0.5})
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 1
    assert result.fused_output[0]["action"] is None


def test_fusion_empty_inputs():
    """Empty detections and obstacles → empty fused_output."""
    fusion = FusionStage()
    data = _make_frame_data(detections=[], obstacles=[])
    result = fusion.process(data)
    assert result.fused_output == []


def test_fusion_bbox_format_xyxy():
    """Output bbox should be [x1, y1, x2, y2], not [x, y, w, h]."""
    fusion = FusionStage()
    obs = {"bbox": [100, 200, 50, 60], "distance_m": 1.5, "zone": "right", "area_px": 3000, "priority": 1.0}
    data = _make_frame_data(detections=[], obstacles=[obs])
    result = fusion.process(data)
    bbox = result.fused_output[0]["bbox"]
    # xyxy: [x1, y1, x1+w, y1+h] = [100, 200, 150, 260]
    assert bbox == [100, 200, 150, 260], f"Expected [100, 200, 150, 260], got {bbox}"


def test_fusion_overlap_ratio_with_area_px():
    """YOLO detection with depth → direct depth sampling gives correct class."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 5, 10])
    obs = {"bbox": [0, 0, 10, 10], "distance_m": 1.5, "zone": "center", "area_px": 60, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs], depth_at={(0, 0, 5, 10): 1.5})
    result = fusion.process(data)
    assert result.fused_output[0]["object_class"] == "person"


def test_fusion_config_thresholds():
    """Priority should use DetectionConfig.danger_distance, not hardcoded 1.0."""
    config = DetectionConfig()
    config.danger_distance = 2.0
    fusion = FusionStage(config=config)
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 1.5})
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 0, f"Expected 0, got {result.fused_output[0]['priority']}"
    assert result.fused_output[0]["action"] == "STOP"


def test_fusion_multiple_yolo_detections():
    """Multiple YOLO detections at different distances → multiple fused results."""
    fusion = FusionStage()
    det1 = MockDetection(0, "person", 0.9, [100, 100, 200, 200])
    det2 = MockDetection(1, "chair", 0.8, [400, 100, 500, 200])
    data = _make_frame_data(
        detections=[det1, det2],
        depth_at={(100, 100, 200, 200): 0.8, (400, 100, 500, 200): 3.0},
    )
    result = fusion.process(data)
    assert len(result.fused_output) == 2
    classes = {r["object_class"] for r in result.fused_output}
    assert "person" in classes
    assert "chair" in classes


def test_fusion_yolo_no_depth_skipped():
    """YOLO detection with no valid depth in bbox → skipped."""
    fusion = FusionStage()
    det = MockDetection(0, "person", 0.9, [100, 100, 200, 200])
    # Background depth = 10m (out of range), no valid depth in bbox
    data = _make_frame_data(detections=[det])
    result = fusion.process(data)
    person_results = [r for r in result.fused_output if r["object_class"] == "person"]
    assert len(person_results) == 0


def test_fusion_obstacle_far_filtered():
    """Depth-only obstacle > 1.5m → filtered out in PASS 2."""
    fusion = FusionStage()
    obs = {"bbox": [100, 100, 50, 50], "distance_m": 2.0, "zone": "center", "area_px": 3000, "priority": 1.0}
    data = _make_frame_data(detections=[], obstacles=[obs])
    result = fusion.process(data)
    assert len(result.fused_output) == 0


def test_fusion_person_at_warning_distance():
    """Person at warning_distance → priority 2 (not 0)."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    config.warning_distance = 3.0
    fusion = FusionStage(config=config)
    det = MockDetection(0, "person", 0.9, [100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 2.5})
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 2
    assert result.fused_output[0]["action"] is None


def test_fusion_nonperson_far_priority_3():
    """Non-person beyond warning_distance → priority 3."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    config.warning_distance = 3.0
    fusion = FusionStage(config=config)
    det = MockDetection(1, "chair", 0.8, [100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 4.0})
    result = fusion.process(data)
    assert result.fused_output[0]["priority"] == 3


def test_fusion_pass2_priority_ladder():
    """Depth-only obstacle priority: <0.5→1, <1.0→2, else→3."""
    fusion = FusionStage()
    obs_close = {"bbox": [10, 10, 50, 50], "distance_m": 0.3, "zone": "left", "area_px": 3000, "priority": 1.0}
    obs_mid = {"bbox": [300, 10, 50, 50], "distance_m": 0.8, "zone": "center", "area_px": 3000, "priority": 1.0}
    obs_far = {"bbox": [580, 10, 50, 50], "distance_m": 1.2, "zone": "right", "area_px": 3000, "priority": 1.0}
    data = _make_frame_data(detections=[], obstacles=[obs_close, obs_mid, obs_far])
    result = fusion.process(data)
    by_dist = {r["distance_m"]: r["priority"] for r in result.fused_output}
    assert by_dist[0.3] == 1
    assert by_dist[0.8] == 2
    assert by_dist[1.2] == 3


def test_fusion_dark_mode_skips_pass1():
    """In dark mode, PASS 1 is skipped — YOLO detections ignored."""
    fusion = FusionStage()
    det = MockDetection(0, "person", 0.9, [100, 100, 300, 400])
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 1.0, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[det], obstacles=[obs], depth_at={(100, 100, 300, 400): 1.0})
    data.metadata["is_dark"] = True
    data.metadata["rgb_confidence"] = 0.1
    result = fusion.process(data)
    person_results = [r for r in result.fused_output if r["object_class"] == "person"]
    assert len(person_results) == 0
    assert result.fused_output[0]["object_class"] == "obstacle"


def test_fusion_normal_mode_matches_yolo():
    """In normal lighting, FusionStage should match YOLO to depth."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 2.0})
    data.metadata["is_dark"] = False
    data.metadata["rgb_confidence"] = 0.9
    result = fusion.process(data)
    assert result.fused_output[0]["object_class"] == "person"


def test_fusion_dim_mode_lower_threshold():
    """Dim mode still samples depth directly for YOLO detections."""
    fusion = FusionStage()
    det = MockDetection(class_id=0, class_name="person", confidence=0.9, bbox=[0, 0, 40, 100])
    data = _make_frame_data(detections=[det], depth_at={(0, 0, 40, 100): 2.0})
    data.metadata["is_dark"] = False
    data.metadata["rgb_confidence"] = 0.3
    result = fusion.process(data)
    assert result.fused_output[0]["object_class"] == "person"


def test_fusion_determine_zone_left():
    fusion = FusionStage()
    assert fusion._determine_zone([0, 0, 100, 100], 640) == "left"


def test_fusion_determine_zone_center():
    fusion = FusionStage()
    assert fusion._determine_zone([280, 0, 360, 100], 640) == "center"


def test_fusion_determine_zone_right():
    fusion = FusionStage()
    assert fusion._determine_zone([500, 0, 600, 100], 640) == "right"


def test_fusion_sample_depth_bbox_clamped():
    """bbox partially outside frame → should clamp and still sample."""
    fusion = FusionStage()
    depth_m = np.full((480, 640), 2.0, dtype=np.float32)
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    # bbox extends beyond frame edges
    dist = fusion._sample_depth_in_bbox(depth_raw, 0.001, [-50, -50, 100, 100])
    assert dist is not None
    assert 0.3 <= dist <= 5.0


def test_fusion_sample_depth_bbox_all_invalid():
    """bbox with all-invalid depth (out of range) → None."""
    fusion = FusionStage()
    depth_m = np.full((480, 640), 0.1, dtype=np.float32)  # Below min_distance
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    dist = fusion._sample_depth_in_bbox(depth_raw, 0.001, [100, 100, 200, 200])
    assert dist is None


def test_fusion_sample_depth_bbox_tiny():
    """Very small bbox → should not crash (margin fallback)."""
    fusion = FusionStage()
    depth_m = np.full((480, 640), 2.0, dtype=np.float32)
    depth_raw = (depth_m / 0.001).astype(np.uint16)
    dist = fusion._sample_depth_in_bbox(depth_raw, 0.001, [100, 100, 103, 103])
    assert dist is not None


def test_fusion_overlap_ratio_identical_boxes():
    """Identical boxes → overlap = 1.0."""
    fusion = FusionStage()
    ratio = fusion._calculate_overlap_ratio([0, 0, 100, 100], [0, 0, 100, 100])
    assert ratio == 1.0


def test_fusion_overlap_ratio_no_overlap():
    """Non-overlapping boxes → overlap = 0.0."""
    fusion = FusionStage()
    ratio = fusion._calculate_overlap_ratio([0, 0, 50, 50], [100, 100, 150, 150])
    assert ratio == 0.0


def test_fusion_output_contract_keys():
    """Every fused_output item must have all required keys."""
    fusion = FusionStage()
    det = MockDetection(0, "person", 0.9, [100, 100, 300, 400])
    data = _make_frame_data(detections=[det], depth_at={(100, 100, 300, 400): 1.0})
    result = fusion.process(data)
    for item in result.fused_output:
        assert "object_class" in item
        assert "distance_m" in item
        assert "zone" in item
        assert "priority" in item
        assert "bbox" in item
        assert "action" in item
        assert len(item["bbox"]) == 4
        assert item["zone"] in ("left", "center", "right")


# ═══════════════════════════════════════════════════════════════════════════════
# Dark Mode Tests — CLAHE + Confidence Fallback
# ═══════════════════════════════════════════════════════════════════════════════

def test_yolo_stage_dark_frame_detection():
    """Dark frame (brightness < 40) should set is_dark=True and rgb_confidence low."""
    stage = YOLODetectionStage()
    rgb_dark = np.full((480, 640, 3), 10, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_dark)
    data = stage.process(data)
    assert data.metadata["is_dark"] == True
    assert data.metadata["rgb_confidence"] < 0.5


def test_yolo_stage_bright_frame_detection():
    """Bright frame should set is_dark=False and rgb_confidence high."""
    stage = YOLODetectionStage()
    rgb_bright = np.full((480, 640, 3), 180, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_bright)
    data = stage.process(data)
    assert data.metadata["is_dark"] == False
    assert data.metadata["rgb_confidence"] > 0.5


def test_yolo_stage_brightness_boundary():
    """Brightness exactly 40 → is_dark = False (threshold is < 40)."""
    stage = YOLODetectionStage()
    rgb = np.full((480, 640, 3), 40, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data = stage.process(data)
    assert data.metadata["is_dark"] == False


def test_yolo_stage_rgb_confidence_capped():
    """rgb_confidence should be capped at 1.0 even for very bright frames."""
    stage = YOLODetectionStage()
    rgb = np.full((480, 640, 3), 255, dtype=np.uint8)  # brightness=255
    data = FrameData(rgb_frame=rgb)
    data = stage.process(data)
    assert data.metadata["rgb_confidence"] == 1.0


def test_clahe_enhances_dark_frame():
    """CLAHE should increase mean brightness of a dark frame."""
    stage = YOLODetectionStage()
    dark = np.random.randint(5, 30, (480, 640, 3), dtype=np.uint8)
    enhanced = stage._enhance_dark_frame(dark)
    assert enhanced.mean() > dark.mean(), f"Enhanced {enhanced.mean():.1f} should be brighter than dark {dark.mean():.1f}"


def test_yolo_stage_no_models_active_model_none():
    """When no models available, active_model should be 'none'."""
    stage = YOLODetectionStage()
    stage._wrapper_rgb = None
    stage._wrapper_depth = None
    stage._depth_model_path = None
    rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data = stage.process(data)
    assert data.metadata["active_model"] == "none"
    assert data.detections == []


def test_yolo_stage_depth_filtered_fallback():
    """Dark + depth model + no raw colormap → fallback to filtered (depth_filtered)."""
    stage = YOLODetectionStage()
    stage._wrapper_rgb = _MockWrapper()
    stage._wrapper_depth = _MockWrapper()
    stage._depth_model_path = "fake_path"
    stage._wrapper_depth._detect_fn = lambda f: [MockDetection(0, "person", 0.9, [100, 100, 300, 400])]

    rgb_dark = np.full((480, 640, 3), 10, dtype=np.uint8)
    depth_colormap = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_dark, depth_colormap=depth_colormap)
    # depth_colormap_raw is None
    data = stage.process(data)
    assert data.metadata["active_model"] == "depth_filtered"
    assert len(data.detections) == 1


def test_metadata_propagation_through_pipeline():
    """is_dark and rgb_confidence should propagate from YOLO stage to Fusion stage."""
    processor = FrameProcessor(DetectionConfig())
    yolo_stage = YOLODetectionStage()
    yolo_stage._wrapper_rgb = None
    yolo_stage._wrapper_depth = None
    yolo_stage._depth_model_path = None
    processor.add_stage(yolo_stage)
    processor.add_stage(FusionStage())

    rgb_dark = np.full((480, 640, 3), 15, dtype=np.uint8)
    depth_m = np.full((480, 640), 2.0, dtype=np.float32)
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    data = processor.process(rgb_dark, depth_raw)
    assert data.metadata["is_dark"] == True
    assert data.metadata["rgb_confidence"] < 0.5
    assert "is_dark" in data.metadata
    assert "rgb_confidence" in data.metadata


def test_fusion_priority_obstacle_in_dark():
    """In dark mode, depth-only obstacle priority follows PASS 2 ladder."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    fusion = FusionStage(config=config)
    obs = {"bbox": [140, 150, 80, 100], "distance_m": 0.3, "zone": "center", "area_px": 8000, "priority": 1.0}
    data = _make_frame_data(detections=[], obstacles=[obs])
    data.metadata["is_dark"] = True
    data.metadata["rgb_confidence"] = 0.1
    result = fusion.process(data)
    assert result.fused_output[0]["object_class"] == "obstacle"
    assert result.fused_output[0]["priority"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-Model Swap Tests
# ═══════════════════════════════════════════════════════════════════════════════

class _MockWrapper:
    """Minimal mock for YOLOWrapper with injectable detect()."""
    def __init__(self):
        self._detect_fn = lambda f: []
    def detect(self, frame):
        return self._detect_fn(frame)


def test_dual_model_depth_active_in_dark():
    """When depth model available and frame is dark, active_model should be 'depth'."""
    stage = YOLODetectionStage()
    stage._wrapper_rgb = _MockWrapper()
    stage._wrapper_depth = _MockWrapper()
    stage._wrapper_depth._detect_fn = lambda f: [MockDetection(0, "person", 0.9, [100, 100, 300, 400])]

    rgb_dark = np.full((480, 640, 3), 10, dtype=np.uint8)
    depth_colormap_raw = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_dark, depth_colormap_raw=depth_colormap_raw)
    data = stage.process(data)
    assert data.metadata["active_model"] == "depth"
    assert len(data.detections) == 1


def test_dual_model_rgb_active_in_light():
    """When frame is bright, active_model should be 'rgb'."""
    stage = YOLODetectionStage()
    stage._wrapper_rgb = _MockWrapper()
    stage._wrapper_depth = _MockWrapper()
    stage._wrapper_rgb._detect_fn = lambda f: [MockDetection(0, "person", 0.9, [100, 100, 300, 400])]

    rgb_bright = np.full((480, 640, 3), 180, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_bright)
    data = stage.process(data)
    assert data.metadata["active_model"] == "rgb"
    assert len(data.detections) == 1


def test_dual_model_fallback_to_rgb_clahe():
    """When dark but no depth model, active_model should be 'rgb_clahe'."""
    stage = YOLODetectionStage()
    stage._wrapper_rgb = _MockWrapper()
    stage._wrapper_depth = None
    stage._depth_model_path = None

    rgb_dark = np.full((480, 640, 3), 10, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb_dark)
    data = stage.process(data)
    assert data.metadata["active_model"] == "rgb_clahe"


# ═══════════════════════════════════════════════════════════════════════════════
# VisualAnnotationStage Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_visual_annotation_none_rgb():
    """None rgb_frame → no crash, data unchanged."""
    stage = VisualAnnotationStage()
    data = FrameData(rgb_frame=None)
    result = stage.process(data)
    assert result.rgb_frame is None


def test_visual_annotation_empty_items():
    """No items to draw → no crash, rgb_frame unchanged."""
    stage = VisualAnnotationStage()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    result = stage.process(data)
    # Frame should be mostly unchanged (only HUD status text drawn)
    # Check that it didn't crash and frame is still valid
    assert result.rgb_frame is not None
    assert result.rgb_frame.shape == (480, 640, 3)


def test_visual_annotation_with_fused_output():
    """fused_output with danger item → modifies rgb_frame in-place."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    stage = VisualAnnotationStage(config=config)
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.fused_output = [{
        "object_class": "person",
        "distance_m": 0.5,
        "zone": "center",
        "priority": 0,
        "bbox": [100, 100, 200, 200],
        "action": "STOP",
    }]
    original = rgb.copy()
    result = stage.process(data)
    # Frame should be modified (drawing added)
    assert not np.array_equal(result.rgb_frame, original)


def test_visual_annotation_with_obstacles_fallback():
    """No fused_output → falls back to obstacles (xywh bbox format)."""
    stage = VisualAnnotationStage()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.obstacles = [{
        "bbox": [100, 100, 80, 80],  # xywh
        "distance_m": 0.8,
        "zone": "center",
        "priority": 1.0,
    }]
    result = stage.process(data)
    assert result.rgb_frame is not None


def test_visual_annotation_with_yolo_fallback():
    """No fused_output, no obstacles, but detections → YOLO fallback mode."""
    stage = VisualAnnotationStage()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.detections = [MockDetection(0, "person", 0.9, [100, 100, 200, 200])]
    result = stage.process(data)
    assert result.rgb_frame is not None


def test_visual_annotation_danger_status():
    """Priority <= 1 → global_status DANGER, red pixels drawn."""
    config = DetectionConfig()
    config.danger_distance = 1.0
    stage = VisualAnnotationStage(config=config)
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.fused_output = [{
        "object_class": "person",
        "distance_m": 0.3,
        "zone": "center",
        "priority": 0,
        "bbox": [100, 100, 200, 200],
        "action": "STOP",
    }]
    stage.process(data)
    # The HUD text area (top-left) should have some red pixels
    hud_region = rgb[20:60, 10:200]
    red_pixels = np.sum((hud_region[:, :, 2] > 200) & (hud_region[:, :, 0] < 100))
    assert red_pixels > 0, "Expected red pixels in HUD status area for DANGER"


def test_visual_annotation_modifies_in_place():
    """VisualAnnotationStage should modify rgb_frame in-place (same object)."""
    stage = VisualAnnotationStage()
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    data = FrameData(rgb_frame=rgb)
    data.fused_output = [{
        "object_class": "person",
        "distance_m": 2.0,
        "zone": "center",
        "priority": 2,
        "bbox": [100, 100, 200, 200],
        "action": None,
    }]
    result = stage.process(data)
    assert result.rgb_frame is rgb  # Same numpy array object


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Integration Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_integration():
    """Full pipeline: Depth → YOLO → Fusion → Annotation with synthetic data."""
    config = DetectionConfig()
    processor = FrameProcessor(config)

    # Add YOLO stage with mock (no real model)
    yolo_stage = YOLODetectionStage()
    yolo_stage._wrapper_rgb = None
    yolo_stage._wrapper_depth = None
    yolo_stage._depth_model_path = None
    processor.add_stage(yolo_stage)

    processor.add_stage(FusionStage(config=config))
    processor.add_stage(VisualAnnotationStage(config=config))

    rgb, depth = make_synthetic_frames()
    result = processor.process(rgb, depth)

    assert result.rgb_frame is not None
    assert result.depth_colormap is not None
    assert isinstance(result.obstacles, list)
    assert isinstance(result.fused_output, list)
    assert "is_dark" in result.metadata
    assert "rgb_confidence" in result.metadata
    assert "active_model" in result.metadata


def test_pipeline_stage_error_doesnt_crash():
    """A stage that throws should not crash the pipeline; error captured in data.errors."""
    class CrashStage(PipelineStage):
        def process(self, data):
            raise RuntimeError("intentional crash")

    p = FrameProcessor(DetectionConfig())
    p.add_stage(CrashStage("Crash"))
    rgb, depth = make_synthetic_frames()
    result = p.process(rgb, depth)
    assert len(result.errors) == 1
    assert "Crash" in result.errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# NavigationStage Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_nav_depth_frame(h=480, w=640, obstacle_regions=None, clear_m=5.0, depth_scale=0.001):
    """Create a depth frame with optional obstacle regions.
    obstacle_regions: list of (x_start_frac, x_end_frac, distance_m)
    """
    depth_m = np.full((h, w), clear_m, dtype=np.float32)
    if obstacle_regions:
        for x_start_frac, x_end_frac, dist_m in obstacle_regions:
            depth_m[:, int(w * x_start_frac):int(w * x_end_frac)] = dist_m
    return (depth_m / depth_scale).astype(np.uint16)


def test_nav_clear_path():
    """No obstacles → status CLEAR, steering 0, speed 1.0."""
    nav = NavigationStage()
    depth_raw = _make_nav_depth_frame(clear_m=5.0)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "CLEAR"
    assert result.navigation["steering_angle_deg"] == 0.0
    assert result.navigation["speed"] == 1.0


def test_nav_blocked_all():
    """All sectors blocked → status BLOCKED, speed 0.0."""
    nav = NavigationStage()
    # All sectors have obstacle at 0.3m (below min_gap_m of 1.1m)
    depth_raw = _make_nav_depth_frame(clear_m=0.3)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "BLOCKED"
    assert result.navigation["speed"] == 0.0


def test_nav_obstacle_left_steers_right():
    """Obstacle on left → steering should be positive (right)."""
    nav = NavigationStage()
    # Obstacle in left third (0-33% of frame), clear elsewhere
    depth_raw = _make_nav_depth_frame(obstacle_regions=[(0.0, 0.33, 0.3)])
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "AVOIDING"
    assert result.navigation["steering_angle_deg"] > 0, "Should steer right to avoid left obstacle"


def test_nav_obstacle_right_steers_left():
    """Obstacle on right → steering should be negative (left)."""
    nav = NavigationStage()
    depth_raw = _make_nav_depth_frame(obstacle_regions=[(0.67, 1.0, 0.3)])
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "AVOIDING"
    assert result.navigation["steering_angle_deg"] < 0, "Should steer left to avoid right obstacle"


def test_nav_obstacle_center_steers_around():
    """Obstacle in center → should steer around (non-zero angle)."""
    nav = NavigationStage()
    depth_raw = _make_nav_depth_frame(obstacle_regions=[(0.33, 0.67, 0.3)])
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "AVOIDING"
    assert result.navigation["steering_angle_deg"] != 0, "Should steer around center obstacle"


def test_nav_no_depth_clear():
    """No depth data (webcam mode) → status CLEAR, speed 1.0."""
    nav = NavigationStage()
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8))
    result = nav.process(data)
    assert result.navigation["status"] == "CLEAR"
    assert result.navigation["speed"] == 1.0


def test_nav_safety_override_person_close():
    """FusionStage priority 0 (person in danger) → STOPPED regardless of gaps."""
    nav = NavigationStage()
    depth_raw = _make_nav_depth_frame(clear_m=5.0)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    data.fused_output = [{"object_class": "person", "distance_m": 0.8, "zone": "center",
                          "priority": 0, "bbox": [100, 100, 200, 200], "action": "STOP"}]
    result = nav.process(data)
    assert result.navigation["status"] == "STOPPED"
    assert result.navigation["speed"] == 0.0


def test_nav_speed_ramps_with_distance():
    """Speed should ramp from 0 (danger) to 1.0 (warning+)."""
    nav = NavigationStage()
    config = DetectionConfig()
    config.danger_distance = 1.0
    config.warning_distance = 3.0
    nav._config = config

    # Obstacle at 2.0m (between danger and warning) → speed should be ~0.5
    depth_raw = _make_nav_depth_frame(clear_m=2.0)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    # 2.0m is between 1.0 and 3.0 → t = (2.0-1.0)/(3.0-1.0) = 0.5
    assert 0.3 < result.navigation["speed"] < 0.7, f"Expected ~0.5, got {result.navigation['speed']}"


def test_nav_output_contract():
    """Navigation output must have all required keys."""
    nav = NavigationStage()
    depth_raw = _make_nav_depth_frame(clear_m=5.0)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    nav_out = result.navigation
    assert "steering_angle_deg" in nav_out
    assert "speed" in nav_out
    assert "status" in nav_out
    assert "gaps" in nav_out
    assert "histogram" in nav_out
    assert "blocked_sectors" in nav_out
    assert isinstance(nav_out["steering_angle_deg"], float)
    assert isinstance(nav_out["speed"], float)
    assert isinstance(nav_out["status"], str)
    assert isinstance(nav_out["gaps"], list)


def test_nav_histogram_correct_sectors():
    """Histogram should have num_sectors entries."""
    nav = NavigationStage(num_sectors=12)
    depth_raw = _make_nav_depth_frame(clear_m=5.0)
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert len(result.navigation["histogram"]) == 12
    assert len(result.navigation["blocked_sectors"]) == 12


def test_nav_steering_clamped():
    """Steering angle should be clamped to max_steer_deg."""
    nav = NavigationStage(max_steer_deg=30.0)
    depth_raw = _make_nav_depth_frame(obstacle_regions=[(0.0, 0.8, 0.3)])
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert abs(result.navigation["steering_angle_deg"]) <= 30.0


def test_nav_gap_found_when_partial_block():
    """Partial blockage should still find a gap (AVOIDING, not BLOCKED)."""
    nav = NavigationStage()
    # Block left half, clear right half
    depth_raw = _make_nav_depth_frame(obstacle_regions=[(0.0, 0.5, 0.3)])
    data = FrameData(rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth_frame=depth_raw, depth_scale=0.001)
    result = nav.process(data)
    assert result.navigation["status"] == "AVOIDING"
    assert len(result.navigation["gaps"]) >= 1


def test_nav_full_pipeline_with_navigation():
    """Full pipeline: Depth → YOLO → Fusion → Navigation → Annotation."""
    config = DetectionConfig()
    processor = FrameProcessor(config)

    yolo_stage = YOLODetectionStage()
    yolo_stage._wrapper_rgb = None
    yolo_stage._wrapper_depth = None
    yolo_stage._depth_model_path = None
    processor.add_stage(yolo_stage)
    processor.add_stage(FusionStage(config=config))
    processor.add_stage(NavigationStage(config=config))
    processor.add_stage(VisualAnnotationStage(config=config))

    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth_m = np.full((480, 640), 5.0, dtype=np.float32)
    depth_m[200:300, 280:360] = 1.5
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    result = processor.process(rgb, depth_raw, 0.001)
    assert result.navigation is not None
    assert "status" in result.navigation
    assert "steering_angle_deg" in result.navigation


if __name__ == "__main__":
    print("=== FrameProcessor Tests ===\n")
    tests = [
        # FrameData
        ("frame data defaults", test_frame_data_defaults),
        ("frame data has_depth", test_frame_data_has_depth),
        # PipelineStage
        ("pipeline stage disabled skips", test_pipeline_stage_disabled_skips_processing),
        ("pipeline stage measure latency", test_pipeline_stage_measure_sets_latency),
        ("pipeline stage exception captured", test_pipeline_stage_exception_captured),
        # FrameProcessor
        ("imports", test_imports),
        ("instantiation", test_instantiation),
        ("process with depth", test_process_with_depth),
        ("process without depth", test_process_without_depth),
        ("stage management", test_stage_management),
        ("get stage nonexistent", test_get_stage_nonexistent),
        ("remove stage nonexistent", test_remove_stage_nonexistent),
        ("threshold update", test_threshold_update),
        ("set action thresholds propagates", test_set_action_thresholds_propagates),
        ("latency report", test_latency_report),
        ("custom stage", test_custom_stage),
        # DepthProcessingStage
        ("depth colormap colors", test_depth_colormap_colors),
        ("depth colormap raw generated", test_depth_colormap_raw_generated),
        ("depth colormap raw not set without raw", test_depth_colormap_raw_not_set_without_raw),
        ("depth colormap out of range is black", test_depth_colormap_out_of_range_is_black),
        ("depth colormap zero depth is black", test_depth_colormap_zero_depth_is_black),
        ("depth lut rebuild on threshold change", test_depth_lut_rebuild_on_threshold_change),
        ("depth stage populates obstacles", test_depth_stage_populates_obstacles),
        # FusionStage
        ("fusion matching", test_fusion_matching),
        ("fusion no match", test_fusion_no_match),
        ("fusion priority person close", test_fusion_priority_person_close),
        ("fusion priority obstacle close", test_fusion_priority_obstacle_close),
        ("fusion empty inputs", test_fusion_empty_inputs),
        ("fusion bbox format xyxy", test_fusion_bbox_format_xyxy),
        ("fusion overlap with area_px", test_fusion_overlap_ratio_with_area_px),
        ("fusion config thresholds", test_fusion_config_thresholds),
        ("fusion multiple yolo", test_fusion_multiple_yolo_detections),
        ("fusion yolo no depth skipped", test_fusion_yolo_no_depth_skipped),
        ("fusion obstacle far filtered", test_fusion_obstacle_far_filtered),
        ("fusion person at warning distance", test_fusion_person_at_warning_distance),
        ("fusion nonperson far priority 3", test_fusion_nonperson_far_priority_3),
        ("fusion pass2 priority ladder", test_fusion_pass2_priority_ladder),
        ("fusion dark mode skips pass1", test_fusion_dark_mode_skips_pass1),
        ("fusion normal mode matches yolo", test_fusion_normal_mode_matches_yolo),
        ("fusion dim mode lower threshold", test_fusion_dim_mode_lower_threshold),
        ("fusion determine zone left", test_fusion_determine_zone_left),
        ("fusion determine zone center", test_fusion_determine_zone_center),
        ("fusion determine zone right", test_fusion_determine_zone_right),
        ("fusion sample depth bbox clamped", test_fusion_sample_depth_bbox_clamped),
        ("fusion sample depth all invalid", test_fusion_sample_depth_bbox_all_invalid),
        ("fusion sample depth tiny bbox", test_fusion_sample_depth_bbox_tiny),
        ("fusion overlap identical boxes", test_fusion_overlap_ratio_identical_boxes),
        ("fusion overlap no overlap", test_fusion_overlap_ratio_no_overlap),
        ("fusion output contract keys", test_fusion_output_contract_keys),
        # Dark Mode
        ("yolo dark frame detection", test_yolo_stage_dark_frame_detection),
        ("yolo bright frame detection", test_yolo_stage_bright_frame_detection),
        ("yolo brightness boundary", test_yolo_stage_brightness_boundary),
        ("yolo rgb confidence capped", test_yolo_stage_rgb_confidence_capped),
        ("clahe enhances dark frame", test_clahe_enhances_dark_frame),
        ("yolo no models active none", test_yolo_stage_no_models_active_model_none),
        ("yolo depth filtered fallback", test_yolo_stage_depth_filtered_fallback),
        ("metadata propagation pipeline", test_metadata_propagation_through_pipeline),
        ("fusion priority obstacle in dark", test_fusion_priority_obstacle_in_dark),
        # Dual model
        ("dual model depth active in dark", test_dual_model_depth_active_in_dark),
        ("dual model rgb active in light", test_dual_model_rgb_active_in_light),
        ("dual model fallback to rgb_clahe", test_dual_model_fallback_to_rgb_clahe),
        # VisualAnnotationStage
        ("visual annotation none rgb", test_visual_annotation_none_rgb),
        ("visual annotation empty items", test_visual_annotation_empty_items),
        ("visual annotation with fused", test_visual_annotation_with_fused_output),
        ("visual annotation with obstacles", test_visual_annotation_with_obstacles_fallback),
        ("visual annotation with yolo fallback", test_visual_annotation_with_yolo_fallback),
        ("visual annotation danger status", test_visual_annotation_danger_status),
        ("visual annotation modifies in place", test_visual_annotation_modifies_in_place),
        # NavigationStage
        ("nav clear path", test_nav_clear_path),
        ("nav blocked all", test_nav_blocked_all),
        ("nav obstacle left steers right", test_nav_obstacle_left_steers_right),
        ("nav obstacle right steers left", test_nav_obstacle_right_steers_left),
        ("nav obstacle center steers around", test_nav_obstacle_center_steers_around),
        ("nav no depth clear", test_nav_no_depth_clear),
        ("nav safety override person close", test_nav_safety_override_person_close),
        ("nav speed ramps with distance", test_nav_speed_ramps_with_distance),
        ("nav output contract", test_nav_output_contract),
        ("nav histogram correct sectors", test_nav_histogram_correct_sectors),
        ("nav steering clamped", test_nav_steering_clamped),
        ("nav gap found when partial block", test_nav_gap_found_when_partial_block),
        ("nav full pipeline with navigation", test_nav_full_pipeline_with_navigation),
        # Integration
        ("full pipeline integration", test_full_pipeline_integration),
        ("pipeline stage error doesnt crash", test_pipeline_stage_error_doesnt_crash),
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
