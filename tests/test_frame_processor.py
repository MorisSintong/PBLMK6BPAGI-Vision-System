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
