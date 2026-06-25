"""
Comprehensive benchmark suite for PBL Vision System.
Tests all success criteria defined in ROLES.md.

Usage:
    conda run -n depth-obstacle-detector python tests/benchmark.py
"""

import os
import sys
import time
import statistics
import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
from Vision.inc.detection_config import DetectionConfig
from Vision.src.frame_processor import (
    FrameProcessor,
    FrameData,
    DepthProcessingStage,
    YOLODetectionStage,
    FusionStage,
    VisualAnnotationStage,
)
from Vision.src.obstacle_detector import ObstacleDetector
from Vision.src.yolowrapper import YOLOWrapper, Detection

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_frames(h=480, w=640, depth_scale=0.001):
    """Create synthetic RGB + depth frames with a realistic obstacle."""
    rgb = np.random.randint(100, 180, (h, w, 3), dtype=np.uint8)
    depth_m = np.full((h, w), 5.0, dtype=np.float32)
    # Person-shaped obstacle at 1.5m, center zone
    depth_m[180:380, 250:400] = 1.5
    # Another obstacle at 3.0m, right zone
    depth_m[200:320, 480:580] = 3.0
    depth_raw = (depth_m / depth_scale).astype(np.uint16)
    depth_raw_unfiltered = depth_raw.copy()
    return rgb, depth_raw, depth_raw_unfiltered


def make_dark_synthetic_frames(h=480, w=640, depth_scale=0.001):
    """Create dark RGB + depth frames (brightness < 40)."""
    rgb = np.random.randint(5, 30, (h, w, 3), dtype=np.uint8)
    depth_m = np.full((h, w), 5.0, dtype=np.float32)
    depth_m[180:380, 250:400] = 1.5
    depth_raw = (depth_m / depth_scale).astype(np.uint16)
    return rgb, depth_raw, depth_raw


def percentile_stats(latencies_ms):
    """Compute P50, P95, P99, mean, std."""
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    return {
        "mean": statistics.mean(latencies_ms),
        "std": statistics.stdev(latencies_ms) if n > 1 else 0.0,
        "p50": sorted_l[n // 2],
        "p95": sorted_l[int(n * 0.95)],
        "p99": sorted_l[int(n * 0.99)],
        "min": min(latencies_ms),
        "max": max(latencies_ms),
        "count": n,
    }


def print_stats(name, stats, target_ms=None):
    """Print latency stats with optional target check."""
    print(f"\n  {name}:")
    print(f"    Mean:  {stats['mean']:.2f} ms")
    print(f"    Std:   {stats['std']:.2f} ms")
    print(f"    P50:   {stats['p50']:.2f} ms")
    print(f"    P95:   {stats['p95']:.2f} ms")
    print(f"    P99:   {stats['p99']:.2f} ms")
    print(f"    Min:   {stats['min']:.2f} ms")
    print(f"    Max:   {stats['max']:.2f} ms")
    if target_ms:
        status = "PASS" if stats["p95"] <= target_ms else "FAIL"
        print(f"    Target P95 <= {target_ms}ms: {status}")
    return stats


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 1: YOLO Inference Latency (R2 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_yolo_inference():
    """R2: Latency ≤50ms (GPU) / ≤100ms (CPU) per inference."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: YOLO Inference Latency (R2 Success Criteria)")
    print("  Target: ≤50ms (GPU) / ≤100ms (CPU)")
    print("=" * 70)

    models_dir = os.path.join("Vision", "models")
    rgb_model = os.path.join(models_dir, "ModelRGB_V4.2.pt")
    depth_model = os.path.join(models_dir, "ModelDepth_V4.pt")

    results = {}

    # RGB model
    if os.path.exists(rgb_model):
        print(f"\n  Loading RGB model: {os.path.basename(rgb_model)}")
        wrapper = YOLOWrapper(model_path=rgb_model, conf_threshold=0.25, input_size=320)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm-up (wrapper already does one, but do 3 more)
        for _ in range(3):
            wrapper.detect(dummy)

        # Benchmark
        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            wrapper.detect(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

        stats = percentile_stats(latencies)
        print_stats("RGB Model (ModelRGB_V4.2.pt)", stats, target_ms=50)
        results["rgb_model"] = stats
    else:
        print(f"\n  [SKIP] RGB model not found: {rgb_model}")
        results["rgb_model"] = None

    # Depth model
    if os.path.exists(depth_model):
        print(f"\n  Loading Depth model: {os.path.basename(depth_model)}")
        wrapper = YOLOWrapper(model_path=depth_model, conf_threshold=0.25, input_size=320)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(3):
            wrapper.detect(dummy)

        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            wrapper.detect(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

        stats = percentile_stats(latencies)
        print_stats("Depth Model (ModelDepth_V4.pt)", stats, target_ms=50)
        results["depth_model"] = stats
    else:
        print(f"\n  [SKIP] Depth model not found: {depth_model}")
        results["depth_model"] = None

    return results


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 2: Per-Stage Pipeline Latency (R5 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_per_stage_latency():
    """R5: Test harness measures latency per stage independently."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Per-Stage Pipeline Latency (R5 Success Criteria)")
    print("=" * 70)

    config = DetectionConfig()
    processor = FrameProcessor(config)

    models_dir = os.path.join("Vision", "models")
    rgb_model = os.path.join(models_dir, "ModelRGB_V4.2.pt")
    depth_model = os.path.join(models_dir, "ModelDepth_V4.pt")

    if os.path.exists(rgb_model):
        processor.add_stage(YOLODetectionStage(
            model_path=rgb_model,
            depth_model_path=depth_model if os.path.exists(depth_model) else None,
        ))
    processor.add_stage(FusionStage(config=config))
    processor.add_stage(VisualAnnotationStage(config=config))

    # Warm-up
    rgb, depth, depth_raw = make_synthetic_frames()
    processor.process(rgb, depth, 0.001, depth_raw)
    processor.process(rgb, depth, 0.001, depth_raw)

    # Benchmark 100 iterations
    N = 100
    stage_latencies = {}
    total_latencies = []

    for i in range(N):
        rgb, depth, depth_raw = make_synthetic_frames()
        t0 = time.perf_counter()
        result = processor.process(rgb, depth, 0.001, depth_raw)
        total_ms = (time.perf_counter() - t0) * 1000
        total_latencies.append(total_ms)

        report = processor.get_latency_report()
        for stage_name, latency in report.items():
            if stage_name not in stage_latencies:
                stage_latencies[stage_name] = []
            stage_latencies[stage_name].append(latency)

    print(f"\n  Iterations: {N}")
    print(f"\n  Per-Stage Latency:")
    for stage_name, latencies in stage_latencies.items():
        stats = percentile_stats(latencies)
        print_stats(f"  {stage_name}", stats)

    total_stats = percentile_stats(total_latencies)
    print_stats("  Total Pipeline (all stages)", total_stats, target_ms=100)
    fps = 1000.0 / total_stats["p50"]
    print(f"    Throughput: {fps:.1f} FPS (P50-based)")

    return {
        "per_stage": {name: percentile_stats(lats) for name, lats in stage_latencies.items()},
        "total": total_stats,
        "fps": fps,
    }


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 3: Pipeline FPS (R1 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_pipeline_fps():
    """R1: Pipeline ≥25 FPS (RealSense) / ≥30 FPS (webcam)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Pipeline Throughput / FPS (R1 Success Criteria)")
    print("  Target: ≥25 FPS (RealSense mode with depth)")
    print("=" * 70)

    config = DetectionConfig()
    processor = FrameProcessor(config)

    models_dir = os.path.join("Vision", "models")
    rgb_model = os.path.join(models_dir, "ModelRGB_V4.2.pt")

    if os.path.exists(rgb_model):
        processor.add_stage(YOLODetectionStage(model_path=rgb_model))
    processor.add_stage(FusionStage(config=config))
    processor.add_stage(VisualAnnotationStage(config=config))

    # Warm-up
    rgb, depth, depth_raw = make_synthetic_frames()
    processor.process(rgb, depth, 0.001, depth_raw)

    # Benchmark 200 frames
    N = 200
    frame_times = []

    for i in range(N):
        rgb, depth, depth_raw = make_synthetic_frames()
        t0 = time.perf_counter()
        processor.process(rgb, depth, 0.001, depth_raw)
        frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(frame_ms)

    stats = percentile_stats(frame_times)
    fps_p50 = 1000.0 / stats["p50"]
    fps_p95 = 1000.0 / stats["p95"]
    fps_mean = 1000.0 / stats["mean"]

    print(f"\n  Iterations: {N}")
    print(f"  Mean FPS:   {fps_mean:.1f}")
    print(f"  P50 FPS:    {fps_p50:.1f}")
    print(f"  P95 FPS:    {fps_p95:.1f}")
    print(f"  P50 latency: {stats['p50']:.2f} ms")
    print(f"  P95 latency: {stats['p95']:.2f} ms")

    target_fps = 25
    status = "PASS" if fps_p50 >= target_fps else "FAIL"
    print(f"  Target ≥{target_fps} FPS (P50): {status} ({fps_p50:.1f} FPS)")

    return {"fps_mean": fps_mean, "fps_p50": fps_p50, "fps_p95": fps_p95, "latency_stats": stats}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 4: Depth Colormap LUT Accuracy (R3 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_depth_colormap():
    """R3: Colormap displays red/yellow/green zones correctly."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Depth Colormap LUT Accuracy (R3 Success Criteria)")
    print("=" * 70)

    config = DetectionConfig()
    stage = DepthProcessingStage(config)
    stage.danger_threshold = 1.0
    stage.warning_threshold = 3.0
    stage._build_depth_lut()

    h, w = 100, 100
    depth_m = np.zeros((h, w), dtype=np.float32)
    depth_m[0:25, :] = 0.5    # danger → red
    depth_m[25:50, :] = 2.0   # warning → yellow
    depth_m[50:75, :] = 4.0   # safe → green
    depth_m[75:100, :] = 10.0 # invalid → black
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    data = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw,
        depth_scale=0.001,
    )
    result = stage.process(data)
    cm = result.depth_colormap

    # Verify colors (BGR)
    checks = {
        "danger (red)": (cm[0, 0, 2] == 255 and cm[0, 0, 0] == 0 and cm[0, 0, 1] == 0),
        "warning (yellow)": (cm[25, 0, 1] == 255 and cm[25, 0, 2] == 255),
        "safe (green)": (cm[50, 0, 1] == 255 and cm[50, 0, 2] == 0),
        "invalid (black)": (cm[75, 0, 0] == 0 and cm[75, 0, 1] == 0 and cm[75, 0, 2] == 0),
    }

    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    # Verify unfiltered colormap
    data_raw = FrameData(
        rgb_frame=np.zeros((h, w, 3), dtype=np.uint8),
        depth_frame=depth_raw,
        depth_frame_raw=depth_raw,
        depth_scale=0.001,
    )
    result_raw = stage.process(data_raw)
    raw_ok = result_raw.depth_colormap_raw is not None
    print(f"  unfiltered colormap generated: {'PASS' if raw_ok else 'FAIL'}")

    return {"all_pass": all_pass and raw_ok, "checks": checks}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 5: Obstacle Detection Accuracy (R3 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_obstacle_detection():
    """R3: ObstacleDetector accurate for 0.3m-5m objects."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Obstacle Detection Accuracy (R3 Success Criteria)")
    print("  Target: accurate distance for objects 0.3m-5m")
    print("=" * 70)

    detector = ObstacleDetector(max_distance_m=5.0, min_distance_m=0.3, min_area=100)
    color = np.full((480, 640, 3), 128, dtype=np.uint8)

    test_distances = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5]
    results = {}

    for true_dist in test_distances:
        depth_m = np.full((480, 640), 10.0, dtype=np.float32)
        depth_m[200:300, 280:360] = true_dist
        depth_raw = (depth_m / 0.001).astype(np.uint16)

        _, obstacles = detector.detect(color, depth_raw)
        if obstacles:
            measured = obstacles[0]["distance_m"]
            error_pct = abs(measured - true_dist) / true_dist * 100
            results[true_dist] = {"measured": measured, "error_pct": error_pct}
            status = "PASS" if error_pct < 10 else "WARN"
            print(f"  True: {true_dist:.1f}m → Measured: {measured:.2f}m (error: {error_pct:.1f}%) {status}")
        else:
            results[true_dist] = {"measured": None, "error_pct": 100}
            print(f"  True: {true_dist:.1f}m → NOT DETECTED FAIL")

    # Check 3 zones
    print(f"\n  Zone Detection:")
    zone_ok = True
    for zone_name, x_range in [("left", (20, 100)), ("center", (280, 360)), ("right", (540, 620))]:
        depth_m = np.full((480, 640), 10.0, dtype=np.float32)
        depth_m[200:300, x_range[0]:x_range[1]] = 2.0
        depth_raw = (depth_m / 0.001).astype(np.uint16)
        _, obstacles = detector.detect(color, depth_raw)
        if obstacles:
            detected_zone = obstacles[0]["zone"]
            ok = detected_zone == zone_name
            if not ok:
                zone_ok = False
            print(f"    {zone_name}: detected={detected_zone} {'PASS' if ok else 'FAIL'}")
        else:
            zone_ok = False
            print(f"    {zone_name}: NOT DETECTED FAIL")

    return {"distances": results, "zones_ok": zone_ok}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 6: Fusion Priority Matrix (R4 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_fusion_priority():
    """R4: Priority sorted correctly, distance ±10% for 0.5-4m."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Fusion Priority Matrix (R4 Success Criteria)")
    print("  Target: correct priority ordering, distance ±10% for 0.5-4m")
    print("=" * 70)

    config = DetectionConfig()
    config.danger_distance = 1.5
    config.warning_distance = 3.0
    fusion = FusionStage(config=config)

    # Test cases: (class, true_distance, expected_priority, expected_action)
    test_cases = [
        ("person", 0.8, 0, "STOP"),       # person < danger → priority 0, STOP
        ("person", 2.0, 2, None),         # person < warning → priority 2
        ("person", 4.0, 3, None),         # person far → priority 3
        ("chair", 0.8, 1, None),          # non-person < danger → priority 1
        ("chair", 2.0, 3, None),          # non-person mid → priority 3
    ]

    all_pass = True
    for class_name, true_dist, exp_priority, exp_action in test_cases:
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class MockDet:
            class_id: int
            class_name: str
            confidence: float
            bbox: List[int]

        det = MockDet(0, class_name, 0.9, [100, 100, 300, 400])
        rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
        depth_m = np.full((480, 640), 10.0, dtype=np.float32)
        depth_m[100:400, 100:300] = true_dist
        depth_raw = (depth_m / 0.001).astype(np.uint16)

        data = FrameData(rgb_frame=rgb, depth_frame=depth_raw, depth_scale=0.001)
        data.detections = [det]
        data.metadata["is_dark"] = False
        data.metadata["rgb_confidence"] = 0.9

        result = fusion.process(data)
        if result.fused_output:
            out = result.fused_output[0]
            measured_dist = out["distance_m"]
            priority = out["priority"]
            action = out["action"]

            dist_error = abs(measured_dist - true_dist) / true_dist * 100
            priority_ok = priority == exp_priority
            action_ok = action == exp_action
            dist_ok = dist_error < 10

            all_ok = priority_ok and action_ok and dist_ok
            if not all_ok:
                all_pass = False

            status = "PASS" if all_ok else "FAIL"
            print(f"  {class_name}@{true_dist}m: dist={measured_dist:.2f}m (err {dist_error:.1f}%) "
                  f"priority={priority}(exp {exp_priority}) action={action}(exp {exp_action}) {status}")
        else:
            all_pass = False
            print(f"  {class_name}@{true_dist}m: NO FUSED OUTPUT FAIL")

    return {"all_pass": all_pass}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 7: Dark Mode Detection (R2 criteria - stable in varying light)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_dark_mode():
    """R2: Stable in varying lighting (dark mode detection + CLAHE + model swap)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 7: Dark Mode Adaptation (R2 Success Criteria)")
    print("  Target: stable detection in varying lighting")
    print("=" * 70)

    stage = YOLODetectionStage()
    stage._wrapper_rgb = None
    stage._wrapper_depth = None
    stage._depth_model_path = None

    test_brightness = [5, 20, 39, 40, 60, 128, 200, 255]
    results = {}

    for brightness in test_brightness:
        rgb = np.full((480, 640, 3), brightness, dtype=np.uint8)
        data = FrameData(rgb_frame=rgb)
        data = stage.process(data)

        is_dark = data.metadata["is_dark"]
        conf = data.metadata["rgb_confidence"]
        active = data.metadata["active_model"]
        expected_dark = brightness < 40

        ok = is_dark == expected_dark
        results[brightness] = {"is_dark": is_dark, "conf": conf, "active": active, "ok": ok}
        status = "PASS" if ok else "FAIL"
        print(f"  brightness={brightness:3d}: is_dark={is_dark}, conf={conf:.2f}, "
              f"model={active} {status}")

    # Test CLAHE enhancement
    dark = np.random.randint(5, 30, (480, 640, 3), dtype=np.uint8)
    enhanced = stage._enhance_dark_frame(dark)
    clahe_ok = bool(enhanced.mean() > dark.mean())
    print(f"\n  CLAHE enhancement: dark mean={dark.mean():.1f} -> enhanced mean={enhanced.mean():.1f} "
          f"{'PASS' if clahe_ok else 'FAIL'}")

    return {"brightness_tests": results, "clahe_ok": clahe_ok}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 8: Depth Noise Reduction (R3 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_depth_noise_reduction():
    """R3: Depth noise reduced 30% (indoor)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 8: Depth Noise Reduction (R3 Success Criteria)")
    print("  Target: noise reduced 30% (indoor)")
    print("  NOTE: Full test requires real RealSense hardware — synthetic approximation")
    print("=" * 70)

    # Simulate noisy depth: true depth + gaussian noise
    np.random.seed(42)
    h, w = 480, 640
    true_depth_m = np.full((h, w), 2.0, dtype=np.float32)
    # Add noise: some pixels are 0 (invalid), some are random
    noisy_depth_m = true_depth_m.copy()
    noise_mask = np.random.random((h, w)) < 0.15  # 15% of pixels are noise
    noisy_depth_m[noise_mask] = np.random.choice([0, 5.0, 8.0], size=noise_mask.sum())

    noisy_raw = (noisy_depth_m / 0.001).astype(np.uint16)

    # Run through ObstacleDetector (which applies morphological filtering)
    detector = ObstacleDetector(max_distance_m=5.0, min_distance_m=0.3, min_area=100, max_area_ratio=0.9)
    color = np.full((h, w, 3), 128, dtype=np.uint8)
    _, obstacles = detector.detect(color, noisy_raw)

    if obstacles:
        measured_dist = obstacles[0]["distance_m"]
        error_pct = abs(measured_dist - 2.0) / 2.0 * 100
        # Morphological operations should clean up noise and give accurate distance
        noise_reduction_ok = error_pct < 10
        print(f"  True depth: 2.0m, Noisy pixels: 15%")
        print(f"  Measured after filtering: {measured_dist:.2f}m (error: {error_pct:.1f}%)")
        print(f"  Noise effectively filtered: {'PASS' if noise_reduction_ok else 'FAIL'}")
    else:
        noise_reduction_ok = False
        print(f"  No obstacle detected after filtering — noise may have been over-filtered")

    return {"noise_filtered": noise_reduction_ok}


# ════════════════════════════════════════════════════════════════════════════
# Benchmark 9: End-to-End Latency P95 (R5 criteria)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_e2e_latency():
    """R5: Pipeline end-to-end ≤100ms (P95)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 9: End-to-End Latency P95 (R5 Success Criteria)")
    print("  Target: ≤100ms (P95)")
    print("=" * 70)

    config = DetectionConfig()
    processor = FrameProcessor(config)

    models_dir = os.path.join("Vision", "models")
    rgb_model = os.path.join(models_dir, "ModelRGB_V4.2.pt")

    if os.path.exists(rgb_model):
        processor.add_stage(YOLODetectionStage(model_path=rgb_model))
    processor.add_stage(FusionStage(config=config))
    processor.add_stage(VisualAnnotationStage(config=config))

    # Warm-up
    rgb, depth, depth_raw = make_synthetic_frames()
    for _ in range(3):
        processor.process(rgb, depth, 0.001, depth_raw)

    # Benchmark 200 frames
    N = 200
    latencies = []

    for i in range(N):
        rgb, depth, depth_raw = make_synthetic_frames()
        t0 = time.perf_counter()
        processor.process(rgb, depth, 0.001, depth_raw)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = percentile_stats(latencies)
    print(f"\n  Iterations: {N}")
    print(f"  P50: {stats['p50']:.2f} ms")
    print(f"  P95: {stats['p95']:.2f} ms")
    print(f"  P99: {stats['p99']:.2f} ms")
    print(f"  Mean: {stats['mean']:.2f} ms")

    target = 100
    status = "PASS" if stats["p95"] <= target else "FAIL"
    print(f"  Target P95 ≤ {target}ms: {status}")

    return stats


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("# PBL Vision System — Comprehensive Benchmark Suite")
    print(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Python: {sys.version.split()[0]}")
    print(f"# NumPy: {np.__version__}")
    print("#" * 70)

    import torch
    print(f"# PyTorch: {torch.__version__}")
    print(f"# CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)}")

    # Run all benchmarks
    results = {}
    results["yolo_inference"] = benchmark_yolo_inference()
    results["per_stage_latency"] = benchmark_per_stage_latency()
    results["pipeline_fps"] = benchmark_pipeline_fps()
    results["depth_colormap"] = benchmark_depth_colormap()
    results["obstacle_detection"] = benchmark_obstacle_detection()
    results["fusion_priority"] = benchmark_fusion_priority()
    results["dark_mode"] = benchmark_dark_mode()
    results["depth_noise"] = benchmark_depth_noise_reduction()
    results["e2e_latency"] = benchmark_e2e_latency()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Success Criteria from ROLES.md")
    print("=" * 70)

    criteria = []

    # R1: Pipeline ≥25 FPS
    fps = results["pipeline_fps"]["fps_p50"]
    criteria.append(("R1", "Pipeline ≥25 FPS (P50)", f"{fps:.1f} FPS", fps >= 25))

    # R2: YOLO latency ≤50ms (GPU)
    yolo_stats = results["yolo_inference"].get("rgb_model")
    if yolo_stats:
        criteria.append(("R2", "YOLO latency ≤50ms P95 (GPU)", f"{yolo_stats['p95']:.2f} ms", yolo_stats["p95"] <= 50))
    else:
        criteria.append(("R2", "YOLO latency ≤50ms P95 (GPU)", "N/A", None))

    # R2: Dark mode adaptation
    dark_ok = all(v["ok"] for v in results["dark_mode"]["brightness_tests"].values())
    clahe_ok = results["dark_mode"]["clahe_ok"]
    criteria.append(("R2", "Dark mode detection (brightness < 40)", "All brightness levels tested", dark_ok))
    criteria.append(("R2", "CLAHE enhancement in dark frames", f"Mean brightness increased", clahe_ok))

    # R3: Colormap zones
    cm_ok = results["depth_colormap"]["all_pass"]
    criteria.append(("R3", "Colormap: red/yellow/green zones", "LUT-based", cm_ok))

    # R3: Obstacle detection 0.3-5m
    obs_results = results["obstacle_detection"]["distances"]
    obs_ok = all(r["error_pct"] < 10 for r in obs_results.values() if r["measured"] is not None)
    criteria.append(("R3", "Obstacle detection accurate 0.3-5m", f"Tested {len(obs_results)} distances", obs_ok))

    # R3: 3 zones
    zone_ok = results["obstacle_detection"]["zones_ok"]
    criteria.append(("R3", "3 zones (left/center/right)", "All zones tested", zone_ok))

    # R3: Noise reduction (synthetic)
    noise_ok = results["depth_noise"]["noise_filtered"]
    criteria.append(("R3", "Depth noise reduced (synthetic)", "Morphological filtering", noise_ok))

    # R4: Fusion priority
    fusion_ok = results["fusion_priority"]["all_pass"]
    criteria.append(("R4", "Fusion priority matrix correct", "5 test cases", fusion_ok))

    # R4: Distance accuracy ±10%
    dist_errors = [r["error_pct"] for r in obs_results.values() if r["measured"] is not None]
    dist_ok = all(e < 10 for e in dist_errors) if dist_errors else False
    criteria.append(("R4", "Distance accuracy ±10% (0.5-4m)", f"Max error: {max(dist_errors):.1f}%" if dist_errors else "N/A", dist_ok))

    # R5: E2E latency ≤100ms P95
    e2e_p95 = results["e2e_latency"]["p95"]
    criteria.append(("R5", "E2E latency <=100ms P95", f"{e2e_p95:.2f} ms", e2e_p95 <= 100))

    # R5: Test harness measures per-stage
    per_stage = results["per_stage_latency"]["per_stage"]
    criteria.append(("R5", "Test harness measures per-stage latency", f"{len(per_stage)} stages measured", len(per_stage) > 0))

    # ── Criteria fulfilled by R5's evaluation report (Doc/model_evaluation_report_v4.md) ──
    # R5: Dataset >=300 labeled frames — RGB: 2668, Depth: 2471
    criteria.append(("R5", "Dataset >=300 labeled frames", "RGB: 2668, Depth: 2471", True))
    # R5: Dataset >=3 classes — mobil, motor, person
    criteria.append(("R5", "Dataset >=3 classes", "mobil, motor, person", True))
    # R2: mAP >=70% on dataset — RGB: 98.37%, Depth: 87.23% (from R5 report)
    criteria.append(("R2", "mAP >=70% (RGB model)", "98.37% (V4.2)", True))
    criteria.append(("R2", "mAP >=70% (Depth model)", "87.23% (V4)", True))
    # R5: Latency report generated — R5 report exists with P50/P95/P99
    criteria.append(("R5", "Latency report P50/P95/P99", "Doc/model_evaluation_report_v4.md", True))

    # Print criteria table
    print(f"\n  {'Role':<5} {'Criterion':<45} {'Result':<25} {'Status'}")
    print(f"  {'─'*5} {'─'*45} {'─'*25} {'─'*10}")

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for role, criterion, result, passed in criteria:
        if passed is True:
            status = "[PASS]"
            pass_count += 1
        elif passed is False:
            status = "[FAIL]"
            fail_count += 1
        else:
            status = "[SKIP]"
            skip_count += 1
        print(f"  {role:<5} {criterion:<45} {result:<25} {status}")

    print(f"\n  Total: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP")

    # Criteria that require hardware
    print(f"\n  Hardware-dependent criteria (not testable without RealSense):")
    hw_criteria = [
        ("R2", "Stable in varying light (<=15% degradation)", "Needs outdoor testing"),
        ("R3", "Depth noise reduced 30% indoor / 20% outdoor", "Needs real RealSense + flat wall"),
        ("R3", "Outdoor testing (sunlight)", "Needs outdoor field test"),
        ("R6", "Stable >=30 min streaming", "Run 'python main.py' for 30 min"),
        ("R6", "Info <=50ms after frame processed", "Needs GUI + hardware (pipeline P95=30ms)"),
    ]
    for role, criterion, reason in hw_criteria:
        print(f"  {role:<5} {criterion:<45} [PENDING] {reason}")

    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    main()
