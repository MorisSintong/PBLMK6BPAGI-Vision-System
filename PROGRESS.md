# Progress Report — Vision System for Security Robot

> Last updated: 26 Juni 2026

---

## Overview

Proyek ini membangun sistem obstacle avoidance berbasis depth camera (Intel RealSense D455) + machine vision (YOLOv8) untuk security robot. Sistem telah mencapai fase production-ready dengan 5-stage pipeline, dual-model YOLO, dark mode adaptation, gap-based navigation, auto-switch view, dan 147 tests.

---

## Status Terkini

### ✅ Completed

| Deliverable | Role | Detail |
|---|---|---|
| `FrameProcessor` pipeline | R1 (Moris) | Chain of Responsibility pattern. 5 stages: DepthProcessing, YOLODetection, Fusion, Navigation, VisualAnnotation. 147/147 tests pass |
| Pipeline → CameraThread integration | R1 | Separate acquisition thread + queue(maxsize=2). Camera capture decoupled from processing |
| Depth filters (preprocessing) | R3 (Long) | Decimation (configurable) → spatial → temporal → hole-filling di CameraThread via pyrealsense2 SDK |
| Unfiltered depth capture | R1 | `depth_frame_raw` captured BEFORE RS filters for depth model inference |
| Multi-zone obstacle detection | R3 | LEFT / CENTER / RIGHT zone. LUT-based colormap (merah/kuning/hijau). Structured output |
| `ObstacleDetector` optimization | R1 | No frame copy (returns same object), no zone tick drawing, reusable float32 buffer |
| LUT depth colormap | R1 | Pre-computed 256-entry LUT, ~3x faster than mask approach. Rebuilds on threshold change |
| `YOLOWrapper` optimization | R1/R2 | FP16 inference (auto-detected), GPU warm-up, input_size=320, batch tensor transfer |
| Dual-model swap | R1/R5 | `ModelRGB_V4.2.pt` (normal) + `ModelDepth_V4.pt` (dark mode on unfiltered depth colormap) |
| Lazy depth model loading | R1 | Depth model loaded only on first dark frame — saves VRAM at startup |
| CLAHE dark mode | R1 | LAB color space enhancement (clipLimit=3.0) for dim scenes. `rgb_confidence` metadata |
| Dark mode detection (hysteresis) | R1 | brightness < 35 enter dark, > 50 exit dark. `active_model` tracks: rgb, rgb_clahe, depth, depth_filtered, none |
| `FusionStage` | R4 (Rasyid) | PASS 1: YOLO-first direct depth sampling. PASS 2: depth-only obstacles. Adaptive overlap threshold (0.3 dark, 0.5 normal). Priority matrix from DetectionConfig |
| `VisualAnnotationStage` | R1 | HUD corner brackets, labels, global status bar (SAFE/WARN/DANGER), steering arrow. Renders in-place on both rgb_frame and depth_colormap (auto-switch view support) |
| Auto-switch RGB/Depth view | R1 | Auto/RGB/Depth buttons. Auto mode follows is_dark, manual override available. Hysteresis prevents flicker. Overlay page removed |
| RadarView 90° FOV | R6 | Cached static background pixmap. Only sweep + blips redrawn. 20fps timer |
| AlertPanel optimization | R6 | Status-change-only stylesheet updates (no redundant recalc). Pre-computed style dicts |
| DepthView optimization | R6 | setScaledContents once in init. Only updates visible page labels (RGB / Depth, no Overlay) |
| CameraThread optimization | R1 | Removed msleep (queue provides flow control). Cached empty depth QImage. numpy BGR→RGB swap |
| ControlsPanel cleanup | R6 | Styles extracted to `styles.py`. Status constants |
| Git best practices doc | R1 | `Doc/texDoc/gitBestPractices/gitBestPractices.pdf` |
| Team roles documentation | R1 | `Doc/texDoc/teamRoles/Roles.pdf` + `ROLES.md` |
| Model evaluation report | R5 (Hamid) | `Doc/model_evaluation_report_v4.md` — V4.2 RGB (98.37% mAP) + V4 Depth (87.23% mAP) |
| Dataset acquisition guide | R5 | `data-collection.md` — comprehensive guide for R5 |
| Test suite | R1 | 147 tests: 92 frame_processor + 31 obstacle_detector + 24 camera_thread. All pass in ~24s |
| Benchmark suite | R1/R5 | `tests/benchmark.py` — 17/17 software criteria PASS (42.5 FPS, P95 30.50ms) |
| NavigationStage (gap-based steering) | R1 | Polar histogram (18 sectors), gap finding + scoring, hysteresis, safety override, speed mapping |
| Dataset acquisition (≥300 frames) | R5 (Hamid) | ✅ RGB: 2668 frames, Depth: 2471 frames |
| Dataset classes (≥3) | R5 | ✅ mobil, motor, person |
| mAP accuracy benchmark | R5 | ✅ RGB 98.37%, Depth 87.23% (>=70% target) |
| Latency report (P50/P95/P99) | R5 | ✅ `Doc/model_evaluation_report_v4.md` |
| Field Test Report (Realtime FPS) | R5 | ✅ `Doc/field_test_report_role5.md` — Analisis 2.032 frame (Siang/Malam) |
### ⏳ In Progress / Not Started

| Deliverable | Role | Status |
|---|---|---|
| Light stability outdoor test (≤15% degradation) | R2 (Husein) | ⏳ Needs outdoor testing |
| Depth noise reduction on hardware (30%) | R3 (Long) | ⏳ Needs RealSense + flat wall |
| Outdoor testing (sunlight) | R3 (Long) | ⏳ Needs outdoor field test |
| 30-min streaming stability | R6 (Adel) | ⏳ Run `python main.py` for 30 min |
| Display latency ≤50ms | R6 | ⏳ Needs GUI + hardware (pipeline P95=30ms) |
| Regression test otomatis | R5 | ⏳ Not started |
| Full pipeline test with real hardware | R1 | ⏳ Pending — all software tests pass |

---

## Ringkasan Completion Per Role

| Role | Completed | Remaining | % Complete |
|------|-----------|-----------|------------|
| R1 (Moris) — ML Pipeline | 28 items | 1 (hardware test) | 97% |
| R2 (Husein) — YOLOv8 | 7 items | 1 (light stability outdoor) | 88% |
| R3 (Long) — Depth | 5 items | 2 (noise reduction on HW, outdoor test) | 71% |
| R4 (Rasyid) — Fusion | 5 items | 0 | 100% |
| R5 (Hamid) — Dataset | 7 items | 1 (regression test) | 88% |
| R6 (Adel) — GUI | 10 items | 2 (30-min soak, display latency) | 83% |

---

## Arsitektur Saat Ini

```
main.py → MainWindow
              ├── DepthView (2 mode: RGB / Depth, visible-only updates, auto-switch by light)
              ├── ControlsPanel (start/stop, thresholds, Auto/RGB/Depth view mode)
              ├── AlertPanel (object info + zone + action, cached stylesheets)
              ├── RadarView (90° FOV wedge, cached static background)
              └── CameraThread
                    ├── Acquisition thread (separate, queue maxsize=2)
                    │     ├── RealSense capture + depth filters
                    │     └── Unfiltered depth capture (before filters)
                    └── Processing loop (QThread)
                          └── FrameProcessor (Chain of Responsibility)
                                ├── DepthProcessingStage (LUT colormap + multi-zone)
                                ├── YOLODetectionStage (dual-model swap + CLAHE)
                                │     ├── Normal: ModelRGB_V4.2.pt on RGB
                                │     ├── Dim: CLAHE + ModelRGB_V4.2.pt
                                │     └── Dark: ModelDepth_V4.pt on unfiltered depth
                                ├── FusionStage (overlap matching + priority matrix)
                                │     ├── PASS 1: YOLO-first direct depth sampling
                                │     └── PASS 2: depth-only obstacles
                                ├── NavigationStage (gap-based steering, VFH-lite)
                                │     ├── Polar histogram (18 sectors)
                                │     ├── Gap finding + scoring + hysteresis
                                │     └── Safety override (priority 0 → STOP)
                                └── VisualAnnotationStage (HUD + steering arrow, in-place)
```

## Alur Data

```
CameraThread (acquisition thread)
    │
    ├── color_bgr, depth_raw (filtered), depth_raw_unfiltered
    │
    └──→ queue(maxsize=2) ──→ Processing loop
                                │
                                └──→ FrameProcessor.process()
                                      │
                                      ├── DepthProcessingStage
                                      │     ├── depth_colormap (LUT, for display)
                                      │     ├── depth_colormap_raw (LUT, for depth model)
                                      │     └── obstacles (contour detection)
                                      │
                                      ├── YOLODetectionStage
                                      │     ├── brightness detection → is_dark
                                      │     ├── model swap (rgb / rgb_clahe / depth)
                                      │     └── detections (List[Detection])
                                      │
                                      ├── FusionStage
                                      │     ├── PASS 1: YOLO + direct depth sampling
                                      │     ├── PASS 2: depth-only obstacles
                                      │     └── fused_output (class + distance + priority)
                                      │
                                      ├── NavigationStage
                                      │     ├── Polar histogram (18 sectors)
                                      │     ├── Gap finding + scoring + hysteresis
                                      │     └── navigation (steering + speed + status)
                                      │
                                      └── VisualAnnotationStage
                                            ├── rgb_frame (HUD + steering arrow, in-place)
                                            └── depth_colormap (HUD + steering arrow, in-place)
                                      │
                                      └──→ Signals → GUI
                                             ├── frame_pair_ready → DepthView
                                             ├── distance_info_ready → AlertPanel
                                             ├── obstacles_ready → RadarView
                                             ├── navigation_ready → AlertPanel + RadarView
                                             ├── light_mode_changed → ControlsPanel (auto-switch)
                                             └── error → MainWindow (camera failure alert)
```

## Optimasi Performa

| Optimization | File | Impact |
|---|---|---|
| FP16 inference | yolowrapper.py | ~2x faster on Tensor Cores |
| GPU warm-up | yolowrapper.py | First frame no longer slow |
| Input size 320 (was 416) | yolowrapper.py | ~40% fewer pixels |
| Batch tensor transfer | yolowrapper.py | Fewer CPU↔GPU transfers |
| LUT depth colormap | frame_processor.py | ~3x faster colormap |
| numpy BGR→RGB swap | camera_thread.py | Faster than cv2.cvtColor |
| Cached empty depth QImage | camera_thread.py | ~1MB alloc eliminated |
| Removed msleep | camera_thread.py | Eliminates jitter |
| No frame copy in ObstacleDetector | obstacle_detector.py | ~1MB copy eliminated |
| Cached radar background | radar_view.py | ~80% less paint work |
| Status-change-only stylesheets | alert_panel.py | 6→0 style recalcs/frame |
| setScaledContents once | depth_view.py | 6 fewer layout passes/frame |
| Visible-only label updates | depth_view.py | 2 fewer pixmap sets/frame |
| Lazy depth model loading | frame_processor.py | Faster startup, less VRAM |

## Cakupan Test

| Test File | Tests | Coverage |
|---|---|---|
| `test_frame_processor.py` | 92 | FrameData, PipelineStage (disabled, latency, exception), FrameProcessor (stages, thresholds, errors), DepthProcessingStage (LUT colors, raw, rebuild), FusionStage (matching, priority, zones, overlap, dark mode, PASS 2 ladder, contract), YOLODetectionStage (dark/bright/CLAHE/dual-model/hysteresis/none), NavigationStage (clear/blocked/steering/safety override/speed/output contract), VisualAnnotationStage (none/empty/fused/obstacles/yolo/danger/in-place/depth colormap/nav HUD), full pipeline integration |
| `test_obstacle_detector.py` | 31 | Instantiation, edge cases (None/zero), zones (single + multi), filtering (min_area, max_area_ratio, distance), priority (inverse, no div-by-zero), distance accuracy, frame handling (no copy regression, no modification), buffer reuse, thread safety, output contract, last_detections copy |
| `test_camera_thread.py` | 24 | Instantiation, thresholds (validation + propagation), BGR→QImage (pixel integrity, grayscale, dimensions), empty depth cache (cached + shape change), thread lifecycle, signals (frame_pair, distance, obstacles, navigation, light_mode) |
| **Total** | **147** | All pass in ~24s |

## Hasil Benchmark (RTX A4000 Laptop GPU, FP16, 320px)

| Criterion | Target | Result | Status |
|---|---|---|---|
| R1: Pipeline FPS (P50) | >=25 FPS | 42.5 FPS | PASS |
| R2: YOLO RGB latency P95 | <=50 ms | 12.94 ms | PASS |
| R2: YOLO Depth latency P95 | <=50 ms | 24.93 ms | PASS |
| R2: mAP@0.5 (RGB model) | >=70% | 98.37% (R5 report) | PASS |
| R2: mAP@0.5 (Depth model) | >=70% | 87.23% (R5 report) | PASS |
| R2: Dark mode + CLAHE | Hysteresis: <35 enter, >50 exit | All 10 levels correct | PASS |
| R3: Colormap zones | red/yellow/green/black | All 4 correct | PASS |
| R3: Obstacle accuracy 0.3-5m | +-10% | Max error 0.2% | PASS |
| R3: 3 zones | left/center/right | All correct | PASS |
| R3: Noise reduction (synthetic) | Morphological filtering | 0.1% error with 15% noise | PASS |
| R4: Fusion priority matrix | 5 test cases | 5/5 correct | PASS |
| R4: Distance accuracy | +-10% | Max error 0.2% | PASS |
| R5: Dataset >=300 frames | >=300 | RGB: 2668, Depth: 2471 | PASS |
| R5: Dataset >=3 classes | >=3 | mobil, motor, person | PASS |
| R5: E2E latency P95 | <=100 ms | 30.50 ms | PASS |
| R5: Per-stage latency harness | All stages | 5 stages measured | PASS |
| **Total** | | **17/17 PASS** | |

## Gap yang Diketahui

| Gap | Detail |
|---|---|
| R2: Light stability outdoor | Needs same scene tested under bright vs dim vs night (<=15% degradation) |
| R3: Depth noise on hardware | Needs real RealSense + flat wall to measure 30% reduction |
| R3: Outdoor sunlight test | RealSense D455 terganggu sinar matahari langsung |
| R6: 30-min streaming stability | Run `python main.py` for 30 min, watch for crashes/memory growth |
| R6: Display latency <=50ms | Needs GUI + hardware (pipeline P95 is 30ms, should pass) |
| R5: Regression test otomatis | Automated regression test not yet built |
| Model weights belum di repo | `ModelRGB_V4.2.pt` dan `ModelDepth_V4.pt` di `.gitignore` |

## Riwayat Merge

| Branch | Role | Status |
|---|---|---|
| `feat/sensor-fusion` | R4 | ✅ Merged |
| `feat/gui-radarview` | R6 | ✅ Merged |
| `feat/gui-radarview-180` | R6 | ✅ Merged |
| `fix/depth-pipeline-integration` | R3 | ✅ Merged |
| `yolowrapper-clean` | R2 + R1 | ✅ Merged |
| `fix/audit-issues` | R1 | ✅ Merged |
| `feature/fusion-stage` | R1/R4 | ✅ Merged |
| `arch/optimizations` | R1 | ✅ Merged |
| `swap_model` | R1/R5 | ✅ Merged (dual-model + CLAHE + unfiltered depth) |
| `feature/navigation-stage` | R1 | ✅ Merged (NavigationStage + auto-switch view + HUD on depth + hysteresis) |
