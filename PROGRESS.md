# Progress Report — Vision System for Security Robot

> Last updated: 26 Juni 2026

---

## Overview

Proyek ini membangun sistem obstacle avoidance berbasis depth camera (Intel RealSense D455) + machine vision (YOLOv8) untuk security robot. Sistem telah mencapai fase production-ready dengan 4-stage pipeline, dual-model YOLO, dark mode adaptation, dan 123 tests.

---

## Status Terkini

### ✅ Completed

| Deliverable | Role | Detail |
|---|---|---|
| `FrameProcessor` pipeline | R1 (Moris) | Chain of Responsibility pattern. 4 stages: DepthProcessing, YOLODetection, Fusion, VisualAnnotation. 123/123 tests pass |
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
| Dark mode detection | R1 | brightness < 40 → `is_dark=True`. `active_model` tracks: rgb, rgb_clahe, depth, depth_filtered, none |
| `FusionStage` | R4 (Rasyid) | PASS 1: YOLO-first direct depth sampling. PASS 2: depth-only obstacles. Adaptive overlap threshold (0.3 dark, 0.5 normal). Priority matrix from DetectionConfig |
| `VisualAnnotationStage` | R1 | HUD corner brackets, labels, global status bar (SAFE/WARN/DANGER). Renders in-place on rgb_frame |
| RadarView 180° | R6 | Cached static background pixmap. Only sweep + blips redrawn. 20fps timer |
| AlertPanel optimization | R6 | Status-change-only stylesheet updates (no redundant recalc). Pre-computed style dicts |
| DepthView optimization | R6 | setScaledContents once in init. Only updates visible page labels |
| CameraThread optimization | R1 | Removed msleep (queue provides flow control). Cached empty depth QImage. numpy BGR→RGB swap |
| ControlsPanel cleanup | R6 | Styles extracted to `styles.py`. Status constants |
| Git best practices doc | R1 | `Doc/texDoc/gitBestPractices/gitBestPractices.pdf` |
| Team roles documentation | R1 | `Doc/texDoc/teamRoles/Roles.pdf` + `ROLES.md` |
| Model evaluation report | R5 (Hamid) | `Doc/model_evaluation_report_v4.md` — V4.2 RGB (98.37% mAP) + V4 Depth (87.23% mAP) |
| Dataset acquisition guide | R5 | `data-collection.md` — comprehensive guide for R5 |
| Test suite | R1 | 123 tests: 69 frame_processor + 31 obstacle_detector + 24 camera_thread. All pass in ~9s |

### ⏳ In Progress / Not Started

| Deliverable | Role | Status |
|---|---|---|
| Dataset acquisition (≥300 frames) | R5 (Hamid) | ❌ Not started |
| Test harness & benchmark | R5 | ❌ Not started |
| Fine-tune YOLOv8 with dataset | R2 (Husein) | ❌ Waiting on R5 |
| mAP accuracy benchmark on deployment hardware | R2 | ❌ Not started — need ≥70% mAP@0.5 |
| Outdoor testing (sunlight) | R3 (Long) | ❌ Not started |
| Full pipeline test with real hardware | R1 | ⏳ Pending — all software tests pass |

---

## Role Completion Summary

| Role | Completed | Remaining | % Complete |
|------|-----------|-----------|------------|
| R1 (Moris) — ML Pipeline | 28 items | 1 (hardware test) | 97% |
| R2 (Husein) — YOLOv8 | 5 items | 2 (fine-tune, mAP) | 71% |
| R3 (Long) — Depth | 5 items | 1 (outdoor test) | 83% |
| R4 (Rasyid) — Fusion | 5 items | 0 | 100% |
| R5 (Hamid) — Dataset | 3 items | 3 (dataset, harness, regression) | 50% |
| R6 (Adel) — GUI | 10 items | 0 | 100% |

---

## Arsitektur Saat Ini

```
main.py → MainWindow
              ├── DepthView (3 mode: RGB / Depth / Overlay, visible-only updates)
              ├── ControlsPanel (start/stop, thresholds, view mode)
              ├── AlertPanel (object info + zone + action, cached stylesheets)
              ├── RadarView (180° semicircle, cached static background)
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
                                └── VisualAnnotationStage (HUD rendering, in-place)
```

## Data Flow

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
                                      └── VisualAnnotationStage
                                            └── rgb_frame (HUD, in-place modification)
                                      │
                                      └──→ Signals → GUI
                                             ├── frame_pair_ready → DepthView
                                             ├── distance_info_ready → AlertPanel
                                             └── obstacles_ready → RadarView
```

## Performance Optimizations

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

## Test Coverage

| Test File | Tests | Coverage |
|---|---|---|
| `test_frame_processor.py` | 69 | FrameData, PipelineStage (disabled, latency, exception), FrameProcessor (stages, thresholds, errors), DepthProcessingStage (LUT colors, raw, rebuild), FusionStage (matching, priority, zones, overlap, dark mode, PASS 2 ladder, contract), YOLODetectionStage (dark/bright/CLAHE/dual-model/none), VisualAnnotationStage (none/empty/fused/obstacles/yolo/danger/in-place), full pipeline integration |
| `test_obstacle_detector.py` | 31 | Instantiation, edge cases (None/zero), zones (single + multi), filtering (min_area, max_area_ratio, distance), priority (inverse, no div-by-zero), distance accuracy, frame handling (no copy regression, no modification), buffer reuse, thread safety, output contract, last_detections copy |
| `test_camera_thread.py` | 24 | Instantiation, thresholds (validation + propagation), BGR→QImage (pixel integrity, grayscale, dimensions), empty depth cache (cached + shape change), thread lifecycle, signals |
| **Total** | **123** | All pass in ~9s |

## Known Gaps

| Gap | Detail |
|---|---|
| Belum ada dataset terlabel | R5 belum mengumpulkan ≥300 frame berlabel |
| Belum ada akurasi benchmark | Tidak ada pengukuran mAP@0.5 pada deployment hardware |
| Belum ada outdoor testing | RealSense D455 terganggu sinar matahari langsung |
| Hardware test pending | All 123 tests use synthetic data — real hardware test needed |
| Model weights belum di repo | `ModelRGB_V4.2.pt` dan `ModelDepth_V4.pt` di `.gitignore` |

## Merge History

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
