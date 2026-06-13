# Progress Report — Vision System for Security Robot

> Last updated: 13 Juni 2026

---

## Overview

Proyek ini membangun sistem obstacle avoidance berbasis depth camera (Intel RealSense D455) + machine vision (YOLOv8) untuk security robot. Fase awal selesai: GUI PyQt6 + interface kamera. Fase sekarang: ML & machine vision pipeline.

---

## Status Terkini

### ✅ Completed

| Deliverable | Role | Detail |
|---|---|---|
| `FrameProcessor` pipeline | R1 (Moris) | Chain of Responsibility pattern. `FrameData` dataclass, `PipelineStage` ABC, `DepthProcessingStage`, `YOLODetectionStage` (placeholder), `FusionStage` (placeholder). 8/8 tests pass (~6ms/frame) |
| Pipeline → CameraThread integration | R1 | CameraThread menerima `FrameProcessor`, dual path (pipeline / legacy fallback). Wiring GUI thresholds |
| Depth filters (preprocessing) | R3 (Long) | Decimation → spatial → temporal → hole-filling di CameraThread via pyrealsense2 SDK |
| Multi-zone obstacle detection | R3 | LEFT / CENTER / RIGHT zone. Colormap merah/kuning/hijau. Structured output `{bbox, distance_m, zone, area_px}` |
| `ObstacleDetector` enrichment | R4 (Rasyid) | Zone calculation, priority scoring, `last_detections` attribute |
| RadarView 180° | R6 | Forward-facing semicircle. Real-time sweep (pendulum). `update_obstacles()` / `clear_obstacles()` API. 3-zone display |
| AlertPanel upgrade | R6 | Object name + distance + zone + action recommendation. Color-coded danger/warning/safe |
| ControlsPanel cleanup | R6 | Styles extracted to `styles.py`. Status constants |
| `styles.py` / `ui_config.py` | R6 | Radar colors, action thresholds, zone labels, action labels |
| Git best practices doc | R1 | `Doc/texDoc/gitBestPractices/gitBestPractices.pdf` |
| Team roles documentation | R1 | `Doc/texDoc/teamRoles/Roles.pdf` + `ROLES.md` |
| Code cleanup | R1 | Dead `roi_ratio` removed. Docstrings restored. Test moved to `tests/`. Linter ignores restored |

### ⏳ In Progress / Not Started

| Deliverable | Role | Status |
|---|---|---|
| YOLOv8 integration (`yolo_wrapper.py`) | R2 (Husein) | ❌ Belum mulai |
| Sensor fusion module (`FusionStage`) | R4 (Rasyid) | ❌ Menunggu R2 + R3 |
| Dataset acquisition & labeling | R5 (Hamid) | ❌ Belum mulai |
| Test harness & benchmark | R5 | ❌ Belum mulai |
| RadarView wiring (data nyata dari pipeline) | R6 | ❌ Menunggu R4 |
| DepthView annotation overlay | R6 | ❌ Belum mulai |
| End-to-end integration test | R1 | ❌ Belum mulai |

---

## Arsitektur Saat Ini

```
main.py → MainWindow
              ├── DepthView (3 mode: RGB / Depth / Overlay)
              ├── ControlsPanel (start/stop, thresholds, view mode)
              ├── AlertPanel (object info + zone + action)
              ├── RadarView (180° semicircle)
              └── CameraThread
                    ├── RealSense capture + depth filters
                    └── FrameProcessor
                          ├── DepthProcessingStage (colormap + multi-zone)
                          ├── YOLODetectionStage (placeholder)
                          └── FusionStage (placeholder)
```

## Data Flow

```
CameraThread (capture + filter)
    │
    └──→ FrameProcessor.process(rgb, depth)
            │
            ├── DepthProcessingStage ──→ FrameData.obstacles ──┐
            ├── YOLODetectionStage ───→ FrameData.detections ──┤
            └── FusionStage ──────────→ FrameData.fused_output  │
                                           │                    │
                                           └──→ GUI ────────────┘
                                                AlertPanel + RadarView + DepthView
```

## File Structure

```
├── main.py                          # Entry point
├── ROLES.md                         # Role assignments (AI-friendly)
├── PROGRESS.md                      # Progress documentation
├── tests/
│   └── test_frame_processor.py      # Standalone pipeline tests (8/8)
├── Vision/
│   ├── src/
│   │   ├── camera_thread.py         # Capture + filter + pipeline integration
│   │   ├── frame_processor.py       # Pipeline orchestrator
│   │   ├── obstacle_detector.py     # Depth obstacle detection
│   │   └── recorder.py              # Standalone recording utility
│   └── inc/
│       ├── detection_config.py      # Detection thresholds
│       └── camera_config.py         # (placeholder)
├── GUI/
│   ├── src/
│   │   ├── main_window.py           # Window layout + wiring
│   │   ├── depth_view.py            # Camera display (3 modes)
│   │   ├── controls_panel.py        # Start/stop + thresholds
│   │   ├── Alert_panel.py           # Object info + alert
│   │   └── radar_view.py            # 180° radar display
│   └── inc/
│       ├── ui_config.py             # UI constants + thresholds
│       └── styles.py                # Global stylesheet + color constants
└── Doc/
    └── texDoc/
        ├── teamRoles/Roles.tex      # Formal role assignments (LaTeX)
        └── gitBestPractices/        # Git workflow guide
```

## Test Coverage

| Test | Status |
|---|---|
| `test_frame_processor.py` (8 tests) | ✅ All pass (~6ms/frame) |
| Import test | ✅ |
| Instantiation test | ✅ |
| Process with depth | ✅ |
| Process without depth (webcam) | ✅ |
| Stage management (add/remove/enable/disable) | ✅ |
| Threshold update | ✅ |
| Latency report | ✅ |
| Custom stage extensibility | ✅ |

## Known Gaps

| Gap | Detail |
|---|---|
| YOLO belum terintegrasi | `YOLODetectionStage` masih placeholder |
| Fusion belum terintegrasi | `FusionStage` placeholder, menunggu R2 + R3 |
| RadarView tidak menerima data | `update_obstacles()` tidak pernah dipanggil — menunggu pipeline produce `angle_deg` |
| Tidak ada integration test | Hanya isolation test. Belum ada test CameraThread + FrameProcessor end-to-end |
| Tidak ada dataset | R5 belum mulai akuisisi & labeling |
| `camera_config.py` placeholder | Tidak ada yang menggunakannya |

## Merge History

| Branch | Role | Status |
|---|---|---|
| `feat/sensor-fusion` | R4 | ✅ Merged |
| `feat/gui-radarview` | R6 | ✅ Merged |
| `feat/gui-radarview-180` | R6 | ✅ Merged |
| `fix/depth-pipeline-integration` | R3 | ✅ Merged (after rebase + fixes) |
