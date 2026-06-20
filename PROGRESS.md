# Progress Report — Vision System for Security Robot

> Last updated: 20 Juni 2026 (Comprehensive Status Update)

---

## Overview

Proyek ini membangun sistem obstacle avoidance berbasis depth camera (Intel RealSense D455) + machine vision (YOLOv8) untuk security robot. Fase awal selesai: GUI PyQt6 + interface kamera. Fase sekarang: ML & machine vision pipeline.

---

## Status Terkini

### ✅ Completed

| Deliverable | Role | Detail |
|---|---|---|
| `FrameProcessor` pipeline | R1 (Moris) | Chain of Responsibility pattern. `FrameData` dataclass, `PipelineStage` ABC, `DepthProcessingStage`, `YOLODetectionStage`, `FusionStage`. 16/16 tests pass (~6ms/frame) |
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
| **Foundation & Infrastructure Phase** | | |
| Thread safety fix | R1 | CameraThread now emits QImage (thread-safe) instead of QPixmap. DepthView converts to QPixmap in main thread. Prevents Qt violations and crashes |
| Division by zero fix | R1 | `obstacle_detector.py:101` — Added safety check `max(distance, 0.01)` to prevent crash when distance=0 |
| Error handling improvements | R1 | `recorder.py` — Added try-except around `wait_for_frames()`. Proper error reporting instead of silent failures |
| Pytest infrastructure | R1 | Added `pytest==8.0.0` + `pytest-qt==4.4.0` to `environment.yml`. Configured `pyproject.toml` with test paths and pythonpath |
| Test: obstacle_detector | R1 | 13 tests covering instantiation, detection, zones, filtering, division safety, output format. All use synthetic frames |
| Test: camera_thread | R1 | 15 tests covering instantiation, thresholds, QImage conversion, signals, processor integration. No camera hardware required |
| Frame rate control | R1 | Added `DISPLAY_FPS` import and `_frame_delay_ms` calculation. Sleep at end of each loop prevents CPU waste from unbounded frame rate |
| Logging framework | R1 | Created `Vision/inc/logging_config.py` with console + file output. Replaced all `print()` statements with proper logging across codebase |
| Camera config implementation | R1 | Filled `camera_config.py` with RealSense D455 settings, webcam fallback settings, depth filter parameters, helper methods |
| Code quality: file naming | R1 | Renamed `Alert_panel.py` → `alert_panel.py` (PEP 8 compliance). Updated import in `main_window.py` |
| **Critical Fixes (19 Juni 2026)** | | |
| RealSense pipeline conflict fix | R3 | Added `_active_pipeline_count` flag to `FrameRecorder` to prevent crash when camera is already active |
| Radar view connection | R4 | Added `obstacles_ready` signal to CameraThread. Connected to RadarView via angle conversion from bbox position |
| Thread safety — last_detections | R1 | Added `threading.Lock` + property getter/setter for `ObstacleDetector.last_detections` |
| Float32 buffer reuse | R1 | Added reusable `_depth_buffer` with `np.multiply(..., out=...)` to avoid ~1.2MB allocation per frame |
| **Audit Issues Resolution (20 Juni 2026)** | | |
| Depth Resize Artifacts | R3 | Removed `decimation_filter` and `cv2.resize` natively eliminating blocky edges |
| QImage Memory Safety | R1 | Replaced raw pointer `.data` with `.tobytes()` avoiding thread-based segmentation faults |
| Alert Threshold Wiring | R6 | Piped `controls_panel.thresholds_changed` directly to `FrameProcessor.set_action_thresholds` |
| YOLO Model Path Robustness | R6 | Switched to `pathlib.Path` for reliable absolute paths to model weights |
| Pipeline Error Catching | R1 | Wrapped `PipelineStage._measure()` with exception handler pushing to `FrameData.errors` |
| Python Package Structure | R6 | Included `__init__.py` files across all `Vision` and `GUI` source directories |
| Signal Typing Strictness | R1 | Upgraded generic `object` signals in `CameraThread` to strictly typed `QImage` signals |
| **Performance Optimizations** | | |
| Delta Sleep FPS Fix | R1 | Replaced fixed `msleep` with `time.time()` delta sleep in `CameraThread` preventing double-blocking latency |
| YOLO FP16 Inference | R2 | ~~Added `half=True` to YOLO predict call~~ REMOVED — Benchmark showed FP16 is 3% slower on RTX A4000 Laptop (small model overhead) |
| OpenCV Kernel Optimized | R1 | Replaced `np.ones` with `cv2.getStructuringElement` in `ObstacleDetector` for morphology operations |
| **YOLOv8 Integration (R2 + R1)** | | |
| `YOLOWrapper` integration | R2 (Husein) + R1 | Cherry-picked `yolowrapper.py` from R2 branch. Removed torch.load security bypass. Added logging. Wired into `YOLODetectionStage` pipeline |
| `Detection` dataclass contract | R2 | Output format `{bbox, class_id, class_name, confidence}` compatible with `FrameData.detections` for R4 (Sensor Fusion) |
| `YOLODetectionStage` wired | R1 | No longer a placeholder. Instantiates `YOLOWrapper` with configurable model_path, conf_threshold, input_size. Graceful fallback if ultralytics not installed |
| `.gitignore` for model weights | R1 | Added `*.pt`, `*.onnx`, `*.engine`, `Vision/models/`, `logs/` to prevent committing large binary files |
| `data.yaml` dataset config | R2 | Roboflow dataset with 3 classes: mobil, motor, person |

### ⏳ In Progress / Not Started

| Deliverable | Role | Status |
|---|---|---|
| Sensor fusion module (`FusionStage`) | R4 (Rasyid) | ✅ Implemented — overlap matching, priority matrix, config thresholds, 8 tests |
| Dataset acquisition & labeling | R5 (Hamid) | ❌ Not started — need ≥300 frames |
| Test harness & benchmark | R5 | ❌ Not started |
| DepthView annotation overlay | R6 (Adel) | ❌ Not started — bbox + label + distance |
| End-to-end integration test | R1 | ❌ Not started |
| Fine-tune YOLOv8 with dataset | R2 (Husein) | ❌ Waiting on R5 |
| mAP accuracy benchmark | R2 | ❌ Not started — need ≥70% mAP@0.5 |
| Outdoor testing (sunlight) | R3 (Long) | ❌ Not started |

---

## Role Completion Summary

| Role | Completed | Remaining | % Complete |
|------|-----------|-----------|------------|
| R1 (Moris) — ML Pipeline | 19 items | 1 (integration test) | 95% |
| R2 (Husein) — YOLOv8 | 3 items | 2 (fine-tune, mAP) | 60% |
| R3 (Long) — Depth | 4 items | 1 (outdoor test) | 80% |
| R4 (Rasyid) — Fusion | 3 items | 0 (FusionStage implemented, priority done) | 100% |
| R5 (Hamid) — Dataset | 1 item | 3 (dataset, harness, regression) | 25% |
| R6 (Adel) — GUI | 8 items | 1 (DepthView overlay) | 89% |

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
                          ├── YOLODetectionStage (YOLOv8 via YOLOWrapper)
                          └── FusionStage (YOLO + Depth matching, priority)
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
├── problems.md                      # Critical audit report + issue tracking
├── tests/
│   ├── test_frame_processor.py      # Standalone pipeline tests (16/16)
│   ├── test_obstacle_detector.py    # ObstacleDetector tests (16 tests)
│   └── test_camera_thread.py        # CameraThread tests (15 tests)
├── Vision/
│   ├── src/
│   │   ├── camera_thread.py         # Capture + filter + pipeline integration (3 signals)
│   │   ├── frame_processor.py       # Pipeline orchestrator
│   │   ├── obstacle_detector.py     # Depth obstacle detection
│   │   ├── recorder.py              # Standalone recording utility
│   │   ├── yolowrapper.py           # YOLOv8 inference wrapper (R2)
│   │   ├── fusion.md                # FusionStage documentation & bug tracking
│   │   └── data.yaml                # Roboflow dataset config (3 classes)
│   ├── models/                      # (.gitignore) Model weights directory
│   │   ├── security_best.pt         # Trained model (not tracked)
│   │   └── yolov8n.pt               # Pre-trained YOLOv8-nano (not tracked)
│   └── inc/
│       ├── detection_config.py      # Detection thresholds
│       ├── camera_config.py         # Camera parameters (RealSense + webcam)
│       └── logging_config.py        # Centralized logging framework
├── GUI/
│   ├── src/
│   │   ├── main_window.py           # Window layout + wiring
│   │   ├── depth_view.py            # Camera display (3 modes)
│   │   ├── controls_panel.py        # Start/stop + thresholds
│   │   ├── alert_panel.py           # Object info + alert
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
| `test_frame_processor.py` (16 tests) | ✅ All pass |
| Import test | ✅ |
| Instantiation test | ✅ |
| Process with depth | ✅ |
| Process without depth (webcam) | ✅ |
| Stage management (add/remove/enable/disable) | ✅ |
| Threshold update | ✅ |
| Latency report | ✅ |
| Custom stage extensibility | ✅ |
| Fusion matching (YOLO→depth) | ✅ |
| Fusion no match fallback | ✅ |
| Fusion priority person close | ✅ |
| Fusion priority obstacle close | ✅ |
| Fusion empty inputs | ✅ |
| Fusion bbox format (xyxy) | ✅ |
| Fusion overlap with area_px | ✅ |
| Fusion config thresholds | ✅ |
| `test_obstacle_detector.py` (16 tests) | ✅ All pass |
| Instantiation & custom params | ✅ |
| Detect no obstacles | ✅ |
| Detect none inputs | ✅ |
| Detect with obstacle | ✅ |
| Zone center/left/right | ✅ |
| Min area filter | ✅ |
| Priority no division by zero | ✅ |
| Last detections updated | ✅ |
| Annotated frame has status | ✅ |
| Output format contract | ✅ |
| Depth buffer reuse | ✅ |
| Depth buffer resize on shape change | ✅ |
| Thread safety last_detections | ✅ |
| `test_camera_thread.py` (15 tests) | ✅ All pass |
| Instantiation & custom camera index | ✅ |
| Depth thresholds (valid/invalid) | ✅ |
| BGR to QImage conversion | ✅ |
| Stop/start capture | ✅ |
| Release resources | ✅ |
| Signals defined | ✅ |
| Processor integration | ✅ |

## Known Gaps

| Gap | Detail |
|---|---|
| FusionStage belum terimplementasi | ~~`FusionStage` masih placeholder, menunggu R4~~ ✅ IMPLEMENTED — overlap matching, priority, config thresholds |
| DepthView tanpa anotasi | Bounding box + label + jarak belum ditampilkan di DepthView |
| Belum ada dataset terlabel | R5 belum mengumpulkan ≥300 frame berlabel |
| Belum ada integration test | Hanya isolation test, belum ada test CameraThread + FrameProcessor end-to-end |
| Belum ada akurasi benchmark | Tidak ada pengukuran mAP@0.5 |
| Belum ada outdoor testing | RealSense D455 terganggu sinar matahari langsung |
| Model weights belum di repo | `security_best.pt` dan `yolov8n.pt` di `.gitignore` — R2 perlu download manual |

## Merge History

| Branch | Role | Status |
|---|---|---|
| `feat/sensor-fusion` | R4 | ✅ Merged |
| `feat/gui-radarview` | R6 | ✅ Merged |
| `feat/gui-radarview-180` | R6 | ✅ Merged |
| `fix/depth-pipeline-integration` | R3 | ✅ Merged (after rebase + fixes) |
| `yolowrapper-clean` | R2 + R1 | ✅ Merged (YOLO integration with security/quality fixes) |
