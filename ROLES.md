# Team Roles & Responsibilities

> AI Agent: Read this document to understand your role. Work according to the goals and success criteria. Coordinate with related roles as per the relationship table.

---

## Role 1 — ML Pipeline Architect / Frame Processor Lead
**Person:** Moris

### Responsibilities
- Design and implement the end-to-end vision pipeline (5 stages)
- Define data contracts between stages
- Orchestrate data flow: Raw Frame → Depth → YOLO → Fusion → Navigation → Annotation → Output
- Code review for all PRs from Roles 2-6
- Performance optimization (FP16, LUT, lazy loading, buffer reuse)

### Files
| File | Description |
|---|---|
| `Vision/src/frame_processor.py` | Main pipeline (5 stages: Depth, YOLO, Fusion, Navigation, Annotation) |
| `Vision/src/camera_thread.py` | Pipeline integration + acquisition thread |
| `Vision/src/yolowrapper.py` | YOLOv8 wrapper (FP16, warm-up, batch transfer) |
| `Vision/inc/detection_config.py` | Centralized configuration |
| `Vision/inc/camera_config.py` | Camera configuration (RealSense + webcam) |
| `main.py` | FrameProcessor initialization |

### Done when
- `frame_processor.py` receives frames from CameraThread, runs all 5 stages, returns annotated frames
- Pipeline ≥25 FPS (RealSense) / ≥30 FPS (webcam)
- All stage contracts documented and approved by team
- All config files managed as single source of truth
- 147/147 tests pass

---

## Role 2 — YOLOv8 Object Detection Specialist
**Person:** Husein

### Responsibilities
- Build `YOLOWrapper` class for model loading, inference, class mapping
- Fine-tune YOLOv8 with dataset from Role 5
- Inference optimization (ONNX, TensorRT, reduced input size)
- Per-frame output: `List[Detection]` (dataclass) in `{bbox, class_id, class_name, confidence}` format

### Files
| File | Description |
|---|---|
| `Vision/src/yolowrapper.py` | YOLO wrapper (FP16, warm-up, 320px, batch transfer) |
| `Vision/models/ModelRGB_V4.2.pt` | RGB model (latest) |
| `Vision/models/ModelDepth_V4.pt` | Depth model (latest, trained on unfiltered depth) |
| `environment.yml` | Dependency ultralytics 8.4.77 |

### Input from
- **Role 5** — labeled train/val dataset

### Output to
- **Role 4** — `FrameData.detections` (List[Detection] dataclass)

### Output contract format
```python
@dataclass
class Detection:
    class_id: int        # COCO class index
    class_name: str      # "person", "mobil", "motor"
    confidence: float    # 0.0 - 1.0
    bbox: List[int]      # [x1, y1, x2, y2] xyxy format
```

### Done when
- YOLOv8 model runs inference on every RGB frame in the pipeline
- Dual-model: RGB model + Depth model for dark mode
- Latency ≤50ms (GPU) / ≤100ms (CPU)
- Accuracy ≥70% mAP@0.5 on target classes in outdoor environments
- Stable under varying lighting (accuracy degradation ≤15%)
- Clean and documented API

---

## Role 3 — Depth Processing & Obstacle Detection Engineer
**Person:** Long

### Responsibilities
- Depth filtering: temporal, spatial edge-preserving, hole-filling, decimation (via pyrealsense2 SDK)
- Multi-zone detection: LEFT / CENTER / RIGHT
- Depth colormap: red (danger), yellow (warning), green (safe) — now LUT-based
- Unfiltered depth capture before filters (for depth model)
- Obstacle detection with bounding box + distance label

### Files
| File | Description |
|---|---|
| `Vision/src/camera_thread.py` | Depth filtering, unfiltered capture, acquisition thread |
| `Vision/src/obstacle_detector.py` | Obstacle detection (no frame copy, buffer reuse) |
| `Vision/inc/camera_config.py` | RealSense D455 settings + filter parameters |

### Output to
- **Role 4** — `FrameData.obstacles` (List[Dict])
- **Role 5** — Depth module for benchmark

### Output contract format
```python
[
    {
        "bbox":        [x, y, w, h],  # bounding box (xywh)
        "distance_m":  float,          # distance in meters
        "zone":        "left"|"center"|"right",
        "area_px":     int,            # cv2.contourArea
        "priority":    float,          # inverse distance (raw)
    },
    ...
]
```

### Done when
- ObstacleDetector runs in real-time, accurate for objects 0.3m–5m
- Depth noise reduced by 30% (indoor) / 20% (outdoor) from raw
- Colormap displays red/yellow/green zones according to thresholds (LUT-based)
- 3 sectors (left/center/right) with minimum distance per sector
- Unfiltered depth frame available for depth model

### Outdoor notes
RealSense D455 is affected by direct sunlight. Test in morning/afternoon, cloudy, or shaded areas.

---

## Role 4 — Sensor Fusion Engineer
**Person:** Rasyid

### Responsibilities
- Merge YOLO detections (R2) + depth obstacles (R3)
- Two-pass architecture: PASS 1 YOLO-first direct depth sampling, PASS 2 depth-only obstacles
- Obstacle priority: person close > obstacle close > others
- Adaptive overlap threshold (0.3 dark, 0.5 normal)

### Files
| File | Description |
|---|---|
| `Vision/src/frame_processor.py` | FusionStage implementation |
| `Vision/src/fusion.md` | FusionStage documentation |

### Input from
- **Role 2** — `FrameData.detections` (List[Detection])
- **Role 3** — `FrameData.obstacles` (List[Dict])

### Output to
- **Role 6** — `FrameData.fused_output` (List[Dict])

### Output contract format
```python
[
    {
        "object_class":  str,                # "person", "chair", "obstacle"
        "distance_m":    float,
        "zone":          "left"|"center"|"right",
        "priority":      int,                # 0 = most dangerous
        "bbox":          [x1, y1, x2, y2],  # xyxy format
        "action":        str | None,         # "STOP" or None
    },
    ...
]
```

### Priority rules
| Pass | Class | Distance | Priority |
|---|---|---|---|
| PASS 1 | person | < danger_distance | 0 (STOP) |
| PASS 1 | other | < danger_distance | 1 |
| PASS 1 | person | < warning_distance | 2 |
| PASS 1 | other | ≥ danger_distance | 3 |
| PASS 2 | obstacle | < 0.5m | 1 |
| PASS 2 | obstacle | < 1.0m | 2 |
| PASS 2 | obstacle | ≥ 1.0m | 3 |

### Implementation
- **PASS 1**: Direct depth sampling from YOLO bbox (center 60%, 25th percentile)
- **PASS 2**: Overlap metric `intersection / min(depth_area, yolo_area)` for covered check
- **Adaptive threshold**: 0.3 in dark/low confidence, 0.5 normal
- **Config**: `DetectionConfig.danger_distance` and `warning_distance` (not hardcoded)

### Done when
- ✅ Every YOLO detection has accurate distance via direct depth sampling
- ✅ Priority is correctly ordered
- ✅ Structured output ready for consumption by Role 6
- ✅ 92 tests covering fusion + navigation + annotation logic

---

## Role 5 — Dataset, Testing & Performance Engineer
**Person:** Hamid

### Responsibilities
- Acquire RealSense recordings outdoors (various scenarios: bright, cloudy, shaded)
- Label YOLO dataset (LabelImg / CVAT / Roboflow)
- Split train/val/test — hand train/val to R2, keep test for evaluation
- Build test harness (measure latency per stage)
- End-to-end benchmark ≤100ms (P95)
- Automated regression testing

### Files
| File | Description |
|---|---|
| `data-collection.md` | Dataset acquisition guide |
| `Doc/model_evaluation_report_v4.md` | V4.2 + V4 model evaluation report |
| Test scripts | Benchmark harness |

### Output to
- **Role 2** — Labeled train/val dataset
- **All roles** — Benchmark & regression test report

### Done when
- Training dataset: ≥3 classes, ≥300 labeled frames (train+val)
- Test dataset: ≥3 scenarios, ≥200 labeled frames
- Test harness measures latency of each stage independently
- Performance report: precision, recall, MAE, latency P50/P95/P99
- Pipeline end-to-end ≤100ms (P95) on target hardware

---

## Role 6 — GUI Maintenance & Operator Console Engineer
**Person:** Adel

### Responsibilities
- Update AlertPanel: object name, distance, zone, danger status, action recommendation
- Integrate RadarView (90° FOV, real data from pipeline, cached background)
- DepthView annotations: bbox + label + distance (visible-only updates, auto-switch RGB/Depth)
- Wire signals from FrameProcessor to GUI
- Maintain stability of all GUI widgets
- Performance optimization (cached pixmaps, change-only stylesheets)

### Files
| File | Description |
|---|---|
| `GUI/src/main_window.py` | Signal wiring + pipeline assembly |
| `GUI/src/depth_view.py` | Camera display (2 modes: RGB/Depth, auto-switch, visible-only updates) |
| `GUI/src/controls_panel.py` | Control panel + threshold sliders + Auto/RGB/Depth view mode |
| `GUI/src/alert_panel.py` | Info panel + alert (cached stylesheets) |
| `GUI/src/radar_view.py` | 90° FOV radar (cached background pixmap) |
| `GUI/inc/ui_config.py` | UI constants |
| `GUI/inc/styles.py` | Stylesheet + color constants |
| `main.py` | Qt bootstrap |

### Input from
- **Role 4** — `FrameData.fused_output`

### Done when
- ✅ AlertPanel format: `PERSON | 2.3 m | CENTER | STOP`
- ✅ RadarView displays real-time obstacle positions (cached background)
- ✅ DepthView annotations: bbox + label + distance (visible-only updates)
- ✅ Detection info shown ≤50ms after frame is processed
- ✅ Operator can make decisions just by looking at GUI
- ✅ All widgets work without bugs

---

## Cross-Role Relationships

| From | To | What |
|---|---|---|
| R1 | R2, R3, R4 | API contracts + code review |
| R1 | R5 | Configuration specifications |
| R1 | R6 | FrameProcessor API specs |
| R2 | R4 | `List[Detection]` per frame |
| R2 | R5 | YOLO model for benchmark |
| R3 | R4 | Zones, distances, mask, colormap |
| R4 | R6 | Structured fusion output |
| R5 | R2 | Train/val dataset |
| R5 | All | Benchmark report |

### Status — Parallel vs Sequential

| Role | Status |
|---|---|
| R1 | ✅ Complete (97% — hardware test pending) |
| R2 | ✅ Models complete (88% — outdoor light stability pending) |
| R3 | ⏳ Outdoor test pending |
| R4 | ✅ Complete (100%) |
| R5 | ✅ Dataset + benchmark complete (86% — regression test pending) |
| R6 | ✅ Complete (83% — 30-min soak + display latency pending) |
