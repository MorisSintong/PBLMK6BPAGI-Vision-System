# PBLMK6BPAGI-Vision-System

Modul vision untuk security robot berbasis **PyQt6 + OpenCV + Intel RealSense D455 + YOLOv8**.

## Overview

Aplikasi desktop untuk obstacle avoidance pada security robot:
- **Depth sensing** — Intel RealSense D455 dengan multi-stage filtering (decimation, spatial, temporal, hole-filling)
- **Object detection** — YOLOv8 dual-model (RGB + Depth), GPU-accelerated with FP16 on RTX A4000
- **Dark mode adaptation** — CLAHE preprocessing + automatic model swap to depth model in low light
- **Sensor fusion** — Depth + YOLO overlap matching with adaptive thresholds and priority matrix
- **Gap-based navigation** — Polar histogram (VFH-lite) with gap finding, steering output, and speed mapping
- **Auto-switch view** — Automatically switches between RGB and Depth view based on ambient light (hysteresis: <35 enter dark, >50 exit)
- **Real-time GUI** — PyQt6 dengan tampilan: Auto/RGB/Depth (auto-switch), Radar 180°, AlertPanel, ControlsPanel

## Arsitektur

```
main.py → MainWindow
              ├── DepthView (RGB / Depth, auto-switch by light level)
              ├── ControlsPanel (start/stop, thresholds, Auto/RGB/Depth view mode)
              ├── AlertPanel (object info + zone + action)
              ├── RadarView (180° semicircle, cached static background)
              └── CameraThread
                    ├── RealSense acquisition thread (separate thread + queue)
                    └── FrameProcessor (Chain of Responsibility)
                          ├── DepthProcessingStage (LUT colormap + multi-zone)
                          ├── YOLODetectionStage (dual-model swap + CLAHE)
                          ├── FusionStage (overlap matching + priority matrix)
                          ├── NavigationStage (gap-based steering, VFH-lite)
                          └── VisualAnnotationStage (HUD + steering arrow)
```

## Fitur Utama

| Fitur | Status | Detail |
|-------|--------|--------|
| RealSense D455 capture | ✅ | 640x480 @ 30fps, depth + RGB streams, unfiltered depth for model |
| Depth filters | ✅ | Spatial, temporal, hole-filling (decimation configurable) |
| Multi-zone detection | ✅ | LEFT / CENTER / RIGHT zones |
| YOLOv8 dual-model | ✅ | `ModelRGB_V4.2.pt` (normal) + `ModelDepth_V4.pt` (dark mode) |
| FP16 inference | ✅ | Auto-detected on CUDA GPUs, ~2x faster on Tensor Cores |
| GPU warm-up | ✅ | Dummy inference at load to pre-compile CUDA kernels |
| CLAHE dark mode | ✅ | LAB color space enhancement for dim scenes |
| LUT depth colormap | ✅ | Pre-computed 256-entry LUT, ~3x faster than mask approach |
| Sensor fusion | ✅ | Overlap matching + direct depth sampling + priority matrix |
| Visual annotation | ✅ | HUD corner brackets, labels, global status bar |
| Radar view | ✅ | 180° real-time, cached static background pixmap |
| Alert panel | ✅ | Status-change-only stylesheet updates (no redundant recalc) |
| Threshold controls | ✅ | Adjustable danger/warning distances, propagates to all stages |
| Lazy depth model | ✅ | Depth model loaded only on first dark frame (saves VRAM) |
| Separate acquisition thread | ✅ | Camera capture decoupled from processing via queue |

## Struktur Project

```
├── main.py                    # Entry point
├── ROLES.md                   # Role assignments
├── PROGRESS.md                # Progress documentation
├── flow.md                    # Architecture & data flow documentation
├── data-collection.md         # Dataset acquisition guide (R5)
├── environment.yml            # Conda/pip dependencies
├── pyproject.toml             # Ruff + pytest config
├── tests/                     # Test suite (147 tests)
│   ├── test_frame_processor.py    # 92 tests — pipeline, fusion, navigation, dark mode, hysteresis, annotation
│   ├── test_obstacle_detector.py  # 31 tests — detection, zones, filtering, thread safety
│   ├── test_camera_thread.py      # 24 tests — signals, thresholds, QImage, cache
│   └── benchmark.py               # Benchmark suite (17 criteria from ROLES.md)
├── Vision/
│   ├── src/                   # Core vision modules
│   │   ├── camera_thread.py   # Capture + filter + pipeline (separate acq thread, light_mode_changed signal)
│   │   ├── frame_processor.py # Pipeline orchestrator (5 stages)
│   │   ├── obstacle_detector.py # Depth obstacle detection (no frame copy)
│   │   ├── yolowrapper.py     # YOLOv8 inference (FP16, warm-up, batch transfer)
│   │   └── recorder.py        # Recording utility
│   ├── models/                # (.gitignore) Model weights
│   │   ├── ModelRGB_V4.2.pt   # RGB model (R2 latest)
│   │   ├── ModelDepth_V4.pt   # Depth model (R2 latest, trained on unfiltered depth)
│   │   └── security_best.pt   # Fallback model
│   └── inc/                   # Config + logging
│       ├── detection_config.py # Detection thresholds
│       ├── camera_config.py   # Camera parameters (RealSense + webcam)
│       └── logging_config.py  # Centralized logging
├── GUI/
│   ├── src/                   # PyQt6 components
│   │   ├── main_window.py     # Window layout + wiring
│   │   ├── depth_view.py      # Camera display (2 modes: RGB/Depth, visible-only updates)
│   │   ├── controls_panel.py  # Start/stop + thresholds
│   │   ├── alert_panel.py     # Object info + alert (cached stylesheets)
│   │   └── radar_view.py      # 180° radar (cached background pixmap)
│   └── inc/                   # UI config + styles
│       ├── ui_config.py       # UI constants + thresholds
│       └── styles.py          # Global stylesheet + color constants
└── Doc/
    ├── problems_audit_report.md  # Historical audit report (archived)
    └── model_evaluation_report_v4.md  # R5 model evaluation
```

## Requirements

- Python 3.10
- Conda (disarankan)
- NVIDIA GPU (optional, untuk YOLOv8 GPU inference with FP16)
- Intel RealSense D455 (optional, webcam sebagai fallback)

## Setup

```bash
# Clone repository
git clone https://github.com/username/PBLMK6BPAGI-Vision-System.git
cd PBLMK6BPAGI-Vision-System

# Create conda environment
conda env create -f environment.yml
conda activate depth-obstacle-detector

# Download model weights
# Place ModelRGB_V4.2.pt and ModelDepth_V4.pt in Vision/models/
```

## Menjalankan Aplikasi

```bash
python main.py
```

## Menjalankan Test

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_frame_processor.py -v
```

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_frame_processor.py` | 92 | FrameData, PipelineStage, FrameProcessor, DepthProcessingStage (LUT), FusionStage (matching, priority, zones, dark mode, overlap), YOLODetectionStage (dark/bright/CLAHE/dual-model/hysteresis), NavigationStage (clear/blocked/steering/safety override/speed), VisualAnnotationStage (RGB + depth colormap + nav HUD), full pipeline integration |
| `test_obstacle_detector.py` | 31 | Detection, zones, filtering (min_area, max_area_ratio, distance), priority, frame handling (no copy regression), buffer reuse, thread safety, output contract |
| `test_camera_thread.py` | 24 | Instantiation, thresholds (validation + propagation), BGR->QImage (pixel integrity, grayscale, dimensions), empty depth cache, thread lifecycle, signals (frame_pair, distance, obstacles, navigation, light_mode) |
| **Total** | **147** | |

## Pipeline Architecture

Pipeline menggunakan pola **Chain of Responsibility** dengan 5 stage:
- Setiap stage mengimplementasikan `PipelineStage` ABC
- Data mengalir sebagai `FrameData` dataclass
- Stage bisa di-enable/disable secara modular
- Exception di stage manapun ditangkap, error dicatat di `FrameData.errors`

```python
# Contoh penggunaan
config = DetectionConfig()
processor = FrameProcessor(config)
processor.add_stage(YOLODetectionStage(
    model_path="Vision/models/ModelRGB_V4.2.pt",
    depth_model_path="Vision/models/ModelDepth_V4.pt",
))
processor.add_stage(FusionStage(config=config))
processor.add_stage(VisualAnnotationStage(config=config))

result = processor.process(rgb_frame, depth_frame, depth_scale=0.001)
# result.rgb_frame — frame teranotasi (HUD)
# result.depth_colormap — visualisasi zona bahaya (LUT)
# result.obstacles — daftar obstacle dari depth
# result.detections — deteksi YOLO
# result.fused_output — hasil fusion (class + distance + priority)
```

## Catatan

- Dukungan D455 menggunakan `pyrealsense2`
- Jika RealSense tidak tersedia, aplikasi memakai webcam biasa (RGB only)
- Pada Windows, capture kamera memprioritaskan backend DirectShow
- Model weights tidak di-track di git (lihat `.gitignore`)
- Dual-model: RGB model untuk kondisi terang, depth model untuk kondisi gelap (di-load lazy)
- Unfiltered depth frame disimpan sebelum RS filters untuk depth model inference

## Team

| Role | Responsibility |
|------|----------------|
| R1 (Moris) | ML Pipeline / Integration |
| R2 (Husein) | YOLOv8 Specialist |
| R3 (Long) | Depth / Camera |
| R4 (Rasyid) | Sensor Fusion |
| R5 (Hamid) | Dataset / Testing |
| R6 (Adel) | GUI / Operator Console |
