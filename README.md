# PBLMK6BPAGI-Vision-System

Modul vision untuk security robot berbasis **PyQt6 + OpenCV + Intel RealSense D455 + YOLOv8**.

## Overview

Aplikasi desktop untuk obstacle avoidance pada security robot:
- **Depth sensing** — Intel RealSense D455 dengan multi-stage filtering (decimation, spatial, temporal, hole-filling)
- **Object detection** — YOLOv8 (GPU-accelerated, ~11ms/frame on RTX A4000)
- **Sensor fusion** — Depth + YOLO untuk situational awareness
- **Real-time GUI** — PyQt6 dengan 4 tampilan: RGB, Depth, Overlay, Radar 180°

## Arsitektur

```
main.py → MainWindow
              ├── DepthView (RGB / Depth / Overlay)
              ├── ControlsPanel (start/stop, thresholds, view mode)
              ├── AlertPanel (object info + zone + action)
              ├── RadarView (180° semicircle)
              └── CameraThread
                    ├── RealSense capture + depth filters
                    └── FrameProcessor (Chain of Responsibility)
                          ├── DepthProcessingStage (colormap + multi-zone)
                          ├── YOLODetectionStage (YOLOv8)
                          └── FusionStage (placeholder)
```

## Fitur Utama

| Fitur | Status | Detail |
|-------|--------|--------|
| RealSense D455 capture | ✅ | 640x480 @ 30fps, depth + RGB streams |
| Depth filters | ✅ | Decimation, spatial, temporal, hole-filling |
| Multi-zone detection | ✅ | LEFT / CENTER / RIGHT zones |
| YOLOv8 integration | ✅ | GPU inference, configurable model |
| Radar view | ✅ | 180° real-time obstacle display |
| Alert panel | ✅ | Object info + action recommendations |
| Threshold controls | ✅ | Adjustable danger/warning distances |

## Struktur Project

```
├── main.py                    # Entry point
├── ROLES.md                   # Role assignments
├── PROGRESS.md                # Progress documentation
├── problems.md                # Critical audit report
├── environment.yml            # Conda/pip dependencies
├── pyproject.toml             # Ruff + pytest config
├── tests/                     # Test suite (39 tests)
├── Vision/
│   ├── src/                   # Core vision modules
│   │   ├── camera_thread.py   # Capture + filter + pipeline
│   │   ├── frame_processor.py # Pipeline orchestrator
│   │   ├── obstacle_detector.py # Depth obstacle detection
│   │   ├── yolowrapper.py     # YOLOv8 inference
│   │   └── recorder.py        # Recording utility
│   ├── models/                # (.gitignore) Model weights
│   └── inc/                   # Config + logging
└── GUI/
    ├── src/                   # PyQt6 components
    └── inc/                   # UI config + styles
```

## Requirements

- Python 3.10
- Conda (disarankan)
- NVIDIA GPU (optional, untuk YOLOv8 GPU inference)
- Intel RealSense D455 (optional, webcam sebagai fallback)

## Setup

```bash
# Clone repository
git clone https://github.com/username/PBLMK6BPAGI-Vision-System.git
cd PBLMK6BPAGI-Vision-System

# Create conda environment
conda env create -f environment.yml
conda activate depth-obstacle-detector

# Download model weights (optional)
# Place yolov8n.pt atau security_best.pt di Vision/models/
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
python -m pytest tests/test_obstacle_detector.py -v
```

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_frame_processor.py` | 8 | Pipeline orchestration, stages, latency |
| `test_obstacle_detector.py` | 16 | Detection, zones, thread safety, buffer |
| `test_camera_thread.py` | 15 | Signals, thresholds, QImage conversion |
| **Total** | **39** | |

## Pipeline Architecture

Pipeline menggunakan pola **Chain of Responsibility**:
- Setiap stage mengimplementasikan `PipelineStage` ABC
- Data mengalir sebagai `FrameData` dataclass
- Stage bisa di-enable/disable secara modular

```python
# Contoh penggunaan
config = DetectionConfig()
processor = FrameProcessor(config)
processor.add_stage(YOLODetectionStage("yolov8n.pt"))

result = processor.process(rgb_frame, depth_frame, depth_scale=0.001)
# result.rgb_frame — frame teranotasi
# result.depth_colormap — visualisasi zona bahaya
# result.obstacles — daftar obstacle
# result.detections — deteksi YOLO
```

## Catatan

- Dukungan D455 menggunakan `pyrealsense2`
- Jika RealSense tidak tersedia, aplikasi memakai webcam biasa (RGB)
- Pada Windows, capture kamera memprioritaskan backend DirectShow
- Model weights tidak di-track di git (lihat `.gitignore`)

## Team

| Role | Responsibility |
|------|----------------|
| R1 (Moris) | ML Pipeline / Integration |
| R2 (Husein) | YOLOv8 Specialist |
| R3 (Long) | Depth / Camera |
| R4 (Rasyid) | Sensor Fusion |
| R5 (Hamid) | Dataset / Testing |
| R6 (Adel) | GUI / Operator Console |
