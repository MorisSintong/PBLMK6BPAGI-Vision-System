# Vision Module

This document explains the components in the `Vision/` folder for frame acquisition and vision processing.

## Module Purpose

The Vision module is responsible for:
- Capturing frames from the camera (RealSense / webcam)
- Producing RGB and Depth data (filtered + unfiltered) for GUI and models
- Running the 5-stage vision pipeline: Depth → YOLO → Fusion → Navigation → Annotation
- Detecting obstacles and objects with sensor fusion

## Folder Structure

- `src/` — camera acquisition and processing logic
- `inc/` — vision parameter configuration
- `models/` — model weights (`.gitignore`)

## Main Components (`src`)

| File | Function |
|---|---|
| `camera_thread.py` | Worker thread for camera capture. Separate acquisition thread + queue(maxsize=2). Unfiltered depth capture before RS filters. Sends memory-safe frames to GUI via Qt signals. |
| `frame_processor.py` | Main engine of the vision pipeline (5 stages). Chain of Responsibility with robust error handling. LUT-based depth colormap. Dual-model YOLO swap + CLAHE + hysteresis. Fusion two-pass architecture. NavigationStage (VFH-lite). VisualAnnotationStage draws HUD on both RGB and depth. |
| `yolowrapper.py` | Loads YOLOv8 model and performs inference. FP16 auto-detected, GPU warm-up, input_size=320, batch tensor transfer. Outputs `Detection` dataclass. |
| `obstacle_detector.py` | Extracts distance and priority information from depth frame. Does not copy/modify color frame. Reusable float32 buffer. Thread-safe `last_detections`. |
| `recorder.py` | Standalone RealSense stream test/record utility. Has mutex flag to prevent crashing with the main pipeline. |
| `fusion.md` | FusionStage documentation: two-pass architecture, overlap metric, priority matrix. |

## Configuration (`inc`)

| File | Function |
|---|---|
| `detection_config.py` | Detection threshold parameters (min/max/danger/warning distance). |
| `camera_config.py` | RealSense D455 + webcam fallback configuration. Depth filter parameters (spatial, temporal, decimation). |
| `logging_config.py` | Centralized logging (console + file output). |

## Model Weights (`models`)

| File | Function |
|---|---|
| `ModelRGB_V4.2.pt` | RGB YOLO model (R2 latest, segmentation, 98.37% mAP) |
| `ModelDepth_V4.pt` | Depth YOLO model (R2 latest, trained on unfiltered depth colormap) |
| `security_best.pt` | Fallback model |

Models are in `.gitignore` — not tracked in git. Team must download manually.

## Current Camera Data Flow

1. `CameraThread.start_capture()` called from GUI.
2. **Acquisition thread** captures frames from hardware (RealSense or OpenCV fallback).
3. Unfiltered depth saved before RS filters (for depth model).
4. RS filters applied (spatial, temporal, hole-filling).
5. Frames queued in `queue(maxsize=2)` for processing loop.
6. **Processing loop** pulls frames from queue, runs `FrameProcessor.process()`.
7. 5-stage pipeline: DepthProcessing → YOLODetection → Fusion → Navigation → VisualAnnotation.
8. Results converted to `QImage` (numpy swap + `.tobytes()` for thread safety) and emitted to GUI.

## Development Notes

- 5-stage pipeline: DepthProcessingStage (LUT colormap), YOLODetectionStage (dual-model + CLAHE + hysteresis), FusionStage (two-pass), NavigationStage (VFH-lite gap-based steering), VisualAnnotationStage (HUD on RGB + depth).
- Dual-model: RGB model for bright conditions, depth model for dark conditions (lazy-loaded).
- Dark mode hysteresis: enter dark at brightness < 35, exit at > 50 (prevents flicker).
- Auto-switch view: GUI automatically switches RGB/Depth view based on is_dark signal.
- FP16 inference active when CUDA available (~2x faster on Tensor Cores).
- LUT depth colormap ~3x faster than mask approach.
- ObstacleDetector does not copy color frame (performance optimization).
