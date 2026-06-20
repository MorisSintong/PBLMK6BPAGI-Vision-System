# Vision System Architecture & Data Flow

This document provides a detailed, step-by-step breakdown of how data moves through the system—from the moment light hits the camera sensor to the moment the GUI renders an alert on the screen.

The system is built on a decoupled architecture using **PyQt6 Signals** for thread-safe communication and the **Chain of Responsibility** pattern for vision processing.

---

## 1. Initialization Phase
When the program starts (`main.py`), it instantiates the GUI (`MainWindow`). The GUI is entirely passive until the user clicks the **Start Camera** button.

1. **GUI Setup:** `MainWindow` initializes the `DepthView` (for images), `ControlsPanel` (for buttons/sliders), `AlertPanel` (for text status), and `RadarView` (for spatial tracking).
2. **Vision Setup:** The `MainWindow` instantiates the `FrameProcessor` (the brain of the vision system) and the `CameraThread` (a background PyQt `QThread` that talks to the hardware).
3. **Signal Routing:** The GUI connects the thread's output signals (e.g., `frame_pair_ready`, `obstacles_ready`) to its own update functions.

---

## 2. Hardware Acquisition (`CameraThread`)
Once the user clicks **Start**, the `CameraThread` enters its loop.

1. **Hardware Poll:** It calls `pipeline.wait_for_frames()`. This physically blocks the thread until the Intel RealSense hardware delivers a synchronized pair of RGB and Depth frames (running precisely at 30 FPS).
   * *Fallback:* If a RealSense camera isn't plugged in, it falls back to a standard OpenCV `VideoCapture` (webcam), providing RGB only.
2. **Hardware Filters:** The raw depth frame is passed through RealSense's onboard DSP filters (Spatial, Temporal, and Hole-Filling) to denoise the 3D data. (Hardware Decimation is configurable via `camera_config.py`).
3. **Numpy Conversion:** The C++ frames are converted into zero-copy Python NumPy arrays (`color_bgr` and `depth_raw`).

---

## 3. The Vision Pipeline (`FrameProcessor`)
The `CameraThread` hands the NumPy arrays to the `FrameProcessor`. The processor bundles them into a `FrameData` object and passes them through a **Chain of Responsibility**.

### Stage A: `YOLODetectionStage` (Semantic Understanding)
* The RGB frame is sent to the `YOLOWrapper`.
* The YOLOv8 neural network (accelerated via FP32/FP16 on the GPU) detects objects like people, chairs, or vehicles.
* It outputs a list of `Detection` dataclasses containing bounding boxes `[x1, y1, x2, y2]` and the object class.

### Stage B: `DepthProcessingStage` (Spatial Understanding)
* This stage passes the Depth frame to the `ObstacleDetector`.
* It converts the raw 16-bit depth values into meters.
* It slices the depth map based on the active obstacle threshold (e.g., 0.3m to 5.0m).
* Using OpenCV contours, it finds physical masses and determines their distance by taking the **5th percentile** of the depth pixels within the object's boundary.
* It assigns a zone (`Left`, `Center`, `Right`) based on the object's horizontal center point.
* It draws the premium HUD visuals (corner brackets, dark text plates, zone ticks) onto the RGB frame.

### Stage C: `FusionStage` (Data Merging)
* *Current capability:* Merges the raw depth arrays with YOLO's bounding boxes to coordinate logic if necessary.
* *Future capability:* Merging the semantic data from YOLO with the spatial data from the Depth sensor to confidently say "Person at 1.5m".

---

## 4. Signal Emission & Memory Safety
Once the `FrameProcessor` finishes the pipeline, the data must be safely sent from the background thread back to the main GUI thread.

1. **QImage Conversion:** The NumPy arrays are converted to Qt `QImage` objects. To prevent memory segmentation faults (where Python garbage-collects the array while the GUI is trying to draw it), the thread calls `.tobytes()`. This creates an isolated, safe memory copy.
2. **Emission:** The thread emits three critical signals:
   * `frame_pair_ready(QImage, QImage)`
   * `distance_info_ready(str, float)`
   * `obstacles_ready(list)`
3. **Delta Sleep Optimizer:** Before restarting the loop, the thread calculates exactly how many milliseconds the processing took, and sleeps *only* the remaining time required to maintain 30 FPS. This prevents CPU hogging and double-blocking.

---

## 5. GUI Rendering Phase
The main thread catches the emitted signals and distributes the data to the visual components.

1. **DepthView:** Converts the safe `QImage` into a hardware-accelerated `QPixmap` and renders it to the screen. It safely handles empty depth maps (if in webcam fallback mode) via `.isNull()` checks.
2. **AlertPanel:** Reads the global status (DANGER, WARN, SAFE) and updates its stylesheet colors and text to alert the operator.
3. **RadarView:** Iterates over the `obstacles_ready` list. It calculates the physical angle of the object based on its `x` coordinate and its distance, rendering a top-down blip on the radar sweep.

---

## Summary Loop
1. **Camera** grabs light ➜ **Thread** filters noise ➜ **YOLO** identifies objects ➜ **Depth** calculates distance ➜ **Signals** transmit data ➜ **GUI** draws UI.
*(This entire cycle happens in under ~33 milliseconds, 30 times a second).*
