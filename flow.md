# Vision System Architecture & Data Flow

This document provides a comprehensive, step-by-step breakdown of how data moves through the system—from the moment light hits the camera sensor to the moment the GUI renders an alert on the screen. It includes all mathematical formulas, signal processing theory, and design decisions.

The system is built on a decoupled architecture using **PyQt6 Signals** for thread-safe communication and the **Chain of Responsibility** pattern for vision processing.

---

## 1. Initialization Phase

When the program starts (`main.py`), it instantiates the GUI (`MainWindow`). The GUI is entirely passive until the user clicks the **Start Camera** button.

1. **GUI Setup:** `MainWindow` initializes the `DepthView` (for images), `ControlsPanel` (for buttons/sliders), `AlertPanel` (for text status), and `RadarView` (for spatial tracking).
2. **Vision Setup:** The `MainWindow` instantiates the `FrameProcessor` (the brain of the vision system) and the `CameraThread` (a background PyQt `QThread` that talks to the hardware).
3. **Pipeline Assembly:** `FrameProcessor` is configured with five stages in order:
   - `DepthProcessingStage` — always present (R3)
   - `YOLODetectionStage` — dual-model swap with CLAHE (R2)
   - `FusionStage` — merges R2 + R3 output (R4)
   - `NavigationStage` — gap-based steering via polar histogram (R1)
   - `VisualAnnotationStage` — HUD rendering (R1)
4. **Signal Routing:** The GUI connects the thread's output signals (e.g., `frame_pair_ready`, `obstacles_ready`, `navigation_ready`) to its own update functions.
5. **GPU Warm-up:** If CUDA is available, `YOLOWrapper` runs a dummy inference at load time to pre-compile CUDA kernels. This prevents the first real frame from being slow.

---

## 2. Hardware Acquisition (`CameraThread`)

Once the user clicks **Start**, the `CameraThread` enters its loop. Camera capture runs in a **separate acquisition thread**, decoupled from the processing loop via a `queue.Queue(maxsize=2)`.

### 2.1 RealSense Depth Sensing Theory

The Intel RealSense D455 uses **stereo infrared depth sensing**. Two IR sensors capture the same scene from slightly different perspectives. The hardware computes **disparity** — the pixel offset between corresponding points in the two images.

**Depth from Disparity:**

```
depth (meters) = baseline × focal_length / disparity
```

Where:
- `baseline` = distance between the two IR sensors (~95mm for D455)
- `focal_length` = lens focal length in pixels
- `disparity` = pixel offset between left and right IR images

The raw depth frame is a 16-bit unsigned integer array (`z16` format) where each value represents depth in **millimeters**. A value of `0` means "no data" (e.g., too far, reflective surface, or outside IR range).

### 2.2 Unfiltered Depth Capture

The acquisition thread captures the **unfiltered depth frame** before any RealSense SDK filters are applied. This raw depth is preserved because R2's depth model (`ModelDepth_V4.pt`) was trained on unfiltered depth colormaps. The filtered depth is used for display and obstacle detection.

```
depth_raw_unfiltered = np.asanyarray(depth_frame.get_data())  # BEFORE filters
# ... apply filters to depth_frame ...
depth_raw_filtered = np.asanyarray(depth_frame.get_data())    # AFTER filters
```

Both are passed through the queue as a 3-element tuple: `(color_bgr, depth_raw, depth_raw_unfiltered)`.

### 2.3 Hardware Filters

The raw depth frame is passed through RealSense's onboard DSP filters to denoise the 3D data:

1. **Decimation Filter** (optional, configurable via `camera_config.py`):
   - Reduces depth resolution by subsampling (e.g., 2x decimation → 320×240 from 640×480)
   - Trades resolution for performance: 4× fewer pixels to process
   - Uses `INTER_LINEAR` interpolation to avoid `INTER_NEAREST` artifacts
   - Disabled by default to preserve full 640×480 resolution

2. **Spatial Filter** (edge-preserving smoothing):
   - Applies a Gaussian-like filter that smooths flat regions while preserving edges
   - Parameters: `smooth_alpha` (0.5), `smooth_delta` (20)
   - Reduces "salt-and-pepper" noise in depth maps

3. **Temporal Filter** (frame-to-frame smoothing):
   - Averages depth values over multiple frames with exponential decay
   - Reduces temporal jitter (flickering depth values)
   - Uses alpha blending: `depth_filtered = α × depth_current + (1-α) × depth_previous`

4. **Hole-Filling Filter** (interpolation):
   - Fills small gaps (holes) in the depth map by interpolating from neighboring valid pixels
   - Uses a sliding window to find nearest valid depth values

### 2.4 Numpy Conversion

The C++ frames are converted into zero-copy Python NumPy arrays (`color_bgr` and `depth_raw`). This is a pointer-level view of the same memory — no data copy occurs.

### 2.5 Fallback: Webcam Mode

If a RealSense camera isn't plugged in, the system falls back to a standard OpenCV `VideoCapture` (webcam), providing RGB only. In this mode, `depth_frame` is `None` and the depth pipeline is skipped entirely.

---

## 3. The Vision Pipeline (`FrameProcessor`)

The processing loop pulls frames from the queue and hands the NumPy arrays to the `FrameProcessor`. The processor bundles them into a `FrameData` object and passes them through a **Chain of Responsibility** — each stage processes the data and passes it to the next.

### 3.1 Data Structures

**`FrameData`** — the single data object flowing through the pipeline:

```python
@dataclass
class FrameData:
    rgb_frame: np.ndarray              # H×W×3 uint8 BGR
    depth_frame: Optional[np.ndarray]  # H×W uint16 (filtered, None for webcam)
    depth_frame_raw: Optional[np.ndarray]  # H×W uint16 (unfiltered, for depth model)
    depth_colormap: Optional[np.ndarray]   # H×W×3 uint8 (filtered, for display)
    depth_colormap_raw: Optional[np.ndarray]  # H×W×3 uint8 (unfiltered, for depth model)
    depth_scale: float                 # raw → meters conversion factor
    obstacles: List[Dict]              # from DepthProcessingStage
    detections: List[Detection]        # from YOLODetectionStage
    fused_output: List[Dict]           # from FusionStage
    metadata: Dict[str, Any]           # timestamp, is_dark, rgb_confidence, active_model
    errors: List[str]                  # pipeline error log
```

### 3.2 Stage A: `DepthProcessingStage` (Spatial Understanding)

This stage converts raw depth data into structured obstacle information and visualizes it as a colored colormap.

#### 3.2.1 LUT-Based Colormap Generation

Instead of creating multiple boolean masks per frame (slow), the stage uses a **pre-computed 256-entry Lookup Table (LUT)** that maps depth indices to BGR colors:

```python
# Build once at init (and rebuild on threshold change)
lut = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    depth_m = (i / 255.0) * max_distance
    if depth_m < min_distance or depth_m > max_distance:
        lut[i] = (0, 0, 0)        # Black = invalid
    elif depth_m < danger_threshold:
        lut[i] = (0, 0, 255)       # Red = danger
    elif depth_m < warning_threshold:
        lut[i] = (0, 255, 255)     # Yellow = warning
    else:
        lut[i] = (0, 255, 0)       # Green = safe

# Per frame: single indexing operation (~3x faster than mask approach)
depth_m = depth_frame.astype(np.float32) * depth_scale
idx = np.clip(depth_m * scale, 0, 255).astype(np.uint8)
colormap = self._depth_lut[idx]
```

The LUT is rebuilt whenever thresholds change via `set_action_thresholds()` or `set_thresholds()`.

Two colormaps are generated:
- `depth_colormap` — from filtered depth (for display)
- `depth_colormap_raw` — from unfiltered depth (for depth model inference)

#### 3.2.2 Obstacle Detection

The stage uses `ObstacleDetector` to find obstacles in the filtered depth frame. The detector:

1. Converts depth to meters using a reusable float32 buffer (avoids ~1.2MB allocation per frame)
2. Creates a binary mask: pixels within `[min_distance, max_distance]`
3. Applies morphological opening (remove noise) and closing (fill holes)
4. Finds contours and filters by area
5. Computes distance using 5th percentile (nearest surface)
6. Returns obstacles **without copying or modifying the color frame**

#### 3.2.3 Zone Assignment

The frame is divided into 3 equal vertical zones:

```
zone_width = frame_width / 3

zone = {
    "left",    if center_x < zone_width
    "center",  if zone_width ≤ center_x < 2 × zone_width
    "right",   if center_x ≥ 2 × zone_width
}
```

Where `center_x = x + w/2` is the horizontal center of the bounding box.

### 3.3 Stage B: `YOLODetectionStage` (Semantic Understanding)

This stage detects objects in the RGB frame using YOLOv8, with **dual-model swap** and **CLAHE dark mode adaptation**.

#### 3.3.1 Dark Mode Detection

Every frame is analyzed for brightness:

```python
brightness = np.mean(data.rgb_frame)
rgb_confidence = min(brightness / 128.0, 1.0)
# Hysteresis: enter dark at < 35, exit at > 50 (prevents flicker near threshold)
if self._is_dark_state:
    is_dark = brightness < 50
else:
    is_dark = brightness < 35
self._is_dark_state = is_dark
```

- `is_dark` — boolean, true when brightness < 35 (enter) or < 50 (exit, hysteresis)
- `rgb_confidence` — float 0–1, used by FusionStage for adaptive thresholds
- `active_model` — tracks which model was used: `"rgb"`, `"rgb_clahe"`, `"depth"`, `"depth_filtered"`, `"none"`

#### 3.3.2 Dual-Model Swap

The stage selects which model to use based on lighting conditions:

| Condition | Model | Input | active_model |
|---|---|---|---|
| Bright (brightness >= 35) | `ModelRGB_V4.2.pt` | RGB frame | `"rgb"` |
| Dark + depth model available | `ModelDepth_V4.pt` | Unfiltered depth colormap | `"depth"` |
| Dark + depth model + no raw | `ModelDepth_V4.pt` | Filtered depth colormap | `"depth_filtered"` |
| Dark + no depth model | `ModelRGB_V4.2.pt` | CLAHE-enhanced RGB | `"rgb_clahe"` |
| No models available | None | — | `"none"` |

The depth model is **lazy-loaded** — it's only loaded into GPU memory on the first dark frame, saving VRAM at startup.

#### 3.3.3 CLAHE Enhancement

For dim scenes (dark but no depth model), the RGB frame is enhanced using **CLAHE** (Contrast Limited Adaptive Histogram Equalization):

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_enhanced = self._clahe.apply(l)  # clipLimit=3.0, tileGridSize=(8,8)
enhanced = cv2.merge([l_enhanced, a, b])
return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

CLAHE works in LAB color space, enhancing the L (lightness) channel while preserving color.

#### 3.3.4 YOLOv8 Inference

The YOLOWrapper performs inference with the following optimizations:

- **FP16 inference** — `half=True` when CUDA is available, ~2x faster on Tensor Cores
- **Input size 320×320** — reduced from 416 for faster inference
- **GPU warm-up** — dummy inference at load time pre-compiles CUDA kernels
- **Batch tensor transfer** — `boxes.xyxy.cpu().numpy()` once, not per-box

```
input_size = 320 × 320 pixels (configurable)
```

**Why 320×320?** This is a balance between speed and accuracy. For a security robot at close range (0.5–5m), 320×320 provides sufficient detection while being ~40% faster than 416×416.

#### 3.3.5 Detection Output Format

```python
@dataclass
class Detection:
    class_id: int        # COCO class index (e.g., 0 = person)
    class_name: str      # Human-readable name (e.g., "person")
    confidence: float    # 0.0 – 1.0
    bbox: List[int]      # [x1, y1, x2, y2] in pixels (xyxy format)
```

### 3.4 Stage C: `FusionStage` (Data Merging)

This stage answers: **"What is this object, and how far away is it?"** by matching YOLO detections to depth obstacles.

#### 3.4.1 Two-Pass Architecture

FusionStage uses a two-pass approach:

**PASS 1 — YOLO-first (skipped in dark mode):**
For each YOLO detection, directly sample depth from the depth frame within the YOLO bbox. This gives both class name and distance in one step.

```python
for det in data.detections:
    dist = self._sample_depth_in_bbox(depth_frame, depth_scale, det.bbox)
    if dist is None:
        continue  # No valid depth in this bbox
    # Assign class + distance + priority
```

**PASS 2 — Depth-only obstacles:**
For depth obstacles not covered by any YOLO detection, add them as generic "obstacle" class. This catches objects YOLO missed.

```python
for obs in data.obstacles:
    if already_covered_by_yolo(obs):
        continue
    if dist > 1.5:
        continue  # Filter far obstacles
    # Add as generic obstacle with demoted priority
```

#### 3.4.2 Direct Depth Sampling

Instead of matching YOLO boxes to depth obstacle contours, PASS 1 samples depth **directly from the depth frame** within the YOLO bbox:

```python
def _sample_depth_in_bbox(depth_frame, depth_scale, bbox):
    # Use center 60% of bbox to avoid background pixels at edges
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)
    region = depth_frame[cy1:cy2, cx1:cx2].astype(np.float32) * depth_scale
    valid = region[(region >= min_dist) & (region <= max_dist)]
    return float(np.percentile(valid, 25))  # 25th percentile
```

**Why 25th percentile?** This gives the distance to the closest surface of the object — what matters for collision avoidance. The center 60% region avoids background pixels that bleed into the bbox edges.

#### 3.4.3 Overlap Metric for PASS 2

In PASS 2, we need to check if a depth obstacle is already covered by a YOLO detection. We use:

```
overlap_ratio = Area(Intersection) / min(Area(Depth), Area(YOLO))
```

This uses the **smallest area** as denominator, so:
- A small depth blob inside a large YOLO box → high overlap (correct)
- A small YOLO box inside a large depth blob → high overlap (correct)

If `overlap_ratio > threshold`, the obstacle is already covered by YOLO and is skipped.

#### 3.4.4 Adaptive Matching Threshold

The overlap threshold adapts to lighting conditions:

| Condition | Threshold | Why |
|---|---|---|
| Normal (is_dark=False, rgb_confidence ≥ 0.5) | 0.5 (50%) | Strict matching when YOLO is reliable |
| Dark or low confidence | 0.3 (30%) | Relaxed matching when YOLO may be inaccurate |

#### 3.4.5 Priority Matrix

**PASS 1 (YOLO detections with depth):**

| Class | Distance | Priority | Action |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < `warning_distance` | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

**PASS 2 (depth-only obstacles):**

| Distance | Priority | Why |
|---|---|---|
| < 0.5m | 1 | Very close — demoted from 0 to avoid false STOP |
| < 1.0m | 2 | Close — warning level |
| ≥ 1.0m | 3 | Normal |

Thresholds come from `DetectionConfig` (configurable at runtime via GUI sliders).

#### 3.4.6 Output Format

```python
{
    "object_class":  str,      # "person", "chair", "obstacle"
    "distance_m":    float,    # meters
    "zone":          str,      # "left" | "center" | "right"
    "priority":      int,      # 0 = most dangerous
    "bbox":          [x1, y1, x2, y2],  # xyxy format
    "action":        str | None,         # "STOP" or None
}
```

### 3.5 Stage D: `NavigationStage` (Path Planning)

This stage computes a steering recommendation using a **polar histogram + gap-based** approach (VFH-lite). It answers: **"Which direction should the robot steer, and how fast?"**

#### 3.5.1 Polar Histogram

The depth frame is divided into N horizontal sectors (default: 18 sectors of ~5° each). For each sector, the 10th percentile distance is computed (robust to noise):

```
Sector 0 (leftmost)  → min_dist = 0.3m  (blocked)
Sector 1             → min_dist = 0.4m  (blocked)
...
Sector 9 (center)    → min_dist = 4.5m  (free)
...
Sector 17 (rightmost)→ min_dist = 3.8m  (free)
```

#### 3.5.2 Blocked Sector Detection

A sector is marked **blocked** if its minimum distance is less than the robot's required clearance:

```
min_gap = robot_width + 2 × safety_margin
       = 0.5m + 2 × 0.3m = 1.1m

blocked[i] = histogram[i] < min_gap
```

#### 3.5.3 Gap Finding

Contiguous free sectors form **gaps**. Each gap is scored:

```
score = 0.5 × center_bias + 0.3 × width_score + 0.2 × clearance_score
```

- `center_bias` — prefer gaps near center (0°), penalize extreme angles
- `width_score` — wider gaps are safer
- `clearance_score` — deeper gaps allow faster travel

The highest-scoring gap's center angle becomes the recommended steering angle.

#### 3.5.4 Hysteresis (Anti-Oscillation)

To prevent the steering from flipping left/right every frame, the stage sticks with the previous heading for N frames (default: 5) if it's still within a free gap.

#### 3.5.5 Safety Override

If FusionStage found a person at priority 0 (in danger zone), the navigation forces `STOPPED` regardless of available gaps. This ensures the robot never steers around a person it should be stopping for.

#### 3.5.6 Speed Mapping

Speed ramps linearly based on the minimum distance in the center sectors:

```
if min_dist < danger_distance:    speed = 0.0 (stop)
if min_dist >= warning_distance:  speed = 1.0 (full)
else:                             speed = (min_dist - danger) / (warning - danger)
```

#### 3.5.7 Output Format

```python
{
    "steering_angle_deg": float,   # -45 (left) to +45 (right), 0 = straight
    "speed": float,                # 0.0 (stop) to 1.0 (full)
    "status": str,                 # "CLEAR" | "AVOIDING" | "BLOCKED" | "STOPPED"
    "gaps": List[Dict],            # Navigable gaps with angle, width, distance
    "histogram": List[float],      # Min distance per sector
    "blocked_sectors": List[bool], # Blocked flag per sector
}
```

### 3.6 Stage E: `VisualAnnotationStage` (HUD Rendering)

The final stage draws HUD overlays onto both `rgb_frame` and `depth_colormap` **in-place** (so the HUD appears on whichever view is active):

1. **Corner brackets** — 8 lines per object (not full rectangles, less visual clutter)
2. **Dark text plate** with label: `[ZONE] distance_m` or `class_name [ZONE] distance_m`
3. **Color coding**: Soft Red (danger, priority <= 1), Amber (warning, priority <= 2), Lime Green (safe)
4. **Global status bar** (top-left): `SYS: SAFE` / `SYS: WARN` / `SYS: DANGER`
5. **Navigation HUD** (bottom-left): `NAV: AVOIDING | STEER +22 deg | SPD 50%`
6. **Steering arrow** (bottom-center): directional arrow showing recommended heading

Data source priority:
1. `fused_output` (from FusionStage) — bbox in xyxy format
2. `obstacles` (from DepthProcessingStage) — bbox in xywh format
3. `detections` (YOLO-only fallback) — bbox in xyxy format, distance=99.0

---

## 4. Signal Emission & Memory Safety

Once the `FrameProcessor` finishes the pipeline, the data must be safely sent from the background thread to the main GUI thread.

### 4.1 QImage Memory Safety

Qt's `QImage` does **not** own the underlying pixel data. If Python garbage-collects the NumPy array while Qt is still using the QImage, it causes a segmentation fault (crash).

**Solution:** Call `.tobytes()` to create an isolated, safe memory copy:

```python
# numpy channel swap (faster than cv2.cvtColor) + .tobytes() for safety
frame_rgb = frame_bgr[:, :, ::-1].copy()
qimage = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, Format_RGB888)
```

### 4.2 Signal Emission

The thread emits five signals:

1. **`frame_pair_ready(QImage, QImage)`** — RGB and depth images for display
2. **`distance_info_ready(str, object, str)`** — label, distance, zone for alert panel
3. **`obstacles_ready(list)`** — fused or raw obstacles for radar view
4. **`navigation_ready(dict)`** — steering angle, speed, status, gaps for alert panel + radar
5. **`light_mode_changed(bool)`** — is_dark flag for auto-switch view mode

All signals cross the thread boundary via Qt's **signal-slot mechanism**, which is thread-safe by design. The slot functions run in the main thread.

### 4.3 Frame Rate Control

The processing loop does **not** use `msleep` — the queue provides natural flow control. The acquisition thread captures at hardware speed (30 FPS), and the processing loop pulls frames as fast as it can process them. If the queue is full, the oldest frame is dropped (backpressure).

---

## 5. GUI Rendering Phase

The main thread catches the emitted signals and distributes the data to the visual components.

### 5.1 DepthView (Camera Display)

Converts the safe `QImage` into a hardware-accelerated `QPixmap` and renders it. Optimizations:
- `setScaledContents(True)` called once at init (not per frame)
- Only updates labels for the currently visible page (RGB / Depth)
- Handles empty depth maps (webcam mode) via `.isNull()` checks

### 5.2 AlertPanel (Status Display)

Reads the distance and zone from `distance_info_ready` and updates:
- Object name (from YOLO class or "OBSTACLE")
- Distance in meters
- Zone (LEFT / CENTER / RIGHT)
- Action recommendation (STOP / SLOWDOWN / GO)
- Color-coded status (DANGER / WARNING / SAFE)

**Optimization:** Stylesheets are only applied when the status **changes** (e.g., SAFE → DANGER). In steady state, zero stylesheet recalculations per frame.

### 5.3 RadarView (90° FOV Spatial Display)

Renders a top-down 90° FOV wedge radar showing obstacle positions.

**Optimization:** The static background (rings, labels, FOV lines, zone lines) is **pre-rendered once** into a cached `QPixmap`. Only the sweep line and obstacle blips are redrawn each frame. This reduces paint work by ~80%.

#### 5.3.1 Polar Coordinate Mapping

Each obstacle's bbox center is mapped to an angle on the radar:

```
angle_deg = 135 - (bbox_center_x / frame_width) × 90
```

- Left edge of frame (0px) → 135° (left of radar)
- Center of frame (320px) → 90° (center of radar)
- Right edge of frame (640px) → 45° (right of radar)

#### 5.3.2 Cartesian Conversion

```
dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
bx = cx + dist_frac × r × cos(angle_deg)
by = cy - dist_frac × r × sin(angle_deg)
```

---

## 6. Performance Summary

| Component | Latency | Notes |
|---|---|---|
| RealSense capture | ~33ms | Hardware-limited at 30 FPS |
| DepthProcessingStage (LUT) | ~1–3ms | LUT indexing + obstacle detection |
| YOLODetectionStage | ~5–10ms | FP16 GPU inference (RTX A4000, 320px) |
| FusionStage | <1ms | Overlap calculation + depth sampling |
| VisualAnnotationStage | ~1ms | OpenCV drawing |
| QImage conversion | ~0.5ms | numpy swap + tobytes() |
| **Total per frame** | **~10–20ms** | Target: 30 FPS (33ms budget) |

---

## Summary Loop

```
Camera grabs light (30 FPS)
  → Acquisition thread captures + filters (spatial, temporal, hole-filling)
    → Unfiltered depth preserved (for depth model)
  → Queue delivers frames to processing loop
    → Depth converts to LUT colormap + obstacle detection
    → YOLO identifies objects (dual-model swap: RGB/depth/CLAHE)
    → Fusion matches them (PASS 1: direct sampling, PASS 2: overlap)
    → Navigation computes steering (polar histogram + gap selection)
    → Visual annotation draws HUD + steering arrow (in-place)
  → Signals transmit data (thread-safe QImage + typed signals)
    → DepthView renders images (visible-only updates)
    → AlertPanel shows status (change-only stylesheets)
    → RadarView plots positions (cached background)
```

*(This entire cycle happens in under ~20 milliseconds, 30+ times a second.)*
