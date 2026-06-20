# Vision System Architecture & Data Flow

This document provides a comprehensive, step-by-step breakdown of how data moves through the system—from the moment light hits the camera sensor to the moment the GUI renders an alert on the screen. It includes all mathematical formulas, signal processing theory, and design decisions.

The system is built on a decoupled architecture using **PyQt6 Signals** for thread-safe communication and the **Chain of Responsibility** pattern for vision processing.

---

## 1. Initialization Phase

When the program starts (`main.py`), it instantiates the GUI (`MainWindow`). The GUI is entirely passive until the user clicks the **Start Camera** button.

1. **GUI Setup:** `MainWindow` initializes the `DepthView` (for images), `ControlsPanel` (for buttons/sliders), `AlertPanel` (for text status), and `RadarView` (for spatial tracking).
2. **Vision Setup:** The `MainWindow` instantiates the `FrameProcessor` (the brain of the vision system) and the `CameraThread` (a background PyQt `QThread` that talks to the hardware).
3. **Pipeline Assembly:** `FrameProcessor` is configured with three stages in order:
   - `DepthProcessingStage` — always present (R3)
   - `YOLODetectionStage` — if model file exists (R2)
   - `FusionStage` — merges R2 + R3 output (R4)
4. **Signal Routing:** The GUI connects the thread's output signals (e.g., `frame_pair_ready`, `obstacles_ready`) to its own update functions.

---

## 2. Hardware Acquisition (`CameraThread`)

Once the user clicks **Start**, the `CameraThread` enters its loop.

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

### 2.2 Hardware Filters

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

### 2.3 Numpy Conversion

The C++ frames are converted into zero-copy Python NumPy arrays (`color_bgr` and `depth_raw`). This is a pointer-level view of the same memory — no data copy occurs.

### 2.4 Fallback: Webcam Mode

If a RealSense camera isn't plugged in, the system falls back to a standard OpenCV `VideoCapture` (webcam), providing RGB only. In this mode, `depth_frame` is `None` and the depth pipeline is skipped entirely.

---

## 3. The Vision Pipeline (`FrameProcessor`)

The `CameraThread` hands the NumPy arrays to the `FrameProcessor`. The processor bundles them into a `FrameData` object and passes them through a **Chain of Responsibility** — each stage processes the data and passes it to the next.

### 3.1 Data Structures

**`FrameData`** — the single data object flowing through the pipeline:

```python
@dataclass
class FrameData:
    rgb_frame: np.ndarray          # H×W×3 uint8 BGR
    depth_frame: Optional[np.ndarray]  # H×W uint16 (None for webcam)
    depth_colormap: Optional[np.ndarray]  # H×W×3 uint8 (visualization)
    depth_scale: float             # raw → meters conversion factor
    obstacles: List[Dict]          # from DepthProcessingStage
    detections: List[Detection]    # from YOLODetectionStage
    fused_output: List[Dict]       # from FusionStage
    metadata: Dict[str, Any]       # timestamp, FPS, etc.
    errors: List[str]              # pipeline error log
```

### 3.2 Stage A: `DepthProcessingStage` (Spatial Understanding)

This stage converts raw depth data into structured obstacle information.

#### 3.2.1 Unit Conversion

The raw depth frame is in millimeters (`uint16`). To convert to meters:

```
depth_meters = depth_raw × depth_scale
```

Where `depth_scale = 0.001` (1mm = 0.001m).

**Performance optimization:** Instead of allocating a new float32 array every frame (~1.2MB for 640×480), we reuse a pre-allocated buffer:

```python
# Allocate once, reuse every frame
if self._depth_buffer is None or self._depth_buffer.shape != depth_frame.shape:
    self._depth_buffer = np.empty_like(depth_frame, dtype=np.float32)
np.multiply(depth_frame, depth_scale, out=self._depth_buffer, casting="unsafe")
```

This avoids GC pauses at 30 FPS.

#### 3.2.2 Range Thresholding

Only pixels within a valid distance range are considered obstacles:

```
obstacle_mask(x, y) = {
    255,  if min_distance_m ≤ depth_meters(x, y) ≤ max_distance_m
    0,    otherwise
}
```

Default thresholds:
- `min_distance_m = 0.3m` — objects closer than this are ignored (camera noise, very close surfaces)
- `max_distance_m = 5.0m` — objects farther than this are ignored (beyond sensor reliable range)

#### 3.2.3 Morphological Filtering

The binary obstacle mask is noisy. We apply two morphological operations to clean it:

**Opening** (erosion followed by dilation):
```
opened = dilate(erode(mask, kernel), kernel)
```
- Removes small isolated noise pixels (salt noise)
- Kernel: 5×5 rectangular structuring element

**Closing** (dilation followed by erosion):
```
closed = erode(dilate(opened, kernel), kernel)
```
- Fills small holes within obstacle regions (pepper noise)
- Connects nearby obstacle fragments

The structuring element is created efficiently using OpenCV's native C++ implementation:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5×5 rectangle
```

#### 3.2.4 Contour Finding

After morphological cleaning, we find connected components (contours) in the binary mask:

```python
contours, _ = cv2.findContours(
    obstacle_mask,
    cv2.RETR_EXTERNAL,      # Only outermost contours (no nesting)
    cv2.CHAIN_APPROX_SIMPLE  # Compress contour points (memory efficient)
)
```

**`RETR_EXTERNAL`** returns only the outermost contour of each blob, ignoring nested contours. This is appropriate because we want each physical object as one blob, not its internal structure.

**`CHAIN_APPROX_SIMPLE`** compresses horizontal/vertical/diagonal runs into single endpoint pairs, reducing memory usage.

#### 3.2.5 Area Filtering

Contours with area below `min_area` (default: 800 pixels) are discarded. This filters out:
- Small noise remnants from depth sensor
- Tiny objects that are too small to be relevant for robot navigation

```
if contour_area < min_area:
    discard contour
```

#### 3.2.6 Distance Calculation via Percentile

For each valid contour, we extract the depth pixels within its bounding box and compute distance:

```python
object_depth = depth_meter[y:y+h, x:x+w]  # Crop depth within bbox
valid_depth = object_depth[
    (object_depth >= min_distance_m) & (object_depth <= max_distance_m)
]
distance = np.percentile(valid_depth, 5)  # 5th percentile
```

**Why 5th percentile instead of mean/median?**

- **Mean** is sensitive to outliers: a few far-away pixels (from background bleeding into the bbox) would inflate the distance
- **Median** (50th percentile) is more robust but still includes background pixels
- **5th percentile** gives the distance to the **closest surface** of the object — this is what matters for collision avoidance. If a person is standing 1.5m away, we want 1.5m, not the average of the person (1.5m) and the wall behind them (4m)

The 5th percentile is a standard technique in depth sensing for "nearest point" estimation.

#### 3.2.7 Priority Calculation

The obstacle detector computes a raw priority score:

```
priority_raw = 1 / max(distance, 0.01)
```

This is an **inverse distance** metric: closer objects get higher scores. The `max(..., 0.01)` prevents division by zero. This raw score is used for sorting; the FusionStage later overrides it with a discrete priority matrix.

#### 3.2.8 Zone Assignment

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

#### 3.2.9 HUD Visual Rendering

Each obstacle gets annotated with:
- **Corner brackets** (not full rectangles) — less visual clutter, professional HUD look
- **Dark text plate** with distance label (e.g., `[C] 1.50m`)
- **Color coding**: Soft Red (danger), Amber (warning), Lime Green (safe)
- **Zone ticks** — small lines at zone boundaries instead of full vertical lines

### 3.3 Stage B: `YOLODetectionStage` (Semantic Understanding)

The RGB frame is sent to the YOLOWrapper for object detection.

#### 3.3.1 YOLOv8 Architecture Theory

YOLOv8 (You Only Look Once, version 8) is a single-shot object detector. Unlike two-stage detectors (R-CNN), it detects objects in one forward pass through the network:

1. **Backbone** (feature extraction): CSPDarknet extracts multi-scale features from the input image
2. **Neck** (feature aggregation): PANet/FPN combines features from different scales
3. **Head** (detection): Predicts bounding boxes, class probabilities, and objectness scores

**Key innovation of YOLO:** The image is divided into an S×S grid. Each grid cell predicts B bounding boxes with their confidence scores and class probabilities. This allows detecting multiple objects simultaneously in a single pass.

#### 3.3.2 Input Preprocessing

The input frame is resized to a fixed size for the neural network:

```
input_size = 416 × 416 pixels (configurable)
```

The frame is normalized to [0, 1] range by dividing pixel values by 255. This is done internally by the ultralytics library.

**Why 416×416?** This is a balance between:
- Speed: Smaller input → faster inference
- Accuracy: Larger input → better detection of small objects
- For a security robot at close range (0.5–5m), 416×416 is sufficient

#### 3.3.3 Inference and Post-Processing

The model outputs raw predictions which are post-processed:

1. **Confidence thresholding:** Only detections with `confidence ≥ conf_threshold` (default: 0.25) are kept
2. **Non-Maximum Suppression (NMS):** When multiple boxes overlap the same object, NMS keeps only the best one:

```
For each pair of overlapping boxes (i, j):
    IoU(i, j) = area(intersection) / area(union)

If IoU(i, j) > iou_threshold:
    Discard the box with lower confidence
```

3. **Coordinate extraction:** The surviving boxes are returned as `[x1, y1, x2, y2]` (xyxy format) in the original image coordinate space.

#### 3.3.4 Detection Output Format

```python
@dataclass
class Detection:
    class_id: int        # COCO class index (e.g., 0 = person)
    class_name: str      # Human-readable name (e.g., "person")
    confidence: float    # 0.0 – 1.0
    bbox: List[int]      # [x1, y1, x2, y2] in pixels
```

### 3.4 Stage C: `FusionStage` (Data Merging)

This stage answers: **"What is this object, and how far away is it?"** by matching YOLO detections to depth obstacles.

#### 3.4.1 The Matching Problem

YOLO and Depth are **independent sensors** with different algorithms:

| | YOLO | Depth |
|---|---|---|
| Input | RGB image | 16-bit depth map |
| Algorithm | Neural network | Contour finding |
| Output bbox | Tight around visible object | Bounding rect around depth contour |
| Knows class? | Yes ("person", "chair") | No ("obstacle") |
| Knows distance? | No | Yes (meters) |

They produce **different bounding boxes for the same physical object**. FusionStage must match them.

#### 3.4.2 Why Not IoU (Intersection over Union)?

Standard IoU is the most common metric for bounding box overlap:

```
IoU = Area(Intersection) / Area(Union)
    = Area(Intersection) / (Area(Box_A) + Area(Box_B) - Area(Intersection))
```

**Problem:** IoU fails when box sizes differ significantly.

Consider this scenario:
- YOLO detects a full "person" (head to toe): 200×500px = 100,000 px²
- Depth clusters only the closest points (chest): 80×100px = 8,000 px²
- The chest is **completely inside** the person box

```
IoU = 8,000 / (100,000 + 8,000 - 8,000)
    = 8,000 / 100,000
    = 0.08  →  REJECTED (below 0.3 threshold)
```

The depth blob is completely inside the YOLO box, but IoU penalizes the size mismatch and rejects a valid match.

#### 3.4.3 Why `intersection / depth_area` Is Correct

We use a **one-directional** overlap metric:

```
overlap_ratio = Area(Intersection) / Area(Depth_Box)
```

This asks: **"What portion of the depth blob is explained by the YOLO detection?"**

Using the same scenario:
```
overlap_ratio = 8,000 / 8,000 = 1.0  →  MATCHED (100% covered)
```

If the YOLO box fully covers the depth blob, the ratio is 1.0 — a perfect match. This is the correct behavior: we're asking "does the YOLO class label apply to this depth blob?"

#### 3.4.4 The AABB Inflation Problem

`cv2.boundingRect(contour)` returns an **Axis-Aligned Bounding Box** around the contour. For irregular shapes, the AABB includes empty space:

```
Actual depth blob (contour pixels):       cv2.boundingRect (AABB):

. . . X X X X                             X X X X X X X
. . . X X X X                             X X X X X X X
. . . X X X X                             X X X X X X X
X X X X X X X                             X X X X X X X
X X X X X X X                             X X X X X X X
X X X X . . .                             X X X X X X X
X X X X . . .                             X X X X X X X

contourArea = 36 px                       boundingRect = 7×7 = 49 px
                                           (includes empty corners)
```

Using AABB area as denominator **deflates** the overlap ratio:

| Shape | Contour Area | AABB Area | Ratio with AABB | Ratio with contourArea |
|---|---|---|---|---|
| L-shape | 36 | 49 | 0.73 | 1.00 |
| Thin arc | 15 | 49 | 0.31 | 0.73 |
| Scattered | 20 | 64 | 0.31 | 0.80 |

Using `area_px` (the actual `cv2.contourArea`) as denominator gives the **true** ratio and avoids rejecting valid matches.

#### 3.4.5 Overlap Calculation Algorithm

```python
def _calculate_overlap_ratio(depth_box, yolo_box, depth_area_px=None):
    # Convert depth [x, y, w, h] to [x1, y1, x2, y2]
    depth_box_xyxy = [x, y, x+w, y+h]

    # Find intersection rectangle
    xA = max(depth_box[0], yolo_box[0])
    yA = max(depth_box[1], yolo_box[1])
    xB = min(depth_box[2], yolo_box[2])
    yB = min(depth_box[3], yolo_box[3])

    # Calculate intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Use contour area if available, else AABB area
    depthBoxArea = depth_area_px or (w * h)

    return interArea / depthBoxArea
```

#### 3.4.6 Matching Threshold

A depth blob is matched to a YOLO class only if:

```
overlap_ratio > 0.5  (50%)
```

This means at least half of the depth blob's actual area must be covered by the YOLO detection. If no YOLO detection meets this threshold, the object defaults to `"obstacle"` (generic class).

#### 3.4.7 Priority Matrix

After matching, the FusionStage assigns a discrete priority based on class and distance:

| Class | Distance | Priority | Action |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < 3.0m | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

Where `danger_distance` comes from `DetectionConfig` (default: 1.5m, configurable via GUI slider).

**Priority 0 (STOP)** is the highest priority — a person within danger distance requires immediate stop. This is the safety-critical case for a security robot.

#### 3.4.8 Output Format

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

---

## 4. Signal Emission & Memory Safety

Once the `FrameProcessor` finishes the pipeline, the data must be safely sent from the background thread to the main GUI thread.

### 4.1 QImage Memory Safety

Qt's `QImage` does **not** own the underlying pixel data. If Python garbage-collects the NumPy array while Qt is still using the QImage, it causes a segmentation fault (crash).

**Solution:** Call `.tobytes()` to create an isolated, safe memory copy:

```python
# UNSAFE — frame_rgb may be GC'd while QImage is in use
qimage = QImage(frame_rgb.data, w, h, bytes_per_line, Format_RGB888)

# SAFE — tobytes() creates a new memory block owned by QImage
qimage = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, Format_RGB888)
```

The tradeoff is a small memory copy (~900KB for 640×480 RGB), but this is negligible compared to the crash risk.

### 4.2 Signal Emission

The thread emits three signals:

1. **`frame_pair_ready(QImage, QImage)`** — RGB and depth images for display
2. **`distance_info_ready(str, float, str)`** — label, distance, zone for alert panel
3. **`obstacles_ready(list)`** — fused or raw obstacles for radar view

All signals cross the thread boundary via Qt's **signal-slot mechanism**, which is thread-safe by design. The slot functions run in the main thread.

### 4.3 Delta Sleep Optimizer

The camera loop targets 30 FPS (33.3ms per frame). Instead of sleeping a fixed amount, we calculate the exact remaining time:

```python
elapsed_ms = int((time.time() - start_time) * 1000)
sleep_ms = max(1, target_frame_ms - elapsed_ms)
self.msleep(sleep_ms)
```

This prevents:
- **Double-blocking:** If processing takes 20ms and we sleep 33ms, total is 53ms (18 FPS instead of 30)
- **CPU hogging:** If processing takes 5ms and we don't sleep, the loop runs at max speed wasting CPU

---

## 5. GUI Rendering Phase

The main thread catches the emitted signals and distributes the data to the visual components.

### 5.1 DepthView (Camera Display)

Converts the safe `QImage` into a hardware-accelerated `QPixmap` and renders it. Handles empty depth maps (webcam mode) via `.isNull()` checks.

### 5.2 AlertPanel (Status Display)

Reads the distance and zone from `distance_info_ready` and updates:
- Object name (from YOLO class or "OBSTACLE")
- Distance in meters
- Zone (LEFT / CENTER / RIGHT)
- Action recommendation (STOP / SLOWDOWN / GO)
- Color-coded status (DANGER / WARNING / SAFE)

**Action logic:**
```
if distance ≤ ACTION_STOP_DISTANCE:
    action = "STOP"
elif distance ≤ ACTION_SLOWDOWN_DISTANCE:
    if zone == "CENTER": action = "SLOWDOWN"
    elif zone == "LEFT":  action = "TURN RIGHT"
    elif zone == "RIGHT": action = "TURN LEFT"
else:
    action = "GO"
```

### 5.3 RadarView (180° Spatial Display)

Renders a top-down semicircular radar showing obstacle positions.

#### 5.3.1 Polar Coordinate Mapping

Each obstacle has a zone (left/center/right) and distance. We map these to polar coordinates on the radar:

```
angle_deg = ZONE_TO_ANGLE[zone]  # left=45°, center=90°, right=135°
dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)  # Normalize to [0, 1]
```

#### 5.3.2 Cartesian Conversion

The radar is a semicircle (0°–180°). We convert polar to Cartesian for painting:

```
bx = cx + dist_frac × r × cos(180° - angle_deg)
by = cy - dist_frac × r × sin(180° - angle_deg)
```

Where:
- `(cx, cy)` = center of the radar (bottom-center of the widget)
- `r` = radius of the radar circle
- `angle_deg` = angle in degrees (0° = right, 90° = center, 180° = left)

The `180° - angle_deg` flip is because the radar's coordinate system has 0° on the right and 180° on the left, matching the robot's forward-facing perspective.

#### 5.3.3 Visual Elements

- **Distance rings:** Concentric semicircles at 25%, 50%, 75%, 100% of max depth
- **Zone lines:** Dashed lines at 60° and 120° dividing the radar into 3 sectors
- **Sweep line:** Animated pendulum line (0° → 180° → 0°) for visual effect
- **Blips:** Colored circles at obstacle positions (center=red, sides=amber, safe=green)

---

## 6. Performance Summary

| Component | Latency | Notes |
|---|---|---|
| RealSense capture | ~33ms | Hardware-limited at 30 FPS |
| DepthProcessingStage | ~2–5ms | Morphology + contour finding |
| YOLODetectionStage | ~7–15ms | GPU inference (RTX A4000) |
| FusionStage | <1ms | Simple overlap calculation |
| QImage conversion | ~1ms | Memory copy via `.tobytes()` |
| **Total per frame** | **~20–30ms** | Leaves headroom for 30 FPS |

---

## Summary Loop

```
Camera grabs light (30 FPS)
  → Thread filters noise (spatial, temporal, hole-filling)
    → Depth converts to meters (unit conversion + morphology + contours)
    → YOLO identifies objects (neural network + NMS)
    → Fusion matches them (overlap ratio + priority matrix)
  → Signals transmit data (thread-safe QImage + typed signals)
    → DepthView renders images
    → AlertPanel shows status
    → RadarView plots positions
```

*(This entire cycle happens in under ~33 milliseconds, 30 times a second.)*
