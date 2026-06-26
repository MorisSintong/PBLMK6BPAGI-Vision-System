# FusionStage — Sensor Fusion Documentation

## Overview

`FusionStage` merges semantic data from YOLO (R2) with spatial data from Depth (R3) to produce a single unified obstacle list. It answers the question: **"What is this object, and how far away is it?"**

### Pipeline Position

```
CameraThread
  └─ FrameProcessor
       ├─ Stage A: DepthProcessingStage (R3)  → FrameData.obstacles
       ├─ Stage B: YOLODetectionStage (R2)    → FrameData.detections
       ├─ Stage C: FusionStage (R4)           → FrameData.fused_output  ← THIS
       ├─ Stage D: NavigationStage (R1)       → FrameData.navigation
       └─ Stage E: VisualAnnotationStage (R1) → FrameData.rgb_frame + depth_colormap (HUD)
```

---

## Input Contracts

### From R3 (DepthProcessingStage) — `FrameData.obstacles`

```python
[
    {
        "bbox":        [x, y, w, h],          # OpenCV xywh format, pixels
        "distance_m":  float,                  # meters (5th percentile)
        "zone":        "left"|"center"|"right",
        "area_px":     int,                    # cv2.contourArea — actual blob pixels
        "priority":    float,                  # inverse distance (raw)
    },
    ...
]
```

### From R2 (YOLODetectionStage) — `FrameData.detections`

```python
[
    Detection(
        bbox:        [x1, y1, x2, y2],        # xyxy format, pixels
        class_id:    int,
        class_name:  str,                      # e.g. "person", "chair"
        confidence:  float,                    # 0.0–1.0
    ),
    ...
]
```

### From Metadata — `FrameData.metadata`

```python
{
    "is_dark": bool,          # True if brightness < 35 (hysteresis: exit at > 50)
    "rgb_confidence": float,  # 0.0–1.0, min(brightness/128, 1.0)
}
```

---

## Two-Pass Architecture

FusionStage uses a **two-pass** approach:

### PASS 1 — YOLO-first Direct Depth Sampling (skipped in dark mode)

For each YOLO detection, depth is **sampled directly** from the depth frame within the YOLO bbox. This gives both class name and distance in one step — no need to match YOLO boxes to depth obstacle contours.

```python
for det in data.detections:
    dist = self._sample_depth_in_bbox(depth_frame, depth_scale, det.bbox)
    if dist is None:
        continue  # No valid depth in this bbox — skip
    matched_yolo_indices.add(i)
    # Assign class + distance + priority
```

#### Direct Depth Sampling

```python
def _sample_depth_in_bbox(depth_frame, depth_scale, bbox):
    x1, y1, x2, y2 = bbox
    # Clamp to frame bounds
    # Use center 60% of bbox to avoid background pixels at edges
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)
    region = depth_frame[cy1:cy2, cx1:cx2].astype(np.float32) * depth_scale
    valid = region[(region >= min_dist) & (region <= max_dist)]
    return float(np.percentile(valid, 25))  # 25th percentile
```

**Why 25th percentile?** This gives the distance to the **closest surface** of the object — what matters for collision avoidance. The center 60% region avoids background pixels that bleed into the bbox edges.

**Why skip PASS 1 in dark mode?** In dark mode, YOLO runs on the depth colormap (not RGB), so the detections may not correspond to the same objects as the depth obstacles. It's better to rely on PASS 2 depth-only obstacles.

### PASS 2 — Depth-only Obstacles

For depth obstacles not covered by any YOLO detection (from PASS 1), add them as generic "obstacle" class. This catches objects YOLO missed.

```python
for obs in data.obstacles:
    # Check if already covered by a YOLO detection
    if already_covered_by_yolo(obs, matched_yolo_indices):
        continue
    # Filter: ignore obstacles > 1.5m (too far for generic obstacle)
    if dist > 1.5:
        continue
    # Add as generic obstacle with demoted priority
```

---

## Overlap Metric for PASS 2

In PASS 2, we need to check if a depth obstacle is already covered by a YOLO detection. We use:

```
overlap_ratio = Area(Intersection) / min(Area(Depth), Area(YOLO))
```

This uses the **smallest area** as denominator, so:
- A small depth blob inside a large YOLO box → high overlap (correct)
- A small YOLO box inside a large depth blob → high overlap (correct)

### Why Not IoU?

Standard IoU fails when box sizes differ significantly:

```
YOLO detects a full "person" (200 × 500px = 100,000 px²)
Depth clusters only the chest (80 × 100px = 8,000 px²)

IoU = 8,000 / (100,000 + 8,000 - 8,000) = 0.08  →  REJECTED

overlap_ratio = 8,000 / min(8,000, 100,000) = 8,000 / 8,000 = 1.0  →  MATCHED
```

### Why `min(area)` instead of `depth_area` only?

Using `min(depth_area, yolo_area)` handles both cases:
- Depth blob small, YOLO box large → denominator = depth_area (same as before)
- YOLO box small, depth blob large → denominator = yolo_area (prevents false match)

### The AABB Inflation Problem

`cv2.boundingRect(contour)` returns an **Axis-Aligned Bounding Box (AABB)** that includes empty space for irregular shapes. Using `area_px` (contourArea) as the depth area gives the **true** ratio and avoids rejecting valid matches.

---

## Adaptive Matching Threshold

The overlap threshold adapts to lighting conditions:

| Condition | Threshold | Why |
|---|---|---|
| Normal (is_dark=False, rgb_confidence ≥ 0.5) | 0.5 (50%) | Strict matching when YOLO is reliable |
| Dark or low confidence | 0.3 (30%) | Relaxed matching when YOLO may be inaccurate |

---

## Priority Matrix

### PASS 1 (YOLO detections with depth)

| Class | Distance | Priority | Action |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < `warning_distance` | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

### PASS 2 (depth-only obstacles)

| Distance | Priority | Why |
|---|---|---|
| < 0.5m | 1 | Very close — demoted from 0 to avoid false STOP on generic obstacles |
| < 1.0m | 2 | Close — warning level |
| ≥ 1.0m | 3 | Normal |

Thresholds come from `DetectionConfig` (configurable at runtime via GUI sliders):
- `danger_distance` (default: 1.5m)
- `warning_distance` (default: 3.0m)

---

## Output Format

```python
[
    {
        "object_class":  str,      # "person", "chair", "obstacle"
        "distance_m":    float,    # meters
        "zone":          str,      # "left" | "center" | "right"
        "priority":      int,      # 0 = most dangerous
        "bbox":          [x1, y1, x2, y2],  # xyxy format
        "action":        str | None,         # "STOP" or None
    },
    ...
]
```

---

## Test Coverage

FusionStage is covered by 30+ tests in `tests/test_frame_processor.py`:

- Matching: YOLO + depth, no match, multiple detections
- Priority: person close (0/STOP), obstacle close (1), warning (2), far (3)
- PASS 2 ladder: <0.5m (1), <1.0m (2), ≥1.0m (3)
- Dark mode: PASS 1 skipped, PASS 2 active
- Overlap: identical boxes (1.0), no overlap (0.0), with area_px
- Depth sampling: clamped bbox, all-invalid, tiny bbox
- Zone: left, center, right
- Config: thresholds from DetectionConfig (not hardcoded)
- Contract: all required keys present
