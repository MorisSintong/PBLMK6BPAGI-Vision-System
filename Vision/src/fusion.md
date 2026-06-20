# FusionStage — Sensor Fusion Documentation

## Overview

`FusionStage` merges semantic data from YOLO (R2) with spatial data from Depth (R3) to produce a single unified obstacle list. It answers the question: **"What is this object, and how far away is it?"**

### Pipeline Position

```
CameraThread
  └─ FrameProcessor
       ├─ Stage A: DepthProcessingStage (R3)  → FrameData.obstacles
       ├─ Stage B: YOLODetectionStage (R2)    → FrameData.detections
       └─ Stage C: FusionStage (R4)           → FrameData.fused_output  ← THIS
```

---

## Input Contracts

### From R3 (DepthProcessingStage) — `FrameData.obstacles`

```python
[
    {
        "bbox":        [x, y, w, h],          # OpenCV format, pixels
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

---

## The Matching Problem

YOLO and Depth are **independent sensors** with different algorithms:

| | YOLO | Depth |
|---|---|---|
| Input | RGB image | 16-bit depth map |
| Algorithm | Neural network (blob detection) | Contour finding (threshold + morphology) |
| Output bbox | Tight around visible object | Bounding rect around depth contour |
| Knows class? | Yes ("person", "chair") | No ("obstacle") |
| Knows distance? | No | Yes (meters) |

They produce **different bounding boxes for the same physical object**. FusionStage must match them.

---

## Overlap Metric: Why Not IoU?

Standard IoU (Intersection over Union) fails in our scenario:

```
YOLO detects a full "person" (head to toe):     Depth clusters only the closest points
                                                (e.g. the person's chest):

┌──────────────────────┐                         ┌───┐
│                      │                         │░░░│
│     YOLO box         │                         │░░░│  Depth box
│     (200 × 500px)    │                         │░░░│  (80 × 100px)
│                      │                         └───┘
│         ┌───┐        │
│         │░░░│        │  ← chest is INSIDE the person box
│         │░░░│        │
│         └───┘        │
│                      │
└──────────────────────┘

IoU = intersection / union
    = 8000 / (100000 + 8000 - 8000)
    = 8000 / 100000
    = 0.08  →  REJECTED (below 0.3 threshold)
```

The depth blob is **completely inside** the YOLO box, but IoU penalizes the size mismatch and rejects a valid match.

## Why `intersection / depth_area` Is Correct

```
ratio = intersection / depth_area
      = 8000 / 8000
      = 1.0  →  MATCHED (100% of depth blob covered by YOLO)
```

This asks: **"What portion of the depth blob is explained by the YOLO detection?"**

If the answer is high, the YOLO class is a good label for that depth blob.

---

## The Bounding Box Inflation Problem

`cv2.boundingRect(contour)` returns an **Axis-Aligned Bounding Box (AABB)** around the contour. For irregular shapes, the AABB includes empty space:

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

### Impact on Overlap Ratio

Using AABB area as denominator **deflates** the ratio:

| Shape | Contour Area | AABB Area | Ratio with AABB | Ratio with contourArea |
|---|---|---|---|---|
| L-shape | 36 | 49 | 0.73 | 1.00 |
| Thin arc | 15 | 49 | 0.31 | 0.73 |
| Scattered | 20 | 64 | 0.31 | 0.80 |

Using `area_px` (contourArea) gives the **true** ratio and avoids rejecting valid matches.

### The Fix

Use `area_px` from the obstacle dict as denominator:

```python
depthBoxArea = obs.get("area_px", w * h)  # fallback to AABB if missing
```

---

## Priority Matrix

After matching a YOLO class to a depth blob, assign priority:

| Class | Distance | Priority | Action |
|---|---|---|---|
| person | < 1.0m | 0 | STOP |
| person | < 3.0m | 2 | None |
| other | < 1.0m | 1 | None |
| other | ≥ 1.0m | 3 | None |

Thresholds come from `DetectionConfig.danger_distance` (configurable at runtime via GUI).

---

## Known Bugs & Issues (All Fixed)

### Bug 1 — Critical: FusionStage Never Added to Pipeline ✅

**File:** `main_window.py:82-89`
**Problem:** `main_window.py` adds `YOLODetectionStage` but never adds `FusionStage`. The pipeline runs Depth → YOLO → END. `FusionStage.process()` is never called.
**Impact:** `result.fused_output` is always `[]`. The system works only because `camera_thread.py` falls back to `result.obstacles`.
**Fix:** Add `self.frame_processor.add_stage(FusionStage(config=config))` after YOLO stage.

### Bug 2 — High: Output Bbox Format Mismatch ✅

**File:** `frame_processor.py:409`
**Problem:** Output bbox is `[x, y, w, h]` (OpenCV format) but the contract (`ROLES.md:150`, `FrameData` docstring) specifies `[x1, y1, x2, y2]`.
**Impact:** Downstream consumers (RadarView, future GUI) expect xyxy but get xywh. Coordinates will be wrong.
**Fix:** Convert `[x, y, w, h]` → `[x, y, x+w, y+h]` before appending.

### Bug 3 — High: Priority Thresholds Hardcoded ✅

**File:** `frame_processor.py:391-402`
**Problem:** Distance thresholds `1.0m` and `3.0m` are hardcoded. `DetectionConfig.danger_distance` (default `1.5m`, adjustable via GUI) is ignored.
**Impact:** GUI slider changes don't affect fusion priority. User sets danger to 2.0m but fusion still uses 1.0m.
**Fix:** Accept `DetectionConfig` in constructor, use its values.

### Bug 4 — High: Zone Not Passed to AlertPanel ✅

**File:** `camera_thread.py:193-195`
**Problem:** `distance_info_ready.emit(label, dist)` sends only 2 args. `alert_panel.py:143` expects 3: `(object_name, distance_m, zone)`. Zone defaults to `CENTER` always.
**Impact:** AlertPanel always shows "ZONE: CENTER" regardless of actual object position.
**Fix:** Emit zone as 3rd argument: `emit(label, dist, zone)`.

### Bug 5 — Medium: `result` Variable Fragile Scope ✅

**File:** `camera_thread.py:213-214`
**Problem:** `result` is referenced inside `if self._processor is not None` but never initialized outside. If processor exists but `process()` throws (caught by `_measure()`), `result` may be incomplete.
**Impact:** Potential `UnboundLocalError` or stale data.
**Fix:** Initialize `result = None` before the `if` block, add null check.

### Bug 6 — Medium: Radar `angle_deg` Never Computed ✅

**File:** `radar_view.py:160`
**Problem:** `obs.get("angle_deg", 90)` — neither ObstacleDetector nor FusionStage computes `angle_deg`. All blips default to 90° (center).
**Impact:** Radar shows all objects in the center regardless of actual zone.
**Fix:** Derive from zone: `left` → 45°, `center` → 90°, `right` → 135°.

### Bug 7 — Low: Stale "PLACEHOLDER" Comment ✅

**File:** `frame_processor.py:317`
**Problem:** Docstring says `PLACEHOLDER — akan diimplementasikan oleh Role 4` but the stage is now implemented.
**Impact:** Misleading to other developers.
**Fix:** Update docstring to reflect actual behavior.

### Bug 8 — Low: No Tests for FusionStage ✅

**File:** `tests/test_frame_processor.py`
**Problem:** FusionStage has zero test coverage. Imported but never tested.
**Impact:** Regressions can slip in silently.
**Fix:** Add tests for matching, priority, empty inputs, bbox format.

---

## Fix Checklist

| # | Severity | File | Fix | Status |
|---|---|---|---|---|
| 1 | Critical | `main_window.py` | Add FusionStage to pipeline | ✅ Done |
| 2 | High | `frame_processor.py` | Fix output bbox to xyxy | ✅ Done |
| 3 | High | `frame_processor.py` | Use config thresholds for priority | ✅ Done |
| 4 | High | `camera_thread.py` | Pass zone to distance_info_ready | ✅ Done |
| 5 | Medium | `camera_thread.py` | Initialize result before if block | ✅ Done |
| 6 | Medium | `radar_view.py` | Compute angle_deg from zone | ✅ Done |
| 7 | Low | `frame_processor.py` | Remove stale placeholder comment | ✅ Done |
| 8 | Low | `test_frame_processor.py` | Add FusionStage tests | ✅ Done |
