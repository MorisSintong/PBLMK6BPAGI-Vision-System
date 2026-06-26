# Data Collection Guide — PBL Vision System

**Target audience:** R5 (Hamid) and AI assistants working on dataset acquisition.

---

## 1. Objective

Collect a labeled RGB image dataset for training YOLOv8 to detect objects relevant to a security robot:
- **person** (orang)
- **motor** (motor/motorcycle)
- **mobil** (mobil/car)

The depth camera (Intel RealSense D455) captures **synchronized RGB + Depth pairs**, but YOLO training only uses the **RGB frames**. Depth frames are saved optionally for quality filtering and future 3D models.

**Minimum requirement:** ≥300 labeled frames across all classes.

**Target performance:** mAP@0.5 ≥ 70% on validation set.

---

## 2. Hardware Setup

### 2.1 Required Equipment

| Item | Notes |
|---|---|
| Intel RealSense D455 | Primary camera, provides RGB + Depth |
| Laptop/PC with GPU | NVIDIA RTX recommended, for training |
| Tripod or stable mount | Reduces motion blur |
| The robot itself | For realistic camera height/angle |

### 2.2 Camera Configuration

```
Resolution:   640 × 480 (RGB + Depth)
Frame rate:   30 FPS
Depth format: Z16 (16-bit unsigned, millimeters)
Color format: BGR8
```

**Camera height:** Mount the RealSense at the robot's expected operating height (~0.5–1.0m from ground). This ensures training data matches deployment conditions.

### 2.3 Environment Conditions

You MUST collect data under **all** these lighting conditions:

| Condition | Description | Why |
|---|---|---|
| Bright daylight | Outdoor, direct sunlight | Tests depth sensor degradation (IR saturation) |
| Overcast / shade | Outdoor, no direct sun | Good depth, moderate RGB |
| Indoor fluorescent | Office / hallway lighting | Standard indoor condition |
| Indoor dim | Low light, few lamps | Tests YOLO degradation |
| Nighttime outdoor | Dark, streetlights only | Worst case for YOLO |
| Nighttime indoor | Dark room, minimal light | Worst case for YOLO |

**Do NOT collect only in good lighting.** The whole point is to make YOLO robust in poor conditions.

---

## 3. Data Collection Procedure

### 3.1 Step-by-Step

1. **Start the RealSense camera** (via the Vision System app or RS SDK)
2. **Position the target object** (person, motor, or car) in the scene
3. **Capture frames** at the following distances for each object:

| Distance | Count per class |
|---|---|
| 0.5m – 1.0m (close) | ≥ 20 frames |
| 1.0m – 2.0m (medium) | ≥ 20 frames |
| 2.0m – 4.0m (far) | ≥ 20 frames |
| > 4.0m (background) | ≥ 10 frames (as negatives) |

4. **Vary the position:** Move the object (or camera) to cover all zones:
   - Left side of frame
   - Center of frame
   - Right side of frame
5. **Vary the angle:** Capture from different orientations (front, side, back)
6. **Capture "negative" frames:** Scenes with NO target objects (empty room, hallway, parking lot) — these teach YOLO what NOT to detect

### 3.2 Frame Selection Rules

**DO save frames that are:**
- Sharp (no motion blur)
- Well-composed (object fully visible, not cut off)
- Varied (different backgrounds, positions, lighting)

**DO NOT save frames that are:**
- Completely black (lens covered, total darkness)
- Completely white (overexposed, camera malfunction)
- Motion blurred (shaky camera during capture)
- Duplicate (consecutive frames of the same static scene — pick the best one)

### 3.3 Minimum Dataset Size

| Class | Minimum Frames | Recommended |
|---|---|---|
| person | 100 | 200 |
| motor | 100 | 200 |
| mobil | 100 | 200 |
| Negative (no object) | 30 | 50 |
| **Total** | **330** | **650** |

**Quality over quantity.** 300 well-labeled diverse frames > 1000 blurry duplicate frames.

---

## 4. Labeling

### 4.1 Bounding Box Format (YOLO)

Each frame gets a corresponding `.txt` label file. Format:

```
class_id  x_center  y_center  width  height
```

All values are **normalized** (0.0 – 1.0) relative to image dimensions.

**Example:** A person at bounding box `[120, 80, 200, 300]` in a 640×480 image:

```
x_center = (120 + 200/2) / 640 = 0.25
y_center = (80 + 300/2) / 480 = 0.417
width    = 200 / 640 = 0.3125
height   = 300 / 480 = 0.625
```

Label file `frame_0001.txt`:
```
0 0.25 0.417 0.3125 0.625
```

### 4.2 Class Mapping

| class_id | class_name |
|---|---|
| 0 | mobil |
| 1 | motor |
| 2 | person |

### 4.3 Labeling Tools

Use one of these tools:

| Tool | Type | Recommended |
|---|---|---|
| [Roboflow](https://roboflow.com) | Web-based | ✅ Yes — auto-labeling + export to YOLO format |
| [CVAT](https://cvat.ai) | Web/self-hosted | ✅ Yes — professional-grade |
| [LabelImg](https://github.com/heartexlabs/labelImg) | Desktop | OK — simple, lightweight |
| [Python script + YOLO auto-label](#44-auto-labeling-with-yolo) | Automated | ✅ Yes — fastest for large datasets |

### 4.4 Auto-Labeling with YOLO

Use the pre-trained YOLOv8n model to auto-label frames, then human-review:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Pre-trained on COCO (80 classes)

results = model.predict(
    source="frames/",
    imgsz=320,
    conf=0.3,
    save_txt=True,          # Save labels in YOLO format
    project="dataset/",
    name="labels/auto",
)
```

This auto-generates `.txt` label files. **Human review is still required** to:
- Fix misclassified objects (e.g., "motorcycle" → "motor")
- Remove false positives
- Add missing detections
- Verify class mapping matches our 3 classes

### 4.5 Labeling Quality Checklist

Before finalizing the dataset, verify:

- [ ] Every frame has a matching `.txt` label file
- [ ] No empty label files (every file has at least one annotation)
- [ ] Bounding boxes are tight around the object (not too loose, not too tight)
- [ ] Class IDs match the mapping (0=mobil, 1=motor, 2=person)
- [ ] No duplicate labels on the same object
- [ ] Occluded objects are labeled (even if partially visible)
- [ ] Objects at the edge of frame are labeled (even if partially cut off)

---

## 5. Dataset Organization

### 5.1 Folder Structure

```
dataset/
├── data.yaml                    # Dataset config (see below)
├── images/
│   ├── train/                   # 80% of frames
│   │   ├── frame_0001.png
│   │   ├── frame_0002.png
│   │   └── ...
│   └── val/                     # 20% of frames
│       ├── frame_0051.png
│       └── ...
├── labels/
│   ├── train/                   # Matching label files
│   │   ├── frame_0001.txt
│   │   ├── frame_0002.txt
│   │   └── ...
│   └── val/
│       ├── frame_0051.txt
│       └── ...
└── depth/                       # Optional: raw depth data
    ├── frame_0001.npy
    └── ...
```

### 5.2 Train/Val Split

- **80% train** — used to train the model
- **20% val** — used to evaluate during training (unseen data)

**Important:** Split by **scene**, not by frame. If you capture 50 frames of the same person in the same location, put 40 in train and 10 in val. Do NOT put 50 in train and 0 in val — this causes data leakage (model memorizes the scene, doesn't generalize).

### 5.3 `data.yaml` Configuration

```yaml
# Dataset config for YOLOv8 training
path: dataset/          # Dataset root relative to project
train: images/train/    # Train images
val: images/val/        # Val images

# Classes
names:
  0: mobil
  1: motor
  2: person

nc: 3  # Number of classes
```

---

## 6. Depth Data (Optional but Recommended)

### 6.1 Why Save Depth?

1. **Quality filtering:** Check if objects are at valid distances
2. **Future models:** Train depth-aware detection (3D bounding boxes)
3. **Validation:** Verify bounding box accuracy using depth consistency

### 6.2 How to Save Depth

Save depth as `.npy` files (NumPy arrays):

```python
import numpy as np

# During capture
depth_raw = np.asanyarray(depth_frame.get_data())  # uint16, millimeters
np.save(f"dataset/depth/{frame_name}.npy", depth_raw)
```

### 6.3 Depth Quality Check

Before saving a frame, verify depth quality:

```python
valid_pixels = np.count_nonzero(depth_raw)
total_pixels = depth_raw.size
valid_ratio = valid_pixels / total_pixels

if valid_ratio < 0.3:
    # Less than 30% valid depth pixels — bad frame, skip
    continue
```

---

## 7. Training

### 7.1 Training Command

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Start from pre-trained weights

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=320,
    batch=16,
    device=0,                # GPU (use "cpu" if no GPU)
    project="runs/train",
    name="security_robot_v1",
    patience=20,             # Early stopping if no improvement
    save=True,
    plots=True,              # Generate training plots
)
```

### 7.2 Training Targets

| Metric | Target | How to check |
|---|---|---|
| mAP@0.5 | ≥ 70% | `results.csv` → `metrics/mAP50(B)` |
| Precision | ≥ 60% | `results.csv` → `metrics/precision(B)` |
| Recall | ≥ 60% | `results.csv` → `metrics/recall(B)` |
| Training loss | Decreasing | `results.csv` → `train/box_loss` |

### 7.3 Overfitting Signs

If val loss increases while train loss decreases → **overfitting**:
- Model memorized training data, doesn't generalize
- **Fix:** More data, data augmentation, reduce epochs, use early stopping

### 7.4 Underfitting Signs

If both train and val loss are high → **underfitting**:
- Model hasn't learned enough
- **Fix:** More epochs, larger model (yolov8s/m), more data

---

## 8. Evaluation

### 8.1 Run Validation

```python
from ultralytics import YOLO

model = YOLO("runs/train/security_robot_v1/weights/best.pt")

metrics = model.val(
    data="dataset/data.yaml",
    imgsz=320,
    device=0,
)

print(f"mAP@0.5:    {metrics.box.map50:.2%}")
print(f"mAP@0.5:0.95: {metrics.box.map:.2%}")
print(f"Precision:  {metrics.box.mp:.2%}")
print(f"Recall:     {metrics.box.mr:.2%}")
```

### 8.2 Per-Class Performance

Check each class individually:

```python
for i, class_name in enumerate(["mobil", "motor", "person"]):
    print(f"{class_name}: AP@0.5 = {metrics.box.ap50[i]:.2%}")
```

If one class is significantly worse, collect more data for that class.

### 8.3 Confusion Matrix

```python
model.val(data="dataset/data.yaml", plots=True, imgsz=320)
# Generates confusion_matrix.png in runs/val/
```

Look for:
- **False positives:** Model detects object where none exists
- **False negatives:** Model misses real objects
- **Class confusion:** Model confuses person with motor, etc.

---

## 9. Common Pitfalls

| Pitfall | Why it's bad | How to avoid |
|---|---|---|
| All frames from same location | Model doesn't generalize to new environments | Collect from 5+ different locations |
| All frames in good lighting | Model fails at night | Collect in ALL lighting conditions |
| Only front-view objects | Model can't detect side/back views | Rotate object or camera |
| Duplicate consecutive frames | Wasted effort, inflates dataset size | Pick best frame per scene, skip duplicates |
| Loose bounding boxes | Model learns wrong object boundaries | Label tightly around visible object |
| Class imbalance (200 person, 20 motor) | Model biased toward majority class | Balance: roughly equal frames per class |
| No negative frames | Model detects objects everywhere | Include 10% empty scenes |

---

## 10. Deliverables Checklist

Before handing off to R2 (Husein) for training:

- [ ] ≥300 labeled RGB frames (recommended: 650+)
- [ ] Balanced across 3 classes (±20% per class)
- [ ] Balanced across lighting conditions (day/night/indoor/outdoor)
- [ ] Balanced across distances (0.5–1m, 1–2m, 2–4m)
- [ ] 80/20 train/val split (by scene, not by frame)
- [ ] YOLO-format `.txt` label files
- [ ] `data.yaml` config file
- [ ] Validation mAP@0.5 ≥ 70%
- [ ] Confusion matrix reviewed for class confusion
- [ ] Depth frames saved (optional but recommended)

---

## 11. Reference: Data Collection Script Template

```python
"""
Data collection script for RealSense D455.
Captures synchronized RGB + Depth frames.
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# Config
OUTPUT_DIR = "dataset/raw"
FRAME_INTERVAL = 0.5  # Seconds between captures (avoid duplicates)
MIN_DEPTH_VALID_RATIO = 0.3  # Minimum valid depth pixels to save

os.makedirs(f"{OUTPUT_DIR}/rgb", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/depth", exist_ok=True)

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frame_count = 0
last_capture_time = 0

print("Data Collection Mode")
print("Press SPACE to capture, ESC to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display
        cv2.imshow("RGB", color_image)

        # Depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = capture
        if key == 32:
            # Check depth quality
            valid_ratio = np.count_nonzero(depth_image) / depth_image.size
            if valid_ratio < MIN_DEPTH_VALID_RATIO:
                print(f"Skipped: depth quality too low ({valid_ratio:.1%})")
                continue

            # Check interval
            current_time = time.time()
            if current_time - last_capture_time < FRAME_INTERVAL:
                print("Skipped: too fast, wait longer")
                continue

            # Save
            frame_name = f"frame_{frame_count:04d}"
            cv2.imwrite(f"{OUTPUT_DIR}/rgb/{frame_name}.png", color_image)
            np.save(f"{OUTPUT_DIR}/depth/{frame_name}.npy", depth_image)

            frame_count += 1
            last_capture_time = current_time
            print(f"Captured: {frame_name} | Total: {frame_count}")

        # ESC = quit
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_count} frames saved to {OUTPUT_DIR}/")
```

**Usage:**
1. Run the script: `python collect_data.py`
2. Point camera at target objects
3. Press SPACE to capture (one frame every 0.5s)
4. Press ESC to quit
5. Label the captured frames with Roboflow/CVAT/LabelImg
6. Export in YOLO format
7. Split into train/val (80/20)
8. Train YOLOv8
9. Evaluate and iterate
