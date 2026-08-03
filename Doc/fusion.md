# FusionStage — Dokumentasi Sensor Fusion

## Overview

`FusionStage` menggabungkan data semantik dari YOLO (R2) dengan data spasial dari Depth (R3) untuk menghasilkan satu daftar obstacle terpadu. Ia menjawab pertanyaan: **"Apakah objek ini, dan seberapa jauh jaraknya?"**

### Posisi Pipeline

```
CameraThread
  └─ FrameProcessor
       ├─ Stage A: DepthProcessingStage (R3)  → FrameData.obstacles
       ├─ Stage B: YOLODetectionStage (R2)    → FrameData.detections
       ├─ Stage C: FusionStage (R4)           → FrameData.fused_output  ← INI
       ├─ Stage D: NavigationStage (R1)       → FrameData.navigation
       └─ Stage E: VisualAnnotationStage (R1) → FrameData.rgb_frame + depth_colormap (HUD)
```

---

## Kontrak Input

### Dari R3 (DepthProcessingStage) — `FrameData.obstacles`

```python
[
    {
        "bbox":        [x, y, w, h],          # format OpenCV xywh, pixel
        "distance_m":  float,                  # meter (percentile ke-5)
        "zone":        "left"|"center"|"right",
        "area_px":     int,                    # cv2.contourArea — pixel blob aktual
        "priority":    float,                  # inverse distance (mentah)
    },
    ...
]
```

### Dari R2 (YOLODetectionStage) — `FrameData.detections`

```python
[
    Detection(
        bbox:        [x1, y1, x2, y2],        # format xyxy, pixel
        class_id:    int,
        class_name:  str,                      # mis. "person", "chair"
        confidence:  float,                    # 0.0–1.0
    ),
    ...
]
```

### Dari Metadata — `FrameData.metadata`

```python
{
    "is_dark": bool,          # True if brightness < 35 (hysteresis: exit at > 50)
    "rgb_confidence": float,  # 0.0–1.0, min(brightness/128, 1.0)
}
```

---

## Arsitektur Two-Pass

FusionStage menggunakan pendekatan **two-pass**:

### PASS 1 — Direct Depth Sampling YOLO-first (dilewati di dark mode)

Untuk setiap deteksi YOLO, depth **di-sampling langsung** dari depth frame di dalam bbox YOLO. Ini memberikan class name dan jarak sekaligus dalam satu langkah — tidak perlu mencocokkan box YOLO ke contour obstacle depth.

```python
for det in data.detections:
    dist = self._sample_depth_in_bbox(depth_frame, depth_scale, det.bbox)
    if dist is None:
        continue  # Tidak ada depth valid di bbox ini — skip
    matched_yolo_indices.add(i)
    # Assign class + distance + priority
```

#### Direct Depth Sampling

```python
def _sample_depth_in_bbox(depth_frame, depth_scale, bbox):
    x1, y1, x2, y2 = bbox
    # Clamp ke batas frame
    # Gunakan 60% tengah bbox untuk menghindari pixel background di tepi
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)
    region = depth_frame[cy1:cy2, cx1:cx2].astype(np.float32) * depth_scale
    valid = region[(region >= min_dist) & (region <= max_dist)]
    return float(np.percentile(valid, 25))  # percentile ke-25
```

**Kenapa persentil ke-25?** Ini memberikan jarak ke **permukaan terdekat** dari objek — yang penting untuk collision avoidance. Region tengah 60% menghindari piksel background yang bocor ke tepi bbox.

**Kenapa PASS 1 dilewati di dark mode?** Di dark mode, YOLO berjalan pada depth colormap (bukan RGB), sehingga deteksi mungkin tidak sesuai dengan objek yang sama seperti obstacle depth. Lebih baik mengandalkan obstacle depth-only dari PASS 2.

### PASS 2 — Obstacle Depth-only

Untuk obstacle depth yang tidak dicover oleh deteksi YOLO manapun (dari PASS 1), tambahkan sebagai class "obstacle" generik. Ini menangkap objek yang terlewat oleh YOLO.

```python
for obs in data.obstacles:
    # Cek apakah sudah dicover oleh deteksi YOLO
    if already_covered_by_yolo(obs, matched_yolo_indices):
        continue
    # Filter: abaikan obstacle > 1.5m (terlalu jauh untuk obstacle generik)
    if dist > 1.5:
        continue
    # Tambahkan sebagai obstacle generik dengan priority diturunkan
```

---

## Metrik Overlap untuk PASS 2

Di PASS 2, kita perlu memeriksa apakah sebuah obstacle depth sudah dicover oleh deteksi YOLO. Kita menggunakan:

```
overlap_ratio = Area(Intersection) / min(Area(Depth), Area(YOLO))
```

Ini menggunakan **area terkecil** sebagai denominator, sehingga:
- Blob depth kecil di dalam box YOLO besar → overlap tinggi (benar)
- Box YOLO kecil di dalam blob depth besar → overlap tinggi (benar)

### Kenapa Tidak IoU?

IoU standar gagal ketika ukuran box berbeda signifikan:

```
YOLO mendeteksi "person" utuh (200 × 500px = 100.000 px²)
Depth hanya meng-cluster dada (80 × 100px = 8.000 px²)

IoU = 8.000 / (100.000 + 8.000 - 8.000) = 0.08  →  DITOLAK

overlap_ratio = 8.000 / min(8.000, 100.000) = 8.000 / 8.000 = 1.0  →  DICOVER
```

### Kenapa `min(area)` alih-alih hanya `depth_area`?

Menggunakan `min(depth_area, yolo_area)` menangani kedua kasus:
- Blob depth kecil, box YOLO besar → denominator = depth_area (sama seperti sebelumnya)
- Box YOLO kecil, blob depth besar → denominator = yolo_area (mencegah false match)

### Masalah Inflasi AABB

`cv2.boundingRect(contour)` mengembalikan **Axis-Aligned Bounding Box (AABB)** yang menyertakan ruang kosong untuk bentuk tidak beraturan. Menggunakan `area_px` (contourArea) sebagai area depth memberikan rasio **sebenarnya** dan menghindari penolakan match yang valid.

---

## Threshold Matching Adaptif

Threshold overlap beradaptasi terhadap kondisi pencahayaan:

| Kondisi | Threshold | Kenapa |
|---|---|---|
| Normal (is_dark=False, rgb_confidence ≥ 0.5) | 0.5 (50%) | Matching ketat saat YOLO andal |
| Dark atau confidence rendah | 0.3 (30%) | Matching dilonggarkan saat YOLO mungkin tidak akurat |

---

## Matriks Prioritas

### PASS 1 (deteksi YOLO dengan depth)

| Kelas | Jarak | Priority | Aksi |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < `warning_distance` | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

### PASS 2 (obstacle depth-only)

| Jarak | Priority | Kenapa |
|---|---|---|
| < 0.5m | 1 | Sangat dekat — diturunkan dari 0 untuk menghindari STOP palsu pada obstacle generik |
| < 1.0m | 2 | Dekat — level warning |
| ≥ 1.0m | 3 | Normal |

Threshold berasal dari `DetectionConfig` (dapat dikonfigurasi saat runtime via slider GUI):
- `danger_distance` (default: 1.5m)
- `warning_distance` (default: 3.0m)

---

## Format Output

```python
[
    {
        "object_class":  str,      # "person", "chair", "obstacle"
        "distance_m":    float,    # meter
        "zone":          str,      # "left" | "center" | "right"
        "priority":      int,      # 0 = paling berbahaya
        "bbox":          [x1, y1, x2, y2],  # format xyxy
        "action":        str | None,         # "STOP" atau None
    },
    ...
]
```

---

## Test Coverage

FusionStage dicover oleh 30+ test di `tests/test_frame_processor.py`:

- Matching: YOLO + depth, no match, multiple detections
- Priority: person dekat (0/STOP), obstacle dekat (1), warning (2), jauh (3)
- PASS 2 ladder: <0.5m (1), <1.0m (2), ≥1.0m (3)
- Dark mode: PASS 1 dilewati, PASS 2 aktif
- Overlap: box identik (1.0), no overlap (0.0), dengan area_px
- Depth sampling: bbox di-clamp, semua invalid, bbox sangat kecil
- Zone: left, center, right
- Config: threshold dari DetectionConfig (tidak hardcoded)
- Contract: semua key required ada