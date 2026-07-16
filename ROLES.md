# Team Roles & Responsibilities

> AI Agent: Baca dokumen ini untuk memahami peran kamu. Kerjakan sesuai sasaran dan kriteria keberhasilan. Koordinasikan dengan role terkait sesuai tabel hubungan.

---

## Role 1 — ML Pipeline Architect / Frame Processor Lead
**Orang:** Moris

### Tanggung jawab
- Merancang dan mengimplementasikan pipeline vision end-to-end (5 stages)
- Mendefinisikan kontrak data antar stage
- Mengorkestrasi aliran data: Raw Frame → Depth → YOLO → Fusion → Navigation → Annotation → Output
- Code review untuk semua PR dari Role 2-6
- Performance optimization (FP16, LUT, lazy loading, buffer reuse)

### File
| File | Keterangan |
|---|---|
| `Vision/src/frame_processor.py` | Pipeline utama (5 stage: Depth, YOLO, Fusion, Navigation, Annotation) |
| `Vision/src/camera_thread.py` | Integrasi pipeline + acquisition thread |
| `Vision/src/yolowrapper.py` | YOLOv8 wrapper (FP16, warm-up, batch transfer) |
| `Vision/inc/detection_config.py` | Konfigurasi terpusat |
| `Vision/inc/camera_config.py` | Konfigurasi kamera (RealSense + webcam) |
| `main.py` | Inisialisasi FrameProcessor |

### Selesai bila
- `frame_processor.py` menerima frame dari CameraThread, menjalankan semua 5 stage, mengembalikan frame teranotasi
- Pipeline ≥25 FPS (RealSense) / ≥30 FPS (webcam)
- Semua kontrak antar stage didokumentasikan dan disetujui tim
- Semua file konfigurasi dikelola sebagai single source of truth
- 194/194 tests pass

---

## Role 2 — YOLOv8 Object Detection Specialist
**Orang:** Husein

### Tanggung jawab
- Membangun `YOLOWrapper` class untuk model loading, inference, class mapping
- Fine-tuning YOLOv8 dengan dataset dari Role 5
- Optimasi inference (ONNX, TensorRT, reduced input size)
- Output per frame: `List[Detection]` (dataclass) dengan format `{bbox, class_id, class_name, confidence}`

### File
| File | Keterangan |
|---|---|
| `Vision/src/yolowrapper.py` | YOLO wrapper (FP16, warm-up, 320px, batch transfer) |
| `Vision/models/ModelRGB_V4.2.pt` | RGB model (latest) |
| `Vision/models/ModelDepth_V4.pt` | Depth model (latest, trained on unfiltered depth) |
| `environment.yml` | Dependency ultralytics 8.4.77 |

### Input dari
- **Role 5** — dataset train/val berlabel

### Output ke
- **Role 4** — `FrameData.detections` (List[Detection] dataclass)

### Format kontrak output
```python
@dataclass
class Detection:
    class_id: int        # COCO class index
    class_name: str      # "person", "mobil", "motor"
    confidence: float    # 0.0 - 1.0
    bbox: List[int]      # [x1, y1, x2, y2] xyxy format
```

### Selesai bila
- Model YOLOv8 berjalan inference pada setiap frame RGB pipeline
- Dual-model: RGB model + Depth model untuk dark mode
- Latency ≤50ms (GPU) / ≤100ms (CPU)
- Akurasi ≥70% mAP@0.5 pada kelas target di lingkungan outdoor
- Stabil pada pencahaayan bervariasi (penurunan akurasi ≤15%)
- API bersih dan terdokumentasi

---

## Role 3 — Depth Processing & Obstacle Detection Engineer
**Orang:** Long

### Tanggung jawab
- Depth filtering: temporal, spatial edge-preserving, hole-filling, decimation (pakai pyrealsense2 SDK)
- Multi-zone detection: LEFT / CENTER / RIGHT
- Depth colormap: merah (danger), kuning (warning), hijau (safe) — sekarang LUT-based
- Unfiltered depth capture sebelum filters (untuk depth model)
- Obstacle detection dengan bounding box + distance label

### File
| File | Keterangan |
|---|---|
| `Vision/src/camera_thread.py` | Depth filtering, unfiltered capture, acquisition thread |
| `Vision/src/obstacle_detector.py` | Obstacle detection (no frame copy, buffer reuse) |
| `Vision/inc/camera_config.py` | RealSense D455 settings + filter parameters |

### Output ke
- **Role 4** — `FrameData.obstacles` (List[Dict])
- **Role 5** — Depth module untuk benchmark

### Format kontrak output
```python
[
    {
        "bbox":        [x, y, w, h],  # bounding box (xywh)
        "distance_m":  float,          # jarak dalam meter
        "zone":        "left"|"center"|"right",
        "area_px":     int,            # cv2.contourArea
        "priority":    float,          # inverse distance (raw)
    },
    ...
]
```

### Selesai bila
- ObstacleDetector berjalan real-time, akurat untuk objek 0.3m–5m
- Depth noise berkurang 30% (indoor) / 20% (outdoor) dari raw
- Colormap menampilkan zona merah/kuning/hijau sesuai threshold (LUT-based)
- 3 sektor (left/center/right) dengan jarak minimum per sektor
- Unfiltered depth frame tersedia untuk depth model

### Catatan outdoor
RealSense D455 terganggu sinar matahari langsung. Uji pagi/sore, mendung, atau area teduh.

---

## Role 4 — Sensor Fusion Engineer
**Orang:** Rasyid

### Tanggung jawab
- Menggabungkan YOLO detections (R2) + depth obstacles (R3)
- Two-pass architecture: PASS 1 YOLO-first direct depth sampling, PASS 2 depth-only obstacles
- Prioritas obstacle: person dekat > obstacle dekat > lainnya
- Adaptive overlap threshold (0.3 dark, 0.5 normal)

### File
| File | Keterangan |
|---|---|
| `Vision/src/frame_processor.py` | FusionStage implementation |
| `Vision/src/fusion.md` | FusionStage documentation |

### Input dari
- **Role 2** — `FrameData.detections` (List[Detection])
- **Role 3** — `FrameData.obstacles` (List[Dict])

### Output ke
- **Role 6** — `FrameData.fused_output` (List[Dict])

### Format kontrak output
```python
[
    {
        "object_class":  str,                # "person", "chair", "obstacle"
        "distance_m":    float,
        "zone":          "left"|"center"|"right",
        "priority":      int,                # 0 = paling bahaya
        "bbox":          [x1, y1, x2, y2],  # xyxy format
        "action":        str | None,         # "STOP" atau None
    },
    ...
]
```

### Aturan prioritas
| Pass | Class | Distance | Priority |
|---|---|---|---|
| PASS 1 | person | < danger_distance | 0 (STOP) |
| PASS 1 | other | < danger_distance | 1 |
| PASS 1 | person | < warning_distance | 2 |
| PASS 1 | other | ≥ danger_distance | 3 |
| PASS 2 | obstacle | < 0.5m | 1 |
| PASS 2 | obstacle | < 1.0m | 2 |
| PASS 2 | obstacle | ≥ 1.0m | 3 |

### Implementasi
- **PASS 1**: Direct depth sampling dari YOLO bbox (center 60%, 25th percentile)
- **PASS 2**: Overlap metric `intersection / min(depth_area, yolo_area)` untuk cek covered
- **Adaptive threshold**: 0.3 saat dark/low confidence, 0.5 normal
- **Config**: `DetectionConfig.danger_distance` dan `warning_distance` (bukan hardcoded)

### Selesai bila
- ✅ Setiap deteksi YOLO punya jarak akurat via direct depth sampling
- ✅ Prioritas diurutkan benar
- ✅ Output terstruktur siap dikonsumsi Role 6
- ✅ 92 tests covering fusion + navigation + annotation logic

---

## Role 5 — Dataset, Testing & Performance Engineer
**Orang:** Hamid

### Tanggung jawab
- Akuisisi rekaman RealSense outdoor (berbagai skenario: terik, mendung, bayangan)
- Labeling dataset YOLO (LabelImg / CVAT / Roboflow)
- Split train/val/test — serahkan train/val ke R2, simpan test untuk evaluasi
- Bangun test harness (ukur latency tiap stage)
- Benchmark end-to-end ≤100ms (P95)
- Regression test otomatis

### File
| File | Keterangan |
|---|---|
| `data-collection.md` | Panduan akuisisi dataset |
| `Doc/model_evaluation_report_v4.md` | Laporan evaluasi model V4.2 + V4 |
| Test scripts | Benchmark harness |

### Output ke
- **Role 2** — Dataset train/val berlabel
- **Semua role** — Laporan benchmark & regression test

### Selesai bila
- Dataset pelatihan: ≥3 kelas, ≥300 frame berlabel (train+val)
- Dataset uji: ≥3 skenario, ≥200 frame berlabel
- Test harness mengukur latency setiap stage secara independen
- Laporan performa: precision, recall, MAE, latency P50/P95/P99
- Pipeline end-to-end ≤100ms (P95) pada hardware target

---

## Role 6 — GUI Maintenance & Operator Console Engineer
**Orang:** Adel

### Tanggung jawab
- Memperbarui AlertPanel: nama objek, jarak, zona, status bahaya, rekomendasi aksi
- Integrasi RadarView (90° FOV, data nyata dari pipeline, cached background)
- DepthView anotasi: bbox + label + jarak (visible-only updates, auto-switch RGB/Depth)
- Wiring sinyal dari FrameProcessor ke GUI
- Maintain stabilitas seluruh widget GUI
- Performance optimization (cached pixmaps, change-only stylesheets)

### File
| File | Keterangan |
|---|---|
| `GUI/src/main_window.py` | Wiring sinyal + pipeline assembly |
| `GUI/src/depth_view.py` | Display kamera (2 mode: RGB/Depth, auto-switch, visible-only updates) |
| `GUI/src/controls_panel.py` | Panel kontrol + threshold sliders + Auto/RGB/Depth view mode |
| `GUI/src/alert_panel.py` | Panel info + alert (cached stylesheets) |
| `GUI/src/radar_view.py` | Radar 90° FOV (cached background pixmap) |
| `GUI/inc/ui_config.py` | Konstanta UI |
| `GUI/inc/styles.py` | Stylesheet + color constants |
| `main.py` | Qt bootstrap |

### Input dari
- **Role 4** — `FrameData.fused_output`

### Selesai bila
- ✅ AlertPanel format: `PERSON | 2.3 m | CENTER | STOP`
- ✅ RadarView menampilkan posisi obstacle real-time (cached background)
- ✅ DepthView anotasi: bbox + label + jarak (visible-only updates)
- ✅ Informasi deteksi tampil ≤50ms setelah frame diproses
- ✅ Operator bisa ambil keputusan hanya dengan melihat GUI
- ✅ Semua widget berfungsi tanpa bug

---

## Hubungan Antar Role

| Dari | Ke | Apa |
|---|---|---|
| R1 | R2, R3, R4 | Kontrak API + code review |
| R1 | R5 | Spesifikasi konfigurasi |
| R1 | R6 | Spesifikasi API FrameProcessor |
| R2 | R4 | `List[Detection]` per frame |
| R2 | R5 | Model YOLO untuk benchmark |
| R3 | R4 | Zones, distances, mask, colormap |
| R4 | R6 | Structured fusion output |
| R5 | R2 | Dataset train/val |
| R5 | Semua | Laporan benchmark |

### Status Parallel vs Sequential

| Role | Status |
|---|---|
| R1 | ✅ Selesai (97% — hardware test pending) |
| R2 | ✅ Model selesai (88% — outdoor light stability pending) |
| R3 | ⏳ Outdoor test pending |
| R4 | ✅ Selesai (100%) |
| R5 | ✅ Dataset + benchmark selesai (86% — regression test pending) |
| R6 | ✅ Selesai (83% — 30-min soak + display latency pending) |
