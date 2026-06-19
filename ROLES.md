# Team Roles & Responsibilities

> AI Agent: Baca dokumen ini untuk memahami peran kamu. Kerjakan sesuai sasaran dan kriteria keberhasilan. Koordinasikan dengan role terkait sesuai tabel hubungan.

---

## Role 1 — ML Pipeline Architect / Frame Processor Lead
**Orang:** Moris

### Tanggung jawab
- Merancang dan mengimplementasikan pipeline vision end-to-end
- Mendefinisikan kontrak data antar stage
- Mengorkestrasi aliran data: Raw Frame → Depth → Obstacle → YOLO → Fusion → Output
- Code review untuk semua PR dari Role 2-6

### File
| File | Keterangan |
|---|---|
| `Vision/src/frame_processor.py` | Pipeline utama |
| `Vision/src/camera_thread.py` | Integrasi pipeline |
| `Vision/inc/detection_config.py` | Konfigurasi terpusat |
| `Vision/inc/camera_config.py` | Konfigurasi kamera |
| `main.py` | Inisialisasi FrameProcessor |

### Selesai bila
- `frame_processor.py` menerima frame dari CameraThread, menjalankan semua stage, mengembalikan frame teranotasi
- Pipeline ≥25 FPS (RealSense) / ≥30 FPS (webcam)
- Semua kontrak antar stage didokumentasikan dan disetujui tim
- Semua file konfigurasi dikelola sebagai single source of truth

---

## Role 2 — YOLOv8 Object Detection Specialist
**Orang:** Husein

### Tanggung jawab
- Membangun `YOLOWrapper` class untuk model loading, inference, class mapping
- Fine-tuning YOLOv8-nano dengan dataset dari Role 5
- Optimasi inference (ONNX, TensorRT, reduced input size)
- Output per frame: `List[Dict]` dengan format `{bbox, class_id, class_name, confidence}`

### File
| File | Keterangan |
|---|---|
| `Vision/src/yolo_wrapper.py` | File baru — YOLO wrapper |
| `environment.yml` | Dependency ultralytics |

### Input dari
- **Role 5** — dataset train/val berlabel

### Output ke
- **Role 4** — `FrameData.detections` (List[Dict])

### Format kontrak output
```python
[
    {
        "bbox":        [x1, y1, x2, y2],  # format xyxy, int
        "class_id":    int,                # indeks kelas COCO
        "class_name":  str,                # "person", "chair", dll.
        "confidence":  float,              # 0.0 - 1.0
    },
    ...
]
```

### Selesai bila
- Model YOLOv8 berjalan inference pada setiap frame RGB pipeline
- Latency ≤50ms (GPU) / ≤100ms (CPU)
- Akurasi ≥70% mAP@0.5 pada kelas target di lingkungan outdoor
- Stabil pada pencahayaan bervariasi (penurunan akurasi ≤15%)
- API bersih dan terdokumentasi

---

## Role 3 — Depth Processing & Obstacle Detection Engineer
**Orang:** Long

### Tanggung jawab
- Depth filtering: temporal, spatial edge-preserving, hole-filling, decimation (pakai pyrealsense2 SDK)
- Multi-zone detection: LEFT / CENTER / RIGHT
- Depth colormap: merah (danger), kuning (warning), hijau (safe)
- Obstacle detection dengan bounding box + distance label

### File
| File | Keterangan |
|---|---|
| `Vision/src/camera_thread.py` | Depth filtering & colormap di pipeline |

### Output ke
- **Role 4** — `FrameData.obstacles` (List[Dict])
- **Role 5** — Depth module untuk benchmark

### Format kontrak output
```python
[
    {
        "bbox":        [x, y, w, h],  # bounding box
        "distance_m":  float,          # jarak dalam meter
        "zone":        "left"|"center"|"right",
        "area_px":     int,            # luas kontur
    },
    ...
]
```

### Selesai bila
- ObstacleDetector berjalan real-time, akurat untuk objek 0.3m–5m
- Depth noise berkurang 30% (indoor) / 20% (outdoor) dari raw
- Colormap menampilkan zona merah/kuning/hijau sesuai threshold
- 3 sektor (left/center/right) dengan jarak minimum per sektor
- Bounding box + label distance di frame RGB

### Catatan outdoor
RealSense D455 terganggu sinar matahari langsung. Uji pagi/sore, mendung, atau area teduh.

---

## Role 4 — Sensor Fusion Engineer
**Orang:** Rasyid

### Tanggung jawab
- Menggabungkan YOLO detections (R2) + depth obstacles (R3)
- Proyeksi 2D bbox ke 3D (intrinsik kamera)
- Prioritas obstacle: person dekat > obstacle dekat > lainnya
- Ground-plane estimation (indoor + outdoor terrain)

### File
| File | Keterangan |
|---|---|
| `Vision/src/obstacle_detector.py` | Extend dengan hasil fusion |
| YOLO wrapper (dari R2) | Konsumsi |
| `environment.yml` | open3d (opsional) |

### Input dari
- **Role 2** — `FrameData.detections`
- **Role 3** — `FrameData.obstacles`

### Output ke
- **Role 6** — `FrameData.fused_output` (List[Dict])

### Format kontrak output
```python
[
    {
        "object_class":  str,                # "person", "chair", dll.
        "distance_m":    float,
        "zone":          "left"|"center"|"right",
        "priority":      int,                # 0 = paling bahaya
        "bbox":          [x1, y1, x2, y2],
        "action":        str | None,         # "STOP", "BELOK KANAN", dll.
    },
    ...
]
```

### Aturan prioritas
| Kondisi | Priority |
|---|---|
| Person < 1m | 0 (STOP) |
| Obstacle < 1m | 1 |
| Person < 3m | 2 |
| Lainnya | 3+ |
| Tidak ada obstacle | list kosong `[]` |

### Selesai bila
- Setiap deteksi YOLO punya jarak akurat (±10% untuk 0.5–4m)
- Prioritas diurutkan benar
- Ground-plane estimation kurangi false positive ≥30% (indoor) / ≥20% (outdoor)
- Output terstruktur siap dikonsumsi Role 6

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
| `Vision/src/recorder.py` | Rekam stream |
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
- Integrasi RadarView (180°, data nyata dari pipeline)
- DepthView overlay: bounding box + label kelas + jarak
- Wiring sinyal dari FrameProcessor ke GUI
- Maintain stabilitas seluruh widget GUI

### File
| File | Keterangan |
|---|---|
| `GUI/src/Alert_panel.py` | Panel info + alert |
| `GUI/src/main_window.py` | Wiring sinyal |
| `GUI/src/depth_view.py` | Display kamera |
| `GUI/src/controls_panel.py` | Panel kontrol |
| `GUI/src/radar_view.py` | Radar 180° |
| `GUI/inc/ui_config.py` | Konstanta UI |
| `GUI/inc/styles.py` | Stylesheet |
| `main.py` | Qt bootstrap |

### Input dari
- **Role 4** — `FrameData.fused_output`

### Selesai bila
- AlertPanel format: `PERSON | 2.3 m | CENTER | STOP`
- RadarView menampilkan posisi obstacle real-time (bukan dummy)
- DepthView anotasi: bbox + label + jarak
- Informasi deteksi tampil ≤50ms setelah frame diproses
- Operator bisa ambil keputusan hanya dengan melihat GUI
- Semua widget berfungsi tanpa bug
- Sistem stabil ≥30 menit streaming kontinu

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

### Parallel vs Sequential

| Role | Status |
|---|---|
| R1 | Sepanjang fase (review + maintain) |
| R2 + R3 | **Paralel** — tidak saling bergantung |
| R4 | **Nunggu** R2 + R3 selesai |
| R5 | Sepanjang fase (dataset + testing) |
| R6 | **Nunggu** R4 selesai |
