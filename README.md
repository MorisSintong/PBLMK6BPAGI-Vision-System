# PBLMK6BPAGI-Vision-System

Modul vision untuk security robot berbasis **PyQt6 + OpenCV + Intel RealSense D455 + YOLOv8**.

## Overview

Aplikasi desktop untuk obstacle avoidance pada security robot:
- **Depth sensing** — Intel RealSense D455 dengan multi-stage filtering (decimation, spatial, temporal, hole-filling)
- **Object detection** — YOLOv8 dual-model (RGB + Depth), GPU-accelerated dengan FP16 pada RTX A4000
- **Dark mode adaptation** — CLAHE preprocessing + model swap otomatis ke depth model saat low light
- **Sensor fusion** — Depth + YOLO overlap matching dengan threshold adaptif dan priority matrix
- **Gap-based navigation** — Polar histogram (VFH-lite) dengan gap finding, steering output, dan speed mapping
- **Auto-switch view** — Beralih otomatis antara RGB dan Depth view berdasarkan pencahayaan ambient (hysteresis: <35 masuk dark, >50 keluar dark)
- **Real-time GUI** — PyQt6 dengan tampilan: Auto/RGB/Depth (auto-switch), Radar 90° FOV, AlertPanel, ControlsPanel

## Arsitektur

```
main.py → MainWindow
              ├── DepthView (RGB / Depth, auto-switch berdasarkan level cahaya)
              ├── ControlsPanel (start/stop, thresholds, Auto/RGB/Depth view mode)
              ├── AlertPanel (info objek + zona + aksi)
              ├── RadarView (90° FOV wedge, cached static background)
              └── CameraThread
                    ├── RealSense acquisition thread (thread terpisah + queue)
                    └── FrameProcessor (Chain of Responsibility)
                          ├── DepthProcessingStage (LUT colormap + multi-zone)
                          ├── YOLODetectionStage (dual-model swap + CLAHE)
                          ├── FusionStage (overlap matching + priority matrix)
                          ├── NavigationStage (gap-based steering, VFH-lite)
                          └── VisualAnnotationStage (HUD + steering arrow)
```

## Fitur Utama

| Fitur | Status | Detail |
|-------|--------|--------|
| RealSense D455 capture | ✅ | 640x480 @ 30fps, depth + RGB streams, unfiltered depth untuk model |
| Depth filters | ✅ | Spatial, temporal, hole-filling (decimation configurable) |
| Multi-zone detection | ✅ | Zona LEFT / CENTER / RIGHT |
| YOLOv8 dual-model | ✅ | `ModelRGB_V4.2.pt` (normal) + `ModelDepth_V4.pt` (dark mode) |
| FP16 inference | ✅ | Auto-detected pada CUDA GPU, ~2x lebih cepat di Tensor Cores |
| GPU warm-up | ✅ | Dummy inference saat load untuk pre-compile CUDA kernels |
| CLAHE dark mode | ✅ | LAB color space enhancement untuk scene temaram |
| LUT depth colormap | ✅ | Pre-computed 256-entry LUT, ~3x lebih cepat dari mask approach |
| Sensor fusion | ✅ | Overlap matching + direct depth sampling + priority matrix |
| Visual annotation | ✅ | HUD corner brackets, label, global status bar |
| Radar view | ✅ | 90° FOV wedge real-time, cached static background pixmap |
| Alert panel | ✅ | Stylesheet update hanya saat status berubah (tanpa recalc redundan) |
| Threshold controls | ✅ | Jarak danger/warning dapat diatur, propagasi ke semua stage |
| Lazy depth model | ✅ | Depth model di-load hanya pada first dark frame (hemat VRAM) |
| Separate acquisition thread | ✅ | Capture kamera terpisah dari pemrosesan via queue |
| Auto-switch view | ✅ | Beralih RGB/Depth otomatis berdasarkan pencahayaan (hysteresis) |

## Struktur Project

```
├── main.py                    # Entry point
├── ROLES.md                   # Pembagian peran anggota tim
├── PROGRESS.md                # Dokumentasi progress
├── flow.md                    # Dokumentasi arsitektur & alur data
├── data-collection.md         # Panduan akuisisi dataset (R5)
├── environment.yml            # Dependensi conda/pip
├── pyproject.toml             # Konfigurasi Ruff + pytest
├── tests/                     # Test suite (147 tests)
│   ├── test_frame_processor.py    # 92 tests — pipeline, fusion, navigation, dark mode, hysteresis, annotation
│   ├── test_obstacle_detector.py  # 31 tests — detection, zona, filtering, thread safety
│   ├── test_camera_thread.py      # 24 tests — signal, threshold, QImage, cache
│   └── benchmark.py               # Benchmark suite (17 kriteria dari ROLES.md)
├── Vision/
│   ├── src/                   # Modul vision utama
│   │   ├── camera_thread.py   # Capture + filter + pipeline (acq thread terpisah, 6 Qt signals incl. error)
│   │   ├── frame_processor.py # Orchestrator pipeline (5 stage)
│   │   ├── obstacle_detector.py # Depth obstacle detection (tanpa frame copy)
│   │   ├── yolowrapper.py     # YOLOv8 inference (FP16, warm-up, batch transfer)
│   │   └── recorder.py        # Utilitas recording
│   ├── models/                # (.gitignore) Model weights
│   │   ├── ModelRGB_V4.2.pt   # Model RGB (R2 latest)
│   │   ├── ModelDepth_V4.pt   # Model Depth (R2 latest, trained pada unfiltered depth)
│   │   └── security_best.pt   # Model fallback
│   └── inc/                   # Konfigurasi + logging
│       ├── detection_config.py # Threshold deteksi
│       ├── camera_config.py   # Parameter kamera (RealSense + webcam)
│       └── logging_config.py  # Logging terpusat
├── GUI/
│   ├── src/                   # Komponen PyQt6
│   │   ├── main_window.py     # Layout window + wiring sinyal
│   │   ├── depth_view.py      # Display kamera (2 mode: RGB/Depth, visible-only updates)
│   │   ├── controls_panel.py  # Start/stop + threshold + Auto/RGB/Depth view mode
│   │   ├── alert_panel.py     # Info objek + alert (cached stylesheets)
│   │   └── radar_view.py      # Radar 90° FOV (cached background pixmap)
│   └── inc/                   # Konfigurasi UI + style
│       ├── ui_config.py       # Konstanta UI + threshold
│       └── styles.py          # Global stylesheet + color constants
└── Doc/
    ├── problems_audit_report.md      # Audit report historis (arsip)
    └── model_evaluation_report_v4.md # Evaluasi model R5
```

## Requirements

- Python 3.10
- Conda (disarankan)
- NVIDIA GPU (opsional, untuk YOLOv8 GPU inference dengan FP16)
- Intel RealSense D455 (opsional, webcam sebagai fallback)

## Setup

```bash
# Clone repository
git clone https://github.com/username/PBLMK6BPAGI-Vision-System.git
cd PBLMK6BPAGI-Vision-System

# Buat conda environment
conda env create -f environment.yml
conda activate depth-obstacle-detector

# Download model weights
# Letakkan ModelRGB_V4.2.pt dan ModelDepth_V4.pt di Vision/models/
```

## Menjalankan Aplikasi

```bash
python main.py
```

## Menjalankan Test

```bash
# Jalankan semua test
python -m pytest tests/ -v

# Jalankan file test tertentu
python -m pytest tests/test_frame_processor.py -v
```

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_frame_processor.py` | 92 | FrameData, PipelineStage, FrameProcessor, DepthProcessingStage (LUT), FusionStage (matching, priority, zona, dark mode, overlap), YOLODetectionStage (dark/bright/CLAHE/dual-model/hysteresis), NavigationStage (clear/blocked/steering/safety override/speed), VisualAnnotationStage (RGB + depth colormap + nav HUD), integrasi pipeline penuh |
| `test_obstacle_detector.py` | 31 | Detection, zona, filtering (min_area, max_area_ratio, distance), priority, frame handling (regresi no-copy), buffer reuse, thread safety, output contract |
| `test_camera_thread.py` | 24 | Instansiasi, threshold (validasi + propagasi), BGR→QImage (integritas pixel, grayscale, dimensi), empty depth cache, thread lifecycle, signal (frame_pair, distance, obstacles, navigation, light_mode) |
| **Total** | **147** | |

## Arsitektur Pipeline

Pipeline menggunakan pola **Chain of Responsibility** dengan 5 stage:
- Setiap stage mengimplementasikan `PipelineStage` ABC
- Data mengalir sebagai `FrameData` dataclass
- Stage dapat di-enable/disable secara modular
- Exception di stage manapun ditangkap, error dicatat di `FrameData.errors`

```python
# Contoh penggunaan
config = DetectionConfig()
processor = FrameProcessor(config)
processor.add_stage(YOLODetectionStage(
    model_path="Vision/models/ModelRGB_V4.2.pt",
    depth_model_path="Vision/models/ModelDepth_V4.pt",
))
processor.add_stage(FusionStage(config=config))
processor.add_stage(VisualAnnotationStage(config=config))

result = processor.process(rgb_frame, depth_frame, depth_scale=0.001)
# result.rgb_frame — frame teranotasi (HUD)
# result.depth_colormap — visualisasi zona bahaya (LUT)
# result.obstacles — daftar obstacle dari depth
# result.detections — deteksi YOLO
# result.fused_output — hasil fusion (class + distance + priority)
```

## Catatan

- Dukungan D455 menggunakan `pyrealsense2`
- Jika RealSense tidak tersedia, aplikasi memakai webcam biasa (RGB only)
- Pada Windows, capture kamera memprioritaskan backend DirectShow
- Model weights tidak di-track di git (lihat `.gitignore`)
- **GPU dipaksa aktif walau laptop berjalan di baterai** (`Vision/src/gpu_utils.py` memanggil `torch.cuda._lazy_init()` + `nvidia-smi -lgc <max>` untuk bypass throttle Windows power management; jalankan sebagai Administrator agar lock clock berhasil)
- Dual-model: RGB model untuk kondisi terang, depth model untuk kondisi gelap (di-load lazy)
- Unfiltered depth frame disimpan sebelum RS filters untuk depth model inference

## Fitur Mendatang (Open3D Roadmap)

Paket `open3d`, `pyqtgraph`, dan `pyserial` sudah disiapkan di `environment.yml` untuk fitur-fitur di bawah ini. Saat ini belum di-import di codebase, tetapi arsitektur pipeline siap menerima integrasi 3D.

| Fitur | Tingkat Kesulitan | Estimasi Waktu | Catatan |
|---|---|---|---|
| **Debug point-cloud viewer** (jendela terpisah) | Mudah | 2–3 hari | Konversi RGB+depth → `o3d.geometry.PointCloud` setiap frame. Perlu downsampling agar FPS tetap bagus. |
| **Embedded 3D view** di GUI | Sedang | 3–5 hari | Tambah tombol "3D" di `ControlsPanel` dan page baru di `DepthView`. Integrasi Open3D + PyQt6 memerlukan bridge rendering (offscreen buffer atau widget terpisah). |
| **3D obstacle clustering** (DBSCAN/RANSAC) | Sedang | 4–7 hari | Ganti `cv2.findContours` dengan clustering point cloud untuk bounding box 3D. Lebih akurat untuk obstacle bertumpuk, tetapi lebih lambat dari deteksi 2D saat ini. |
| **3D voxel navigation / collision map** | Sedang-Sulit | 1–2 minggu | Ganti polar histogram 2D dengan voxel occupancy grid. Perlu ground-plane removal agar lantai tidak dianggap obstacle. |
| **SLAM / mesh reconstruction** | Sangat Sulit | berminggu-minggu | Di luar scope obstacle avoidance saat ini; memerlukan perubahan arsitektur besar. |

### Risiko utama Open3D

- **Waktu import / startup**: Open3D lambat di-load. Direkomendasikan lazy-import di dalam stage/widget saja.
- **FPS drop**: Point cloud 640×480 penuh sangat berat. Wajib `voxel_down_sample(voxel_size=0.02–0.05)` dan crop ROI.
- **Integrasi PyQt6**: Visualizer Open3D bisa konflik dengan event loop Qt. Solusi umum: jalankan di thread/proses terpisah.
- **Intrinsik kamera**: Perlu fx, fy, cx, cy dari `pyrealsense2` profile agar point cloud tidak distorsi.

Rekomendasi langkah pertama: implementasi **debug point-cloud viewer** karena risiko paling rendah dan tidak mengganggu pipeline/GUI yang sudah berjalan.

## Tim

| Role | Tanggung Jawab |
|------|----------------|
| R1 (Moris) | ML Pipeline / Integration |
| R2 (Husein) | YOLOv8 Specialist |
| R3 (Long) | Depth / Camera |
| R4 (Rasyid) | Sensor Fusion |
| R5 (Hamid) | Dataset / Testing |
| R6 (Adel) | GUI / Operator Console |
