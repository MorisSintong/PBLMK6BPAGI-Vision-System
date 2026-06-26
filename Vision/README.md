# Vision Module

Dokumentasi ini menjelaskan komponen pada folder `Vision/` untuk akuisisi frame dan pemrosesan vision.

## Tujuan Modul

Modul Vision bertanggung jawab untuk:
- Mengambil frame dari kamera (RealSense / webcam)
- Menghasilkan data RGB dan Depth (filtered + unfiltered) untuk GUI dan model
- Menjalankan pipeline vision 5-stage: Depth → YOLO → Fusion → Navigation → Annotation
- Mendeteksi obstacle dan objek dengan sensor fusion

## Struktur Folder

- `src/` — logic akuisisi kamera dan pemrosesan
- `inc/` — konfigurasi parameter vision
- `models/` — model weights (`.gitignore`)

## Komponen Utama (`src`)

| File | Fungsi |
|---|---|
| `camera_thread.py` | Worker thread untuk capture kamera. Separate acquisition thread + queue(maxsize=2). Unfiltered depth capture sebelum RS filters. Mengirim frame memory-safe ke GUI via sinyal Qt. |
| `frame_processor.py` | Engine utama pipeline vision (5 stage). Chain of Responsibility dengan error handling robust. LUT-based depth colormap. Dual-model YOLO swap + CLAHE + hysteresis. Fusion two-pass architecture. NavigationStage (VFH-lite). VisualAnnotationStage draws HUD on both RGB and depth. |
| `yolowrapper.py` | Memuat model YOLOv8 dan melakukan inference. FP16 auto-detected, GPU warm-up, input_size=320, batch tensor transfer. Output `Detection` dataclass. |
| `obstacle_detector.py` | Mengekstrak informasi jarak dan prioritas dari depth frame. Tidak mengcopy/memodifikasi color frame. Reusable float32 buffer. Thread-safe `last_detections`. |
| `recorder.py` | Utilitas uji/rekam stream RealSense secara mandiri. Memiliki flag mutex agar tidak crash dengan pipeline utama. |
| `fusion.md` | Dokumentasi FusionStage: two-pass architecture, overlap metric, priority matrix. |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `detection_config.py` | Parameter threshold deteksi (min/max/danger/warning distance). |
| `camera_config.py` | Konfigurasi kamera RealSense D455 + webcam fallback. Depth filter parameters (spatial, temporal, decimation). |
| `logging_config.py` | Centralized logging (console + file output). |

## Model Weights (`models`)

| File | Fungsi |
|---|---|
| `ModelRGB_V4.2.pt` | RGB YOLO model (R2 latest, segmentation, 98.37% mAP) |
| `ModelDepth_V4.pt` | Depth YOLO model (R2 latest, trained on unfiltered depth colormap) |
| `security_best.pt` | Fallback model |

Models di `.gitignore` — tidak di-track di git. Tim harus download manual.

## Alur Singkat Kamera Saat Ini

1. `CameraThread.start_capture()` dipanggil dari GUI.
2. **Acquisition thread** mengambil frame dari hardware (RealSense atau OpenCV fallback).
3. Unfiltered depth disimpan sebelum RS filters (untuk depth model).
4. RS filters diterapkan (spatial, temporal, hole-filling).
5. Frame masuk antrian `queue(maxsize=2)` untuk processing loop.
6. **Processing loop** menarik frame dari queue, menjalankan `FrameProcessor.process()`.
7. Pipeline 5-stage: DepthProcessing → YOLODetection → Fusion → Navigation → VisualAnnotation.
8. Hasil dikonversi ke `QImage` (numpy swap + `.tobytes()` untuk thread safety) dan dipancarkan ke GUI.

## Catatan Pengembangan

- Pipeline 5-stage: DepthProcessingStage (LUT colormap), YOLODetectionStage (dual-model + CLAHE + hysteresis), FusionStage (two-pass), NavigationStage (VFH-lite gap-based steering), VisualAnnotationStage (HUD on RGB + depth).
- Dual-model: RGB model untuk kondisi terang, depth model untuk kondisi gelap (lazy-loaded).
- Dark mode hysteresis: enter dark at brightness < 35, exit at > 50 (prevents flicker).
- Auto-switch view: GUI automatically switches RGB/Depth view based on is_dark signal.
- FP16 inference aktif saat CUDA tersedia (~2x faster on Tensor Cores).
- LUT depth colormap ~3x lebih cepat dari mask approach.
- ObstacleDetector tidak mengcopy color frame (performance optimization).
