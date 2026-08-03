# Vision Module

Dokumentasi ini menjelaskan komponen pada folder `Vision/` untuk akuisisi frame dan pemrosesan vision.

## Tujuan Modul

Modul Vision bertanggung jawab untuk:
- Mengambil frame dari kamera (RealSense / webcam)
- Menghasilkan data RGB dan Depth (filtered + unfiltered) untuk GUI dan model
- Menjalankan pipeline vision 5-stage: Depth → YOLO → Fusion → Navigation → Annotation
- Mendeteksi obstacle dan objek dengan sensor fusion
- Merekam sesi live dan memutar ulang rekaman melalui pipeline (untuk validasi tanpa hardware)

## Struktur Folder

- `src/` — logic akuisisi kamera dan pemrosesan
- `inc/` — konfigurasi parameter vision
- `models/` — model weights (`.gitignore`)

## Komponen Utama (`src`)

| File | Fungsi |
|---|---|
| `camera_thread.py` | Worker thread untuk capture kamera. Separate acquisition thread + queue(maxsize=2). Unfiltered depth capture sebelum RS filters. Mengirim frame memory-safe ke GUI via sinyal Qt. |
| `frame_processor.py` | Engine utama pipeline vision (5 stage). Chain of Responsibility dengan error handling robust. LUT-based depth colormap. Dual-model YOLO swap + CLAHE + hysteresis. Fusion two-pass architecture. NavigationStage (VFH-lite). VisualAnnotationStage draws HUD on both RGB and depth. |
| `yolowrapper.py` | Memuat model YOLOv8 dan melakukan inference. FP16 auto-detected, GPU warm-up, input_size=320, batch tensor transfer. Output `Detection` dataclass. Memvalidasi kontrak model saat load: file exists, task detect/segment, class set `mobil/motor/person` — weight yang tidak cocok ditolak dengan `ModelValidationError`. |
| `obstacle_detector.py` | Mengekstrak informasi jarak dan prioritas dari depth frame. Tidak mengcopy/memodifikasi color frame. Reusable float32 buffer. Thread-safe `last_detections`. |
| `video_recorder.py` | Non-blocking recording API (start/stop/save). Saves RGB AVI + depth NPY + metadata JSON. CLI mode available. |
| `video_playback_thread.py` | Replays recorded RGB+depth videos through full 5-stage pipeline. Supports individual NPY + stacked NPY depth formats. |
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

## Recording & Playback API

### VideoRecorder

`video_recorder.py` menyediakan dua mode:

**Mode GUI (non-blocking)** — dipanggil dari pipeline:
```python
from Vision.src.video_recorder import VideoRecorder
rec = VideoRecorder(save_dir="data/recordings")
rec.start_recording()
rec.record_frame(rgb_bgr, depth_raw_uint16, depth_filtered_uint16)
# ... ulangi untuk setiap frame ...
rec.stop_recording()  # Simpan AVI + NPY + metadata.json
```

**Mode CLI (standalone blocking)** — tanpa GUI, untuk akuisisi dataset:
```bash
python -m Vision.src.video_recorder --output data/recordings --duration 60
```

Output per session: `data/recordings/recording_YYYYMMDD_HHMMSS/`
- `rgb.avi` — RGB video (MJPG codec)
- `depth/frame_*.npy` — depth filtered per frame
- `depth_raw/frame_*.npy` — depth unfiltered per frame
- `metadata.json` — width, height, fps, frame_count, depth_scale, timestamps

### VideoPlaybackThread

`QThread` yang membaca folder recording dan menjalankan frame pair (RGB+depth) melalui pipeline 5-stage penuh. Sinyal yang dipancarkan identik dengan `CameraThread` — sehingga MainWindow bisa menukar keduanya tanpa mengubah wiring widget.

**Kontrol playback (public API):**

| Method | Fungsi |
|---|---|
| `start_playback(recording_dir)` | Mulai playback dari folder recording |
| `stop_playback()` | Hentikan playback |
| `set_paused(bool)` | Pause / resume |
| `toggle_pause()` | Toggle pause state |
| `set_speed(multiplier)` | Set kecepatan (0.25–4.0x, di-clamp) |
| `set_loop(bool)` | Aktifkan/nonaktifkan loop otomatis |
| `is_paused` (property) | Status pause saat ini |
| `set_depth_thresholds(min_m, max_m)` | Update threshold depth (compat dengan CameraThread) |

**Format depth yang didukung:**
- Stacked `.npy` (cepat, preferensi): `depth.npy` shape `(N, H, W)` uint16
- Individual `.npy` (legacy): `depth/frame_00000.npy`, `depth/frame_00001.npy`, ...
- RGB-only: tanpa file depth → stage depth di-skip otomatis

### Input Source switcher (Live Camera ↔ Video File)

`ControlsPanel` (GUI) menyediakan toggle **Live Camera** ↔ **Video File**. Saat user memilih Video File, `QFileDialog` terbuka dan `MainWindow`:
1. Menghentikan `CameraThread` (jika sedang jalan)
2. Membuat `VideoPlaybackThread` baru
3. Me-reconnect sinyal identik ke slot GUI yang sama
4. Memulai playback dari folder recording

Karena kontrak sinyal identik, semua widget (DepthView, RadarView, AlertPanel) bekerja tanpa modifikasi. Untuk kembali ke live, user memilih Live Camera dan `CameraThread` di-restart.

## Catatan Pengembangan

- Pipeline 5-stage: DepthProcessingStage (LUT colormap), YOLODetectionStage (dual-model + CLAHE + hysteresis), FusionStage (two-pass), NavigationStage (VFH-lite gap-based steering), VisualAnnotationStage (HUD on RGB + depth).
- Dual-model: RGB model untuk kondisi terang, depth model untuk kondisi gelap (lazy-loaded).
- Tiga mode aktif (tracked di `active_model`): `rgb` (terang), `rgb_clahe` (dim, CLAHE-enhanced), `depth` (gelap). `none` saat model belum di-load.
- Dark mode hysteresis: enter dark at brightness < 35, exit at > 50 (prevents flicker).
- Auto-switch view: GUI automatically switches RGB/Depth view based on is_dark signal.
- FP16 inference aktif saat CUDA tersedia (~2x faster on Tensor Cores).
- Model validation guard: `YOLOWrapper` menolak weight yang file-nya hilang/korup, task-nya bukan detect/segment (mis. classify), atau class set-nya beda dari `mobil/motor/person` — mencegah silent failure (deteksi kosong / label desync di FusionStage & AlertPanel).
- LUT depth colormap ~3x lebih cepat dari mask approach.
- ObstacleDetector tidak mengcopy color frame (performance optimization).
