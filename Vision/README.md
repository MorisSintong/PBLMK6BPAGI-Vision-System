# Vision Module

Dokumentasi ini menjelaskan komponen pada folder `Vision/` untuk akuisisi frame dan pemrosesan vision.

## Tujuan Modul

Modul Vision bertanggung jawab untuk:
- mengambil frame dari kamera (RealSense / webcam),
- menghasilkan data RGB dan Depth untuk GUI,
- menyediakan fondasi pemrosesan objek/obstacle.

## Struktur Folder

- `src/` — logic akuisisi kamera dan pemrosesan
- `inc/` — konfigurasi parameter vision

## Komponen Utama (`src`)

| File | Fungsi |
|---|---|
| `camera_thread.py` | Worker thread utama untuk capture kamera. Memiliki Delta Sleep optimizer untuk performa maksimal. Mengirim frame yang *memory-safe* ke GUI via sinyal Qt. |
| `frame_processor.py` | Engine utama pipeline vision. Mengimplementasi *Chain of Responsibility* (YOLO, Depth, Fusion) yang berjalan pada setiap frame dengan pelacakan error robust. |
| `yolowrapper.py` | Memuat model YOLOv8 dan melakukan inference object detection. Mendukung optimasi FP16 di GPU. |
| `obstacle_detector.py` | Modul yang mengekstrak informasi jarak dan prioritas menggunakan HUD visual overlay premium. |
| `recorder.py` | Utilitas uji/rekam stream RealSense secara mandiri. Memiliki flag *mutex* agar tidak crash dengan pipeline utama. |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `detection_config.py` | Parameter threshold deteksi (min/max/danger distance). |
| `camera_config.py` | Placeholder konfigurasi kamera tambahan. |

## Alur Singkat Kamera Saat Ini

1. `CameraThread.start_capture()` dipanggil dari GUI.
2. Thread mengambil frame dari hardware (RealSense atau OpenCV fallback).
3. Frame diproses oleh `FrameProcessor` melalui rangkaian *PipelineStage* (mis: Deteksi YOLO, Ekstraksi Depth).
4. Hasil komputasi dan draw layer dikemas dalam objek `FrameData`.
5. Frame hasil dikonversi ke format `QImage` (menggunakan byte copying untuk thread safety) dan dipancarkan ke GUI.
6. `CameraThread` melakukan Delta Sleep untuk mempertahankan stabil 30 FPS tanpa menyebabkan lonjakan CPU usage.

## Catatan Pengembangan

- Untuk fitur D455 lanjutan (filter depth, point cloud, calibration), gunakan API `pyrealsense2` di layer ini.
- `obstacle_detector.py` sudah berisi dasar deteksi, tetapi integrasinya ke alur GUI real-time masih bisa dikembangkan lebih lanjut.
