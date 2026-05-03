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
| `camera_thread.py` | Worker thread utama untuk capture kamera. Prioritas RealSense (`pyrealsense2`), fallback ke OpenCV webcam. Mengirim frame ke GUI via sinyal Qt. |
| `recorder.py` | Utilitas uji/rekam stream RealSense (color + depth) secara mandiri. |
| `obstacle_detector.py` | Kelas deteksi obstacle berbasis ROI dan threshold depth (siap dipakai/integrasi lanjutan). |
| `frame_processor.py` | Placeholder untuk pipeline pemrosesan frame lanjutan. |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `detection_config.py` | Parameter threshold deteksi (min/max/danger distance). |
| `camera_config.py` | Placeholder konfigurasi kamera tambahan. |

## Alur Singkat Kamera Saat Ini

1. `CameraThread.start_capture()` dipanggil dari GUI.
2. Thread mencoba inisialisasi RealSense (color + depth stream).
3. Jika gagal, thread fallback ke webcam OpenCV.
4. Frame dikonversi ke `QPixmap` lalu dipancarkan ke GUI (`frame_pair_ready`).
5. `DepthView` di GUI menampilkan frame sesuai mode yang dipilih user.

## Catatan Pengembangan

- Untuk fitur D455 lanjutan (filter depth, point cloud, calibration), gunakan API `pyrealsense2` di layer ini.
- `obstacle_detector.py` sudah berisi dasar deteksi, tetapi integrasinya ke alur GUI real-time masih bisa dikembangkan lebih lanjut.
