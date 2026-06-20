# GUI Module

Dokumentasi ini menjelaskan struktur dan tanggung jawab komponen pada folder `GUI/`.

## Tujuan Modul

Modul GUI bertanggung jawab untuk:
- menampilkan stream kamera (RGB/Depth/Overlay),
- menerima interaksi operator (start/stop kamera, pilih mode tampilan),
- menampilkan informasi status/alert ke operator.

## Struktur Folder

- `src/` — widget dan logic utama GUI
- `inc/` — konstanta UI dan style pendukung

## Komponen Utama (`src`)

| File | Fungsi |
|---|---|
| `main_window.py` | Menyusun layout utama, menghubungkan sinyal *type-safe* antar-panel, dan me-routing konfigurasi GUI ke core pipeline di `CameraThread`. |
| `depth_view.py` | Area display dengan 3 mode: RGB, Depth, dan Overlay. Menangani empty fallback frames `QImage.isNull()` dengan stabil. |
| `controls_panel.py` | Kontrol kamera utama dan pengaturan jarak alert dinamis (mengirim sinyal threshold langsung ke Vision pipeline). |
| `Alert_panel.py` | Menampilkan info objek/jarak serta perubahan warna sesuai status threshold (DANGER/WARN/SAFE). |
| `radar_view.py` | Widget radar visual yang terhubung ke data resolusi spatial via `obstacles_ready` signal. |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `ui_config.py` | Konstanta UI global (nama app, ukuran minimum window, threshold default, warna). |
| `styles.py` | Tempat stylesheet global (opsional, tergantung implementasi). |

## Alur Singkat GUI

1. User berinteraksi dengan **Start/Stop** atau **Threshold Sliders** di `ControlsPanel`.
2. `main_window.py` menghubungkan input ini dan mengirimkannya ke `CameraThread` dan `FrameProcessor`.
3. Frame *memory-safe* (`QImage`) dan notifikasi status dari Vision pipeline dikirim melalui emit signal.
4. `DepthView` memeriksa integritas *image buffer* dan me-render visual overlay (seperti HUD bounding box) ke layar.
5. `RadarView` dan `AlertPanel` memperbarui UI secara real-time dari data spasial obstacle.
