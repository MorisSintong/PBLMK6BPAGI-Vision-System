# GUI Module

Dokumentasi ini menjelaskan struktur dan tanggung jawab komponen pada folder `GUI/`.

## Tujuan Modul

Modul GUI bertanggung jawab untuk:
- Menampilkan stream kamera (RGB/Depth/Overlay)
- Menerima interaksi operator (start/stop kamera, pilih mode tampilan)
- Menampilkan informasi status/alert ke operator
- Merender radar 180° dengan posisi obstacle real-time

## Struktur Folder

- `src/` — widget dan logic utama GUI
- `inc/` — konstanta UI dan style pendukung

## Komponen Utama (`src`)

| File | Fungsi |
|---|---|
| `main_window.py` | Menyusun layout utama, menghubungkan sinyal type-safe antar-panel, me-routing konfigurasi GUI ke core pipeline, dan mengassembl pipeline 4-stage. |
| `depth_view.py` | Area display dengan 3 mode: RGB, Depth, dan Overlay. `setScaledContents` di-set sekali di init. Hanya update label untuk page yang sedang visible. Menangani empty fallback frames via `QImage.isNull()`. |
| `controls_panel.py` | Kontrol kamera utama dan pengaturan jarak alert dinamis (mengirim sinyal threshold langsung ke Vision pipeline). |
| `alert_panel.py` | Menampilkan info objek/jarak serta perubahan warna sesuai status threshold (DANGER/WARN/SAFE). Stylesheet hanya di-update saat status berubah (pre-computed style dicts). |
| `radar_view.py` | Widget radar 180° dengan cached static background pixmap. Hanya sweep line dan obstacle blips yang di-redraw per frame. Terhubung ke data via `obstacles_ready` signal. |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `ui_config.py` | Konstanta UI global (nama app, ukuran minimum window, threshold default, radar dimensions, zone labels, action labels). |
| `styles.py` | Global stylesheet + color constants (status colors, radar colors, infobox styles, text colors). |

## Alur Singkat GUI

1. User berinteraksi dengan **Start/Stop** atau **Threshold Sliders** di `ControlsPanel`.
2. `main_window.py` menghubungkan input ini dan mengirimkannya ke `CameraThread` dan `FrameProcessor`.
3. Frame *memory-safe* (`QImage`) dan notifikasi status dari Vision pipeline dikirim melalui emit signal.
4. `DepthView` memeriksa integritas image buffer dan me-render visual overlay (HUD bounding box) ke layar. Hanya label untuk page visible yang di-update.
5. `RadarView` dan `AlertPanel` memperbarui UI secara real-time dari data spasial obstacle. Radar menggunakan cached background, AlertPanel hanya update stylesheet saat status berubah.

## Performance Optimizations

| Optimization | File | Impact |
|---|---|---|
| Cached static background pixmap | `radar_view.py` | ~80% less paint work (rings, labels, FOV lines pre-rendered) |
| Status-change-only stylesheets | `alert_panel.py` | 6→0 style recalcs/frame in steady state |
| setScaledContents once in init | `depth_view.py` | 6 fewer layout passes/frame |
| Visible-only label updates | `depth_view.py` | 2 fewer pixmap sets/frame |
