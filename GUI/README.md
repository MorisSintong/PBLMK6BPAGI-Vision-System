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
| `main_window.py` | Menyusun layout utama, menghubungkan sinyal antar-panel, dan mengelola `CameraThread`. |
| `depth_view.py` | Area display kamera dengan 3 mode: RGB, Depth, dan Overlay (side-by-side). |
| `controls_panel.py` | Tombol Start/Stop kamera dan tombol pemilihan mode tampilan. |
| `Alert_panel.py` | Menampilkan info objek/jarak serta perubahan warna sesuai threshold. |
| `radar_view.py` | Widget radar visual (komponen visual terpisah, tidak selalu dipakai di layout utama saat ini). |

## Konfigurasi (`inc`)

| File | Fungsi |
|---|---|
| `ui_config.py` | Konstanta UI global (nama app, ukuran minimum window, threshold default, warna). |
| `styles.py` | Tempat stylesheet global (opsional, tergantung implementasi). |

## Alur Singkat GUI

1. User klik **Start** di `ControlsPanel`.
2. `main_window.py` menerima sinyal dan menjalankan `CameraThread`.
3. Frame RGB/Depth dari `CameraThread` dikirim ke `DepthView.update_frames(...)`.
4. User bisa ganti mode display ke RGB/Depth/Overlay dari tombol mode.
5. Jika ada error kamera, status di panel kontrol diperbarui.
