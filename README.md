# PBLMK6BPAGI-Vision-System

Modul vision untuk security robot berbasis **PyQt6 + OpenCV + Intel RealSense (D455)**.
Fokus utama project ini adalah aplikasi desktop untuk menampilkan stream kamera, memantau jarak objek, dan mengontrol mode tampilan secara real-time.

## Overview

Secara umum aplikasi ini melakukan:
- Akuisisi stream kamera (RealSense sebagai prioritas, webcam sebagai fallback)
- Menampilkan mode **RGB**, **Depth**, dan **Overlay (RGB + Depth berdampingan)**
- Kontrol kamera dari panel GUI (start/stop + pemilihan mode tampilan)
- Menampilkan informasi alert berdasarkan jarak objek

## Struktur Project

- `main.py` — entry point aplikasi Qt
- `GUI/` — komponen antarmuka pengguna  
  Lihat detail: **[`GUI/README.md`](GUI/README.md)**
- `Vision/` — komponen akuisisi frame dan pemrosesan vision  
  Lihat detail: **[`Vision/README.md`](Vision/README.md)**
- `environment.yml` — daftar dependency conda/pip

## Requirements

- Python 3.10
- Conda (disarankan)
- Intel RealSense D455 

## Setup

```bash
conda env create -f environment.yml
conda activate depth-obstacle-detector
```

## Menjalankan Aplikasi

```bash
python main.py
```

## Catatan

- Dukungan D455 menggunakan `pyrealsense2`.
- Jika RealSense tidak tersedia, aplikasi memakai webcam biasa (RGB).
- Pada Windows, capture kamera memprioritaskan backend DirectShow untuk stabilitas.

