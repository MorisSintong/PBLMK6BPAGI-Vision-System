# YOLOv8 Model Evaluation Report (V4.2 & V4 Depth)

**Author**: Hamid (Role 5 - Dataset, Testing & Performance Engineer)  
**Date**: 25 Juni 2026  
**Hardware Target**: AMD Ryzen 5 6600H CPU  

---

## 1. Executive Summary

Pengujian komprehensif telah dilakukan terhadap dua pasang model YOLOv8 (RGB dan Depth) untuk memvalidasi **Akurasi (mAP@0.5)** dan **Latensi (Speed P95)** sesuai dengan standar KPI proyek (Role 5). 

Hasil evaluasi menunjukkan bahwa iterasi model terbaru (**`ModelRGB_V4.2.pt`** dan **`ModelDepth_V4.pt`**) berhasil melampaui target akurasi dengan sangat memuaskan. Namun, model terbaru gagal memenuhi *budget* latensi real-time sebesar $\le$ 100ms karena beralih dari arsitektur *Object Detection* murni menjadi *Instance Segmentation*.

## 2. Evaluation Metrics & Targets
- **Target Akurasi (Detection):** `mAP@0.5` $\ge$ 70.0%
- **Target Latensi (End-to-End):** `P95 Latency` $\le$ 100.0 ms

---

## 3. Results: RGB Models Comparison

Pengujian dilakukan menggunakan `DatasetRGB_V4.2-2668 Frame` (split *test*).

| Metric | ModelRGB_V4_Beta.pt (Detection) | ModelRGB_V4.2.pt (Segmentation) | Status V4.2 |
|---|---|---|---|
| **mAP@0.5** | 96.35% | **98.37%** | ✅ Passed |
| **mAP@0.5:0.95** | 84.51% | **90.82%** | ✅ Passed |
| **Avg Latency** | 129.46 ms (7.7 FPS) | **196.15 ms (5.1 FPS)** | ❌ Failed |
| **P50 Latency** | 127.87 ms | **192.85 ms** | ❌ Failed |
| **P95 Latency** | 135.93 ms | **214.46 ms** | ❌ Failed (Critical) |

> [!CAUTION]
> Latensi P95 untuk model RGB V4.2 memuncak pada 214 ms per frame. Hal ini sangat berbahaya jika diimplementasikan langsung untuk kontrol navigasi robot (*obstacle avoidance*), karena robot tidak akan merespon tepat waktu saat menemui rintangan mendadak.

---

## 4. Results: Depth Models Comparison

Pengujian dilakukan menggunakan `DatasetDepth_V4-2471 Frame` (split *test*).

| Metric | ModelDepth.pt (Detection) | ModelDepth_V4.pt (Segmentation) | Status V4 |
|---|---|---|---|
| **mAP@0.5** | 85.50% | **87.23%** | ✅ Passed |
| **mAP@0.5:0.95** | 63.52% | **67.10%** | ✅ Passed |
| **Avg Latency** | **68.40 ms** (14.6 FPS) | 196.29 ms (5.1 FPS) | ❌ Failed |
| **P50 Latency** | **66.14 ms** | 194.42 ms | ❌ Failed |
| **P95 Latency** | **80.86 ms** | 207.65 ms | ❌ Failed (Critical) |

> [!WARNING]
> Kecepatan *inference depth* merosot tajam (dari 80ms menjadi 207ms) akibat beban perhitungan tambahan dari algoritma *Masking/Segmentation* pada model V4.

---

## 5. Temuan Utama & Rekomendasi

### Temuan
1. **Peningkatan Akurasi Signifikan:** Transisi dari model algoritma *Bounding Box* (Detection) ke *Pixel Masking* (Segmentation) membuat prediksi bentuk objek menjadi jauh lebih presisi.
2. **Bottleneck Komputasi CPU:** Kompleksitas *Segmentation* melipatgandakan waktu pemrosesan di CPU, menyebabkan performa jatuh menjadi rata-rata $\sim$ 5 FPS.

### Action Items (Rekomendasi untuk Tim)
1. **Fallback (Rollback) Sementara:** 
   Untuk saat ini, `ModelRGB_V4_Beta.pt` dan `ModelDepth.pt` harus ditetapkan sebagai model *default* (*production*) agar keselamatan robot terjaga dan sistem *Sensor Fusion* (oleh Role 4) tidak *delay*.
   
2. **Tugas Optimasi untuk Role 2 (Husein):**
   Husein wajib melakukan optimasi *Inference Engine* pada model *Segmentation* tersebut sebelum bisa digunakan. Beberapa langkah yang disarankan:
   - Mengekspor model `.pt` ke format `.onnx` atau **OpenVINO** untuk mempercepat pemrosesan CPU.
   - Mengurangi *input size* resolusi YOLO dari bawaan menjadi lebih kecil (`416x416` atau `320x320`).
   
3. **Deployment Masa Depan:**
   Jika pada akhirnya robot dibekali komputer edge dengan kemampuan GPU (*NVIDIA Jetson*), model V4.2 dan V4 ini bisa langsung dipasang dengan format **TensorRT** untuk mengembalikan FPS ke angka normal.

---
**Report Generated via Automated Test Harness**  
*Script: `01_Scripts/benchmark_yolo.py`*
