# Panduan Pengumpulan Data — PBL Vision System

**Target audiens:** R5 (Hamid) dan asisten AI yang mengerjakan akuisisi dataset.

---

## 1. Tujuan

Mengumpulkan dataset gambar RGB berlabel untuk training YOLOv8 guna mendeteksi objek yang relevan dengan robot keamanan:
- **person** (orang)
- **motor** (motor/motorcycle)
- **mobil** (mobil/car)

Kamera depth (Intel RealSense D455) menangkap **pasangan RGB + Depth yang sinkron**, tetapi training YOLO hanya menggunakan **frame RGB**. Frame depth disimpan secara opsional untuk filtering kualitas dan model 3D di masa depan.

**Persyaratan minimum:** ≥300 frame berlabel di semua class.

**Target performa:** mAP@0.5 ≥ 70% pada validation set.

---

## 2. Setup Hardware

### 2.1 Peralatan yang Diperlukan

| Item | Catatan |
|---|---|
| Intel RealSense D455 | Kamera utama, menyediakan RGB + Depth |
| Laptop/PC dengan GPU | NVIDIA RTX direkomendasikan, untuk training |
| Tripod atau dudukan stabil | Mengurangi motion blur |
| Robot itu sendiri | Untuk tinggi/sudut kamera yang realistis |

### 2.2 Konfigurasi Kamera

```
Resolution:   640 × 480 (RGB + Depth)
Frame rate:   30 FPS
Depth format: Z16 (16-bit unsigned, millimeters)
Color format: BGR8
```

**Tinggi kamera:** Pasang RealSense pada tinggi operasi robot yang diharapkan (~0.5–1.0m dari tanah). Ini memastikan data training sesuai dengan kondisi deployment.

### 2.3 Kondisi Lingkungan

Anda HARUS mengumpulkan data dalam **semua** kondisi pencahayaan berikut:

| Kondisi | Deskripsi | Alasan |
|---|---|---|
| Terang siang hari | Outdoor, sinar matahari langsung | Menguji degradasi sensor depth (saturasi IR) |
| Berawan / teduh | Outdoor, tanpa matahari langsung | Depth bagus, RGB moderat |
| Lampu fluorescent indoor | Pencahayaan kantor / lorong | Kondisi indoor standar |
| Redup indoor | Cahaya rendah, sedikit lampu | Menguji degradasi YOLO |
| Malam hari outdoor | Gelap, hanya lampu jalan | Kasus terburuk untuk YOLO |
| Malam hari indoor | Ruangan gelap, cahaya minimal | Kasus terburuk untuk YOLO |

**JANGAN mengumpulkan hanya dalam pencahayaan yang baik.** Intinya adalah membuat YOLO robust dalam kondisi buruk.

---

## 3. Prosedur Pengumpulan Data

### 3.1 Langkah demi Langkah

1. **Nyalakan kamera RealSense** (melalui aplikasi Vision System atau RS SDK)
2. **Posisikan objek target** (orang, motor, atau mobil) di dalam scene
3. **Tangkap frame** pada jarak berikut untuk setiap objek:

| Jarak | Jumlah per class |
|---|---|
| 0.5m – 1.0m (dekat) | ≥ 20 frame |
| 1.0m – 2.0m (sedang) | ≥ 20 frame |
| 2.0m – 4.0m (jauh) | ≥ 20 frame |
| > 4.0m (latar belakang) | ≥ 10 frame (sebagai negatif) |

4. **Variasikan posisi:** Pindahkan objek (atau kamera) untuk mencakup semua zona:
   - Sisi kiri frame
   - Tengah frame
   - Sisi kanan frame
5. **Variasikan sudut:** Tangkap dari orientasi berbeda (depan, samping, belakang)
6. **Tangkap frame "negatif":** Scene TANPA objek target (ruangan kosong, lorong, parkiran) — ini mengajari YOLO apa yang TIDAK boleh dideteksi

### 3.2 Aturan Pemilihan Frame

**SIMPAN frame yang:**
- Tajam (tanpa motion blur)
- Komposisi baik (objek terlihat penuh, tidak terpotong)
- Bervariasi (latar belakang, posisi, pencahayaan berbeda)

**JANGAN SIMPAN frame yang:**
- Sepenuhnya hitam (lensa tertutup, gelap total)
- Sepenuhnya putih (overexposed, kerusakan kamera)
- Motion blur (kamera goyang saat pengambilan)
- Duplikat (frame berurutan dari scene statis yang sama — pilih yang terbaik)

### 3.3 Ukuran Dataset Minimum

| Class | Frame Minimum | Direkomendasikan |
|---|---|---|
| person | 100 | 200 |
| motor | 100 | 200 |
| mobil | 100 | 200 |
| Negatif (tanpa objek) | 30 | 50 |
| **Total** | **330** | **650** |

**Kualitas lebih penting daripada kuantitas.** 300 frame berlabel yang tajam dan bervariasi > 1000 frame buram dan duplikat.

---

## 4. Labeling

### 4.1 Format Bounding Box (YOLO)

Setiap frame mendapat file label `.txt` yang sesuai. Format:

```
class_id  x_center  y_center  width  height
```

Semua nilai **ternormalisasi** (0.0 – 1.0) relatif terhadap dimensi gambar.

**Contoh:** Seorang person pada bounding box `[120, 80, 200, 300]` dalam gambar 640×480:

```
x_center = (120 + 200/2) / 640 = 0.25
y_center = (80 + 300/2) / 480 = 0.417
width    = 200 / 640 = 0.3125
height   = 300 / 480 = 0.625
```

File label `frame_0001.txt`:
```
0 0.25 0.417 0.3125 0.625
```

### 4.2 Pemetaan Class

| class_id | class_name |
|---|---|
| 0 | mobil |
| 1 | motor |
| 2 | person |

### 4.3 Tools Labeling

Gunakan salah satu tools berikut:

| Tool | Tipe | Direkomendasikan |
|---|---|---|
| [Roboflow](https://roboflow.com) | Berbasis web | ✅ Ya — auto-labeling + ekspor ke format YOLO |
| [CVAT](https://cvat.ai) | Web/self-hosted | ✅ Ya — kelas profesional |
| [LabelImg](https://github.com/heartexlabs/labelImg) | Desktop | OK — sederhana, ringan |
| [Python script + YOLO auto-label](#44-auto-labeling-with-yolo) | Otomatis | ✅ Ya — tercepat untuk dataset besar |

### 4.4 Auto-Labeling dengan YOLO

Gunakan model YOLOv8n yang sudah ter-trained untuk auto-label frame, lalu tinjau oleh manusia:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Pre-trained on COCO (80 classes)

results = model.predict(
    source="frames/",
    imgsz=320,
    conf=0.3,
    save_txt=True,          # Save labels in YOLO format
    project="dataset/",
    name="labels/auto",
)
```

Ini otomatis menghasilkan file label `.txt`. **Tinjauan manusia tetap diperlukan** untuk:
- Memperbaiki objek yang salah klasifikasi (mis., "motorcycle" → "motor")
- Menghapus false positive
- Menambahkan deteksi yang terlewat
- Memverifikasi pemetaan class sesuai 3 class kita

### 4.5 Checklist Kualitas Labeling

Sebelum menyelesaikan dataset, verifikasi:

- [ ] Setiap frame memiliki file label `.txt` yang cocok
- [ ] Tidak ada file label kosong (setiap file minimal memiliki satu anotasi)
- [ ] Bounding box pas di sekitar objek (tidak terlalu longgar, tidak terlalu ketat)
- [ ] Class ID sesuai pemetaan (0=mobil, 1=motor, 2=person)
- [ ] Tidak ada label duplikat pada objek yang sama
- [ ] Objek yang terhalang diberi label (meskipun hanya sebagian terlihat)
- [ ] Objek di tepi frame diberi label (meskipun sebagian terpotong)

---

## 5. Organisasi Dataset

### 5.1 Struktur Folder

```
dataset/
├── data.yaml                    # Dataset config (see below)
├── images/
│   ├── train/                   # 80% of frames
│   │   ├── frame_0001.png
│   │   ├── frame_0002.png
│   │   └── ...
│   └── val/                     # 20% of frames
│       ├── frame_0051.png
│       └── ...
├── labels/
│   ├── train/                   # Matching label files
│   │   ├── frame_0001.txt
│   │   ├── frame_0002.txt
│   │   └── ...
│   └── val/
│       ├── frame_0051.txt
│       └── ...
└── depth/                       # Optional: raw depth data
    ├── frame_0001.npy
    └── ...
```

### 5.2 Split Train/Val

- **80% train** — digunakan untuk training model
- **20% val** — digunakan untuk evaluasi selama training (data yang belum dilihat)

**Penting:** Split berdasarkan **scene**, bukan per frame. Jika Anda menangkap 50 frame orang yang sama di lokasi yang sama, letakkan 40 di train dan 10 di val. JANGAN letakkan 50 di train dan 0 di val — ini menyebabkan data leakage (model menghafal scene, tidak generalisasi).

### 5.3 Konfigurasi `data.yaml`

```yaml
# Dataset config for YOLOv8 training
path: dataset/          # Dataset root relative to project
train: images/train/    # Train images
val: images/val/        # Val images

# Classes
names:
  0: mobil
  1: motor
  2: person

nc: 3  # Number of classes
```

---

## 6. Data Depth (Opsional tapi Direkomendasikan)

### 6.1 Mengapa Menyimpan Depth?

1. **Filtering kualitas:** Periksa apakah objek berada pada jarak yang valid
2. **Model masa depan:** Training deteksi yang sadar depth (3D bounding box)
3. **Validasi:** Verifikasi akurasi bounding box menggunakan konsistensi depth

### 6.2 Cara Menyimpan Depth

Simpan depth sebagai file `.npy` (NumPy array):

```python
import numpy as np

# During capture
depth_raw = np.asanyarray(depth_frame.get_data())  # uint16, millimeters
np.save(f"dataset/depth/{frame_name}.npy", depth_raw)
```

### 6.3 Pemeriksaan Kualitas Depth

Sebelum menyimpan frame, verifikasi kualitas depth:

```python
valid_pixels = np.count_nonzero(depth_raw)
total_pixels = depth_raw.size
valid_ratio = valid_pixels / total_pixels

if valid_ratio < 0.3:
    # Less than 30% valid depth pixels — bad frame, skip
    continue
```

---

## 7. Training

### 7.1 Perintah Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Start from pre-trained weights

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=320,
    batch=16,
    device=0,                # GPU (use "cpu" if no GPU)
    project="runs/train",
    name="security_robot_v1",
    patience=20,             # Early stopping if no improvement
    save=True,
    plots=True,              # Generate training plots
)
```

### 7.2 Target Training

| Metrik | Target | Cara cek |
|---|---|---|
| mAP@0.5 | ≥ 70% | `results.csv` → `metrics/mAP50(B)` |
| Precision | ≥ 60% | `results.csv` → `metrics/precision(B)` |
| Recall | ≥ 60% | `results.csv` → `metrics/recall(B)` |
| Training loss | Menurun | `results.csv` → `train/box_loss` |

### 7.3 Tanda Overfitting

Jika val loss meningkat sementara train loss menurun → **overfitting**:
- Model menghafal data training, tidak generalisasi
- **Solusi:** Tambah data, data augmentation, kurangi epochs, gunakan early stopping

### 7.4 Tanda Underfitting

Jika train dan val loss keduanya tinggi → **underfitting**:
- Model belum belajar cukup
- **Solusi:** Tambah epochs, model lebih besar (yolov8s/m), tambah data

---

## 8. Evaluasi

### 8.1 Jalankan Validation

```python
from ultralytics import YOLO

model = YOLO("runs/train/security_robot_v1/weights/best.pt")

metrics = model.val(
    data="dataset/data.yaml",
    imgsz=320,
    device=0,
)

print(f"mAP@0.5:    {metrics.box.map50:.2%}")
print(f"mAP@0.5:0.95: {metrics.box.map:.2%}")
print(f"Precision:  {metrics.box.mp:.2%}")
print(f"Recall:     {metrics.box.mr:.2%}")
```

### 8.2 Performa Per-Class

Periksa setiap class secara individual:

```python
for i, class_name in enumerate(["mobil", "motor", "person"]):
    print(f"{class_name}: AP@0.5 = {metrics.box.ap50[i]:.2%}")
```

Jika satu class jauh lebih buruk, kumpulkan lebih banyak data untuk class tersebut.

### 8.3 Confusion Matrix

```python
model.val(data="dataset/data.yaml", plots=True, imgsz=320)
# Generates confusion_matrix.png in runs/val/
```

Perhatikan:
- **False positive:** Model mendeteksi objek padahal tidak ada
- **False negative:** Model melewatkan objek nyata
- **Class confusion:** Model bingung antara person dengan motor, dll.

---

## 9. Kesalahan Umum

| Kesalahan | Mengapa buruk | Cara menghindari |
|---|---|---|
| Semua frame dari lokasi yang sama | Model tidak generalisasi ke environment baru | Kumpulkan dari 5+ lokasi berbeda |
| Semua frame dalam pencahayaan baik | Model gagal di malam hari | Kumpulkan dalam SEMUA kondisi pencahayaan |
| Hanya objek dari tampilan depan | Model tidak bisa mendeteksi tampilan samping/belakang | Putar objek atau kamera |
| Frame berurutan yang duplikat | Pemborosan tenaga, menggembungkan ukuran dataset | Pilih frame terbaik per scene, lewati duplikat |
| Bounding box longgar | Model belajar batas objek yang salah | Label pas di sekitar objek yang terlihat |
| Class imbalance (200 person, 20 motor) | Model bias ke class mayoritas | Seimbangkan: frame per class kira-kira sama |
| Tidak ada frame negatif | Model mendeteksi objek di mana-mana | Sertakan 10% scene kosong |

---

## 10. Checklist Deliverable

Sebelum diserahkan ke R2 (Husein) untuk training:

- [ ] ≥300 frame RGB berlabel (direkomendasikan: 650+)
- [ ] Seimbang antar 3 class (±20% per class)
- [ ] Seimbang antar kondisi pencahayaan (siang/malam/indoor/outdoor)
- [ ] Seimbang antar jarak (0.5–1m, 1–2m, 2–4m)
- [ ] Split train/val 80/20 (per scene, bukan per frame)
- [ ] File label `.txt` format YOLO
- [ ] File config `data.yaml`
- [ ] Validation mAP@0.5 ≥ 70%
- [ ] Confusion matrix ditinjau untuk class confusion
- [ ] Frame depth disimpan (opsional tapi direkomendasikan)

---

## 11. Referensi: Template Script Pengumpulan Data

```python
"""
Data collection script for RealSense D455.
Captures synchronized RGB + Depth frames.
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# Config
OUTPUT_DIR = "dataset/raw"
FRAME_INTERVAL = 0.5  # Seconds between captures (avoid duplicates)
MIN_DEPTH_VALID_RATIO = 0.3  # Minimum valid depth pixels to save

os.makedirs(f"{OUTPUT_DIR}/rgb", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/depth", exist_ok=True)

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frame_count = 0
last_capture_time = 0

print("Data Collection Mode")
print("Press SPACE to capture, ESC to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display
        cv2.imshow("RGB", color_image)

        # Depth colormap for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = capture
        if key == 32:
            # Check depth quality
            valid_ratio = np.count_nonzero(depth_image) / depth_image.size
            if valid_ratio < MIN_DEPTH_VALID_RATIO:
                print(f"Skipped: depth quality too low ({valid_ratio:.1%})")
                continue

            # Check interval
            current_time = time.time()
            if current_time - last_capture_time < FRAME_INTERVAL:
                print("Skipped: too fast, wait longer")
                continue

            # Save
            frame_name = f"frame_{frame_count:04d}"
            cv2.imwrite(f"{OUTPUT_DIR}/rgb/{frame_name}.png", color_image)
            np.save(f"{OUTPUT_DIR}/depth/{frame_name}.npy", depth_image)

            frame_count += 1
            last_capture_time = current_time
            print(f"Captured: {frame_name} | Total: {frame_count}")

        # ESC = quit
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_count} frames saved to {OUTPUT_DIR}/")
```

**Penggunaan:**
1. Jalankan script: `python collect_data.py`
2. Arahkan kamera ke objek target
3. Tekan SPACE untuk menangkap (satu frame setiap 0.5s)
4. Tekan ESC untuk keluar
5. Label frame yang ditangkap dengan Roboflow/CVAT/LabelImg
6. Ekspor dalam format YOLO
7. Split menjadi train/val (80/20)
8. Training YOLOv8
9. Evaluasi dan iterasi
