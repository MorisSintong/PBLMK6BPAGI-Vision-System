# Arsitektur Sistem Vision & Alur Data

Dokumen ini menyajikan rincian komprehensif langkah demi langkah tentang bagaimana data bergerak melalui sistem—dari saat cahaya mengenai sensor kamera hingga saat GUI merender sebuah peringatan di layar. Termasuk semua rumus matematis, teori pemrosesan sinyal, dan keputusan desain.

Sistem dibangun di atas arsitektur terpisah (decoupled) menggunakan **PyQt6 Signals** untuk komunikasi thread-safe dan pola **Chain of Responsibility** untuk pemrosesan vision.

---

## 1. Fase Inisialisasi

Saat program dimulai (`main.py`), ia menginstansiasi GUI (`MainWindow`). GUI sepenuhnya pasif hingga user menekan tombol **Start Camera**.

1. **GUI Setup:** `MainWindow` menginisialisasi `DepthView` (untuk gambar), `ControlsPanel` (untuk tombol/slider), `AlertPanel` (untuk status teks), dan `RadarView` (untuk pelacakan spasial).
2. **Vision Setup:** `MainWindow` menginstansiasi `FrameProcessor` (otak sistem vision) dan `CameraThread` (sebuah PyQt `QThread` latar yang berkomunikasi dengan hardware).
3. **Pipeline Assembly:** `FrameProcessor` dikonfigurasi dengan lima stage secara berurutan:
   - `DepthProcessingStage` — selalu hadir (R3)
   - `YOLODetectionStage` — dual-model swap dengan CLAHE (R2)
   - `FusionStage` — menggabungkan output R2 + R3 (R4)
   - `NavigationStage` — steering berbasis gap via polar histogram (R1)
   - `VisualAnnotationStage` — rendering HUD (R1)
4. **Signal Routing:** GUI menghubungkan signal output thread (`frame_pair_ready`, `distance_info_ready`, `obstacles_ready`, `navigation_ready`, `light_mode_changed`, `error`) ke fungsi update miliknya sendiri.
5. **GPU Warm-up:** Jika CUDA tersedia, `YOLOWrapper` menjalankan inference dummy saat loading untuk pre-compile kernel CUDA. Ini mencegah frame pertama yang sebenarnya menjadi lambat.

---

## 2. Akuisisi Hardware (`CameraThread`)

Setelah user menekan **Start**, `CameraThread` memasuki loop-nya. Capture kamera berjalan dalam **acquisition thread terpisah**, terpisah dari loop pemrosesan via sebuah `queue.Queue(maxsize=2)`.

### 2.1 Teori Depth Sensing RealSense

Intel RealSense D455 menggunakan **stereo infrared depth sensing**. Dua sensor IR menangkap scene yang sama dari perspektif yang sedikit berbeda. Hardware menghitung **disparity** — pergeseran pixel antara titik-titik yang sesuai pada dua gambar.

**Depth dari Disparity:**

```
depth (meters) = baseline × focal_length / disparity
```

Di mana:
- `baseline` = jarak antara dua sensor IR (~95mm untuk D455)
- `focal_length` = focal length lensa dalam pixel
- `disparity` = pergeseran pixel antara gambar IR kiri dan kanan

Frame depth mentah adalah array unsigned integer 16-bit (`z16` format) di mana setiap nilai merepresentasikan depth dalam **milimeter**. Nilai `0` berarti "tidak ada data" (mis., terlalu jauh, permukaan reflektif, atau di luar jangkauan IR).

### 2.2 Capture Depth Tanpa Filter

Acquisition thread menangkap **unfiltered depth frame** sebelum filter RealSense SDK diterapkan. Depth mentah ini dipertahankan karena model depth R2 (`ModelDepth_V4.pt`) dilatih pada colormap depth tanpa filter. Depth yang difilter digunakan untuk display dan deteksi obstacle.

```
depth_raw_unfiltered = np.asanyarray(depth_frame.get_data())  # SEBELUM filter
# ... terapkan filter ke depth_frame ...
depth_raw_filtered = np.asanyarray(depth_frame.get_data())    # SESUDAH filter
```

Keduanya dilewatkan melalui queue sebagai tuple 3-elemen: `(color_bgr, depth_raw, depth_raw_unfiltered)`.

### 2.3 Filter Hardware

Frame depth mentah dilewatkan melalui filter DSP onboard RealSense untuk men-denoise data 3D:

1. **Decimation Filter** (opsional, dapat dikonfigurasi via `camera_config.py`):
   - Mengurangi resolusi depth dengan subsampling (mis., decimation 2x → 320×240 dari 640×480)
   - Menukar resolusi dengan performa: 4× lebih sedikit pixel untuk diproses
   - Menggunakan interpolasi `INTER_LINEAR` untuk menghindari artefak `INTER_NEAREST`
   - Dinonaktifkan secara default untuk mempertahankan resolusi penuh 640×480

2. **Spatial Filter** (smoothing yang mempertahankan edge):
   - Menerapkan filter mirip-Gaussian yang menghaluskan region datar sambil mempertahankan edge
   - Parameter: `smooth_alpha` (0.5), `smooth_delta` (20)
   - Mengurangi noise "salt-and-pepper" pada depth map

3. **Temporal Filter** (smoothing antar-frame):
   - Menghasilkan rata-rata nilai depth selama beberapa frame dengan exponential decay
   - Mengurangi temporal jitter (nilai depth yang berkedip)
   - Menggunakan alpha blending: `depth_filtered = α × depth_current + (1-α) × depth_previous`

4. **Hole-Filling Filter** (interpolasi):
   - Mengisi celah kecil (holes) pada depth map dengan interpolasi dari pixel valid terdekat
   - Menggunakan sliding window untuk mencari nilai depth valid terdekat

### 2.4 Konversi Numpy

Frame C++ dikonversi menjadi array NumPy Python zero-copy (`color_bgr` dan `depth_raw`). Ini adalah view level pointer dari memory yang sama — tidak terjadi copy data.

### 2.5 Fallback: Mode Webcam

Jika kamera RealSense tidak terhubung, sistem beralih ke `VideoCapture` OpenCV standar (webcam), hanya menyediakan RGB. Pada mode ini, `depth_frame` adalah `None` dan pipeline depth dilewati sepenuhnya.

---

## 3. Pipeline Vision (`FrameProcessor`)

Loop pemrosesan menarik frame dari queue dan menyerahkan array NumPy ke `FrameProcessor`. Processor membungkusnya menjadi objek `FrameData` dan melewatinya melalui **Chain of Responsibility** — setiap stage memproses data dan meneruskannya ke stage berikutnya.

### 3.1 Struktur Data

**`FrameData`** — objek data tunggal yang mengalir melalui pipeline:

```python
@dataclass
class FrameData:
    rgb_frame: np.ndarray              # H×W×3 uint8 BGR
    depth_frame: Optional[np.ndarray]  # H×W uint16 (terfilter, None untuk webcam)
    depth_frame_raw: Optional[np.ndarray]  # H×W uint16 (tanpa filter, untuk model depth)
    depth_colormap: Optional[np.ndarray]   # H×W×3 uint8 (terfilter, untuk display)
    depth_colormap_raw: Optional[np.ndarray]  # H×W×3 uint8 (tanpa filter, untuk model depth)
    depth_scale: float                 # faktor konversi raw → meter
    obstacles: List[Dict]              # dari DepthProcessingStage
    detections: List[Detection]        # dari YOLODetectionStage
    fused_output: List[Dict]           # dari FusionStage
    metadata: Dict[str, Any]           # timestamp, is_dark, rgb_confidence, active_model
    errors: List[str]                  # log error pipeline
```

### 3.2 Stage A: `DepthProcessingStage` (Pemahaman Spasial)

Stage ini mengkonversi data depth mentah menjadi informasi obstacle terstruktur dan memvisualisasikannya sebagai colormap berwarna.

#### 3.2.1 Generasi Colormap Berbasis LUT

Daripada membuat beberapa boolean mask per frame (lambat), stage ini menggunakan **Lookup Table (LUT) pre-computed 256-entry** yang memetakan indeks depth ke warna BGR:

```python
# Build sekali saat init (dan rebuild saat threshold berubah)
lut = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    depth_m = (i / 255.0) * max_distance
    if depth_m < min_distance or depth_m > max_distance:
        lut[i] = (0, 0, 0)        # Hitam = invalid
    elif depth_m < danger_threshold:
        lut[i] = (0, 0, 255)       # Merah = danger
    elif depth_m < warning_threshold:
        lut[i] = (0, 255, 255)     # Kuning = warning
    else:
        lut[i] = (0, 255, 0)       # Hijau = safe

# Per frame: operasi indexing tunggal (~3x lebih cepat dari mask approach)
depth_m = depth_frame.astype(np.float32) * depth_scale
idx = np.clip(depth_m * scale, 0, 255).astype(np.uint8)
colormap = self._depth_lut[idx]
```

**Formula LUT scale:**

```
scale = 255.0 / max_distance
```

Setiap nilai depth dalam meter dikalikan dengan `scale` untuk dipetakan ke indeks 0–255. Misalnya, dengan `max_distance = 5.0m` menghasilkan `scale = 51.0`, sehingga depth 2.5m → indeks 127. Dengan threshold default (danger=1.5m, warning=3.0m), indeks 127 berada di zona kuning (warning). Nilai di luar `[0, max_distance]` di-clip ke 0 atau 255, yang sudah dipetakan ke hitam (invalid) di LUT.

LUT dibangun ulang setiap kali threshold berubah via `set_action_thresholds()` atau `set_thresholds()`.

Dua colormap dihasilkan:
- `depth_colormap` — dari depth terfilter (untuk display)
- `depth_colormap_raw` — dari depth tanpa filter (untuk inference model depth)

#### 3.2.2 Deteksi Obstacle

Stage ini menggunakan `ObstacleDetector` untuk mencari obstacle pada frame depth terfilter. Detector:

1. Mengkonversi depth ke meter menggunakan buffer float32 yang dapat digunakan kembali (menghindari alokasi ~1.2MB per frame)
2. Membuat binary mask: pixel dalam `[min_distance, max_distance]`
3. Menerapkan morphological opening (hapus noise) dan closing (isi holes)
4. Mencari contour dan memfilter berdasarkan area
5. Menghitung jarak menggunakan percentile ke-5 (permukaan terdekat)
6. Mengembalikan obstacle **tanpa copy atau memodifikasi color frame**

**Kenapa percentile ke-5?** Stage ini menggunakan `np.percentile(valid_depth, 5)` — yaitu nilai di mana 5% pixel terdekat berada di bawahnya. Ini memberikan estimasi jarak ke **permukaan terdekat** obstacle (apa yang penting untuk collision avoidance), tetapi lebih robust terhadap noise daripada `np.min()` yang bisa terpengaruh satu pixel outlier.

**Perbandingan percentile di seluruh pipeline:**

| Stage | Percentile | Alasan |
|-------|-----------|--------|
| ObstacleDetector (depth) | 5th | Permukaan terdekat obstacle — paling agresif (safety-critical) |
| FusionStage (direct sampling) | 25th | Stabil terhadap noise edge bbox, tetapi masih merepresentasikan bagian depan objek |
| NavigationStage (polar histogram) | 10th | Kompromi antara min dan median — robust untuk steering per sektor |

Logikanya: obstacle detection butuh konservatif (5th, terdekat), fusion butuh akurat di bbox YOLO (25th, lebih stabil), dan navigation butuh kompromi per sektor (10th, tidak terlalu sensitif terhadap satu pixel noise).

#### 3.2.3 Penempatan Zone

Frame dibagi menjadi 3 zone vertikal yang sama:

```
zone_width = frame_width / 3

zone = {
    "left",    if center_x < zone_width
    "center",  if zone_width ≤ center_x < 2 × zone_width
    "right",   if center_x ≥ 2 × zone_width
}
```

Di mana `center_x = x + w/2` adalah pusat horizontal dari bounding box.

### 3.3 Stage B: `YOLODetectionStage` (Pemahaman Semantik)

Stage ini mendeteksi objek pada frame RGB menggunakan YOLOv8, dengan **dual-model swap** dan **adaptasi dark mode CLAHE**.

#### 3.3.1 Deteksi Dark Mode

Setiap frame dianalisis untuk kecerahan:

```python
brightness = np.mean(data.rgb_frame)
rgb_confidence = min(brightness / 128.0, 1.0)
# Hysteresis: masuk dark saat < 35, keluar saat > 50 (mencegah flicker di sekitar threshold)
if self._is_dark_state:
    is_dark = brightness < 50
else:
    is_dark = brightness < 35
self._is_dark_state = is_dark
```

- `is_dark` — boolean, true saat brightness < 35 (masuk) atau < 50 (keluar, hysteresis)
- `rgb_confidence` — float 0–1, digunakan oleh FusionStage untuk threshold adaptif
- `active_model` — melacak model mana yang digunakan: `"rgb"`, `"rgb_clahe"`, `"depth"`, `"depth_filtered"`, `"none"`

#### 3.3.2 Dual-Model Swap

Stage ini memilih model mana yang akan digunakan berdasarkan kondisi pencahayaan:

| Condition | Model | Input | active_model |
|---|---|---|---|
| Bright (brightness >= 35) | `ModelRGB_V4.2.pt` | RGB frame | `"rgb"` |
| Dark + depth model available | `ModelDepth_V4.pt` | Unfiltered depth colormap | `"depth"` |
| Dark + depth model + no raw | `ModelDepth_V4.pt` | Filtered depth colormap | `"depth_filtered"` |
| Dark + no depth model | `ModelRGB_V4.2.pt` | CLAHE-enhanced RGB | `"rgb_clahe"` |
| No models available | None | — | `"none"` |

Model depth di-**lazy-load** — hanya dimuat ke memory GPU pada frame gelap pertama, menghemat VRAM saat startup.

#### 3.3.3 Enhancement CLAHE

Untuk scene remang (gelap tapi tanpa model depth), frame RGB di-enhance menggunakan **CLAHE** (Contrast Limited Adaptive Histogram Equalization):

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_enhanced = self._clahe.apply(l)  # clipLimit=3.0, tileGridSize=(8,8)
enhanced = cv2.merge([l_enhanced, a, b])
return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

CLAHE bekerja di ruang warna LAB, meningkatkan channel L (lightness) sambil mempertahankan warna.

#### 3.3.4 Inference YOLOv8

YOLOWrapper melakukan inference dengan optimisasi berikut:

- **FP16 inference** — `half=True` saat CUDA tersedia, ~2x lebih cepat pada Tensor Cores
- **Input size 320×320** — dikurangi dari 416 untuk inference lebih cepat
- **GPU warm-up** — inference dummy saat loading pre-compile kernel CUDA
- **Batch tensor transfer** — `boxes.xyxy.cpu().numpy()` sekali, bukan per-box

```
input_size = 320 × 320 pixels (configurable)
```

**Mengapa 320×320?** Ini adalah keseimbangan antara kecepatan dan akurasi. Untuk robot keamanan pada jarak dekat (0.5–5m), 320×320 memberikan deteksi yang memadai sambil ~40% lebih cepat daripada 416×416.

#### 3.3.5 Validasi Kontrak Model (Load-time Guard)

Sebelum inference pertama, `YOLOWrapper.__init__` memvalidasi weight file yang di-load terhadap kontrak pipeline:

1. **File ada** — `ModelValidationError` jika path tidak ditemukan.
2. **Load berhasil** — exception saat parse (weight korup / arsitektur lebih baru dari `ultralytics` terpasang, mis. YOLOv26) ditangkap dan di-re-raise sebagai `ModelValidationError` dengan pesan yang menjelaskan kemungkinan mismatch arsitektur.
3. **Task didukung** — model harus bertipe `detect` atau `segment`. Model `classify`/`pose` ditolak karena tidak mengekspos `.boxes` yang dibutuhkan `detect()` (tanpa guard, model `classify` akan diam-diam mengembalikan deteksi kosong setiap frame).
4. **Class set cocok** — nama class harus persis `EXPECTED_CLASS_NAMES = ("mobil", "motor", "person")`. Model dengan class berbeda ditolak untuk mencegah desync `class_id → label` di FusionStage/AlertPanel.

Guard berjalan sekali saat load (bukan per-frame) sehingga tidak mempengaruhi performa runtime. Tujuannya: kegagalan weight yang salah muncul **loudly di startup**, bukan crash/silent behavior di tengah operasi. Test: `tests/test_yolowrapper_validation.py` (11 tests).

#### 3.3.6 Format Output Detection

```python
@dataclass
class Detection:
    class_id: int        # indeks class COCO (mis., 0 = person)
    class_name: str      # nama yang dapat dibaca (mis., "person")
    confidence: float    # 0.0 – 1.0
    bbox: List[int]      # [x1, y1, x2, y2] dalam pixel (format xyxy)
```

### 3.4 Stage C: `FusionStage` (Penggabungan Data)

Stage ini menjawab: **"Apa objek ini, dan seberapa jauh jaraknya?"** dengan mencocokkan detection YOLO ke obstacle depth.

#### 3.4.1 Arsitektur Two-Pass

FusionStage menggunakan pendekatan two-pass:

**PASS 1 — YOLO-first (dilewati dalam dark mode):**
Untuk setiap detection YOLO, langsung menyampling depth dari depth frame di dalam bbox YOLO. Ini memberikan nama kelas dan jarak dalam satu langkah.

```python
for det in data.detections:
    dist = self._sample_depth_in_bbox(depth_frame, depth_scale, det.bbox)
    if dist is None:
        continue  # No valid depth in this bbox
    # Assign class + distance + priority
```

**PASS 2 — Obstacle depth-only:**
Untuk obstacle depth yang tidak tercakup oleh detection YOLO manapun, tambahkan sebagai kelas generik "obstacle". Ini menangkap objek yang dilewatkan YOLO.

```python
for obs in data.obstacles:
    if already_covered_by_yolo(obs):
        continue
    if dist > 1.5:
        continue  # Filter far obstacles
    # Add as generic obstacle with demoted priority
```

#### 3.4.2 Direct Depth Sampling

Daripada mencocokkan box YOLO ke contour obstacle depth, PASS 1 menyampling depth **langsung dari depth frame** di dalam bbox YOLO:

```python
def _sample_depth_in_bbox(depth_frame, depth_scale, bbox):
    # Use center 60% of bbox to avoid background pixels at edges
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.2)
    region = depth_frame[cy1:cy2, cx1:cx2].astype(np.float32) * depth_scale
    valid = region[(region >= min_dist) & (region <= max_dist)]
    return float(np.percentile(valid, 25))  # 25th percentile
```

**Mengapa percentile ke-25?** Ini memberikan jarak ke permukaan terdekat objek — yang penting untuk collision avoidance. Region 60% tengah menghindari pixel background yang masuk ke tepi bbox.

#### 3.4.3 Metrik Overlap untuk PASS 2

Pada PASS 2, kita perlu memeriksa apakah sebuah obstacle depth sudah tercakup oleh detection YOLO. Kita menggunakan:

```
overlap_ratio = Area(Intersection) / min(Area(Depth), Area(YOLO))
```

Ini menggunakan **area terkecil** sebagai denominator, sehingga:
- Depth blob kecil di dalam box YOLO besar → overlap tinggi (benar)
- Box YOLO kecil di dalam depth blob besar → overlap tinggi (benar)

Jika `overlap_ratio > threshold`, obstacle sudah tercakup oleh YOLO dan dilewati.

#### 3.4.4 Threshold Matching Adaptif

Threshold overlap beradaptasi dengan kondisi pencahayaan:

| Kondisi | Threshold | Alasan |
|---|---|---|
| Normal (is_dark=False, rgb_confidence ≥ 0.5) | 0.5 (50%) | Matching ketat saat YOLO reliable |
| Dark atau low confidence | 0.3 (30%) | Matching longgar saat YOLO mungkin tidak akurat |

#### 3.4.5 Matriks Prioritas

**PASS 1 (YOLO detections dengan depth):**

| Kelas | Jarak | Priority | Aksi |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < `warning_distance` | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

**PASS 2 (obstacle depth-only):**

| Jarak | Priority | Alasan |
|---|---|---|
| < 0.5m | 1 | Sangat dekat — diturunkan dari 0 untuk menghindari false STOP pada obstacle generik |
| < 1.0m | 2 | Dekat — level warning |
| ≥ 1.0m | 3 | Normal |

Threshold berasal dari `DetectionConfig` (dapat dikonfigurasi saat runtime via slider GUI).

#### 3.4.6 Format Output

```python
{
    "object_class":  str,      # "person", "chair", "obstacle"
    "distance_m":    float,    # meter
    "zone":          str,      # "left" | "center" | "right"
    "priority":      int,      # 0 = paling berbahaya
    "bbox":          [x1, y1, x2, y2],  # format xyxy
    "action":        str | None,         # "STOP" atau None
}
```

### 3.5 Stage D: `NavigationStage` (Path Planning)

Stage ini menghitung rekomendasi steering menggunakan pendekatan **polar histogram + berbasis gap** (VFH-lite). Ia menjawab: **"Ke arah mana robot harus steering, dan seberapa cepat?"**

#### 3.5.1 Polar Histogram

Frame depth dibagi menjadi N sektor horizontal (default: 18 sektor dengan lebar ~5° per sektor). Untuk setiap sektor, jarak percentile ke-10 dihitung (robust terhadap noise):

```
Sector 0 (leftmost)  → min_dist = 0.3m  (blocked)
Sector 1             → min_dist = 0.4m  (blocked)
...
Sector 9 (center)    → min_dist = 4.5m  (free)
...
Sector 17 (rightmost)→ min_dist = 3.8m  (free)
```

#### 3.5.2 Deteksi Sector Terhalang

Sebuah sektor ditandai **blocked** jika jarak minimumnya kurang dari clearance yang dibutuhkan robot:

```
min_gap = robot_width + 2 × safety_margin
       = 0.5m + 2 × 0.3m = 1.1m

blocked[i] = histogram[i] < min_gap
```

#### 3.5.3 Pencarian Gap

Sektor bebas yang berdekatan membentuk **gap**. Untuk setiap gap, indeks sektor pusat dikonversi ke sudut:

```
angle_per_sector = (2 × max_steer_deg) / num_sectors
center_angle = -max_steer_deg + center_sector × angle_per_sector
```

Dengan `max_steer_deg = 45°` dan `num_sectors = 18`:
- `angle_per_sector = 90 / 18 = 5°` per sektor
- Sektor 0 (paling kiri) → `center_angle = -45°` (belok kiri maksimal)
- Sektor 9 (tengah) → `center_angle = 0°` (lurus)
- Sektor 17 (paling kanan) → `center_angle = +45°` (belok kanan maksimal)

Setiap gap diberi skor:

```
score = 0.5 × center_bias + 0.3 × width_score + 0.2 × clearance_score
```

- `center_bias = 1.0 - (|center_angle| / max_steer_deg)` — lebih suka gap dekat pusat (0°), penalti sudut ekstrem
- `width_score = width_sectors / num_sectors` — gap yang lebih lebar lebih aman
- `clearance_score = min(min_distance / max_distance, 1.0)` — gap yang lebih dalam memungkinkan perjalanan lebih cepat

Sudut pusat gap dengan skor tertinggi menjadi sudut steering yang direkomendasikan.

#### 3.5.4 Hysteresis (Anti-Oscillation)

Untuk mencegah steering berpindah kiri/kanan setiap frame, stage ini tetap pada heading sebelumnya selama N frame (default: 5) jika masih dalam gap bebas. Pengecekan dilakukan dengan mengkonversi sudut steering sebelumnya kembali ke indeks sektor:

```
prev_sector = int((prev_angle + max_steer_deg) / angle_per_sector)
```

Jika `prev_sector` valid (dalam rentang `[0, num_sectors)`) dan sektor tersebut tidak blocked, maka steering sebelumnya dipertahankan. Jika tidak, gap terbaik baru dipilih dan counter hysteresis direset ke N.

#### 3.5.5 Safety Override

Jika FusionStage menemukan person pada priority 0 (di danger zone), navigation memaksa `STOPPED` terlepas dari gap yang tersedia. Ini memastikan robot tidak pernah steering menghindari person yang seharusnya ia hentikan.

#### 3.5.6 Mapping Kecepatan

Kecepatan dihitung berdasarkan jarak minimum pada **sektor tengah** frame (sepertiga bagian tengah dari histogram):

```
center_start = num_sectors // 3
center_end = 2 × num_sectors // 3
min_dist_ahead = min(histogram[center_start:center_end])
```

Dengan `num_sectors = 18`: sektor 6–11 (sepertiga tengah) merepresentasikan area langsung di depan robot. Kecepatan naik secara linear berdasarkan `min_dist_ahead`:

```
if min_dist_ahead < danger_distance:    speed = 0.0 (stop)
if min_dist_ahead >= warning_distance:  speed = 1.0 (full)
else:                                   speed = (min_dist_ahead - danger) / (warning - danger)
```

**Kenapa sepertiga tengah?** Sektor tepi (kiri/kanan ekstrem) bisa terhalang oleh dinding atau obstacle di samping yang tidak menghalangi jalur lurus robot. Hanya area di depan langsung yang menentukan apakah robot harus melambat.

#### 3.5.7 Format Output

```python
{
    "steering_angle_deg": float,   # -45 (kiri) sampai +45 (kanan), 0 = lurus
    "speed": float,                # 0.0 (stop) sampai 1.0 (full)
    "status": str,                 # "CLEAR" | "AVOIDING" | "BLOCKED" | "STOPPED"
    "gaps": List[Dict],            # Gap yang dapat dilalui dengan angle, width, distance
    "histogram": List[float],      # Jarak minimum per sektor
    "blocked_sectors": List[bool], # Flag blocked per sektor
}
```

### 3.6 Stage E: `VisualAnnotationStage` (Rendering HUD)

Stage terakhir menggambar overlay HUD ke `rgb_frame` dan `depth_colormap` **in-place** (sehingga HUD muncul pada view mana pun yang aktif):

1. **Corner brackets** — 8 garis per objek (bukan rectangle penuh, lebih sedikit clutter visual)
2. **Dark text plate** dengan label: `[ZONE] distance_m` atau `class_name [ZONE] distance_m`
3. **Color coding**: Soft Red (danger, priority <= 1), Amber (warning, priority <= 2), Lime Green (safe)
4. **Global status bar** (kiri-atas): `SYS: SAFE` / `SYS: WARN` / `SYS: DANGER`
5. **Navigation HUD** (kiri-bawah): `NAV: AVOIDING | STEER +22 deg | SPD 50%`
6. **Steering arrow** (tengah-bawah): panah berarah yang menunjukkan heading yang direkomendasikan

**Formula steering arrow pada frame:**

```
cx = frame_width / 2
cy = frame_height - 40
arrow_len = 60
angle_rad = radians(steering_angle_deg)
ax = cx + arrow_len × sin(angle_rad)
ay = cy - arrow_len × cos(angle_rad)
```

Sudut 0° menghasilkan panah lurus ke atas. Sudut +22° (belok kanan) menggeser ujung panah ke kanan dengan `sin(22°) ≈ 0.37`, sehingga `ax = cx + 22`. Sudut negatif (belok kiri) menggeser ke kiri. Komponen `cos` mengontrol panjang vertikal panah — pada 0°, panah penuh 60px ke atas; pada ±45°, panah lebih pendek secara vertikal (`cos(45°) ≈ 0.71`) dan lebih panjang secara horizontal.

Prioritas sumber data:
1. `fused_output` (dari FusionStage) — bbox dalam format xyxy
2. `obstacles` (dari DepthProcessingStage) — bbox dalam format xywh
3. `detections` (fallback YOLO-only) — bbox dalam format xyxy, distance=99.0

---

## 4. Pancaran Sinyal & Memory Safety

Setelah `FrameProcessor` menyelesaikan pipeline, data harus dikirim dengan aman dari background thread ke main GUI thread.

### 4.1 Memory Safety QImage

`QImage` milik Qt **tidak** memiliki data pixel yang mendasarinya. Jika Python melakukan garbage-collect pada array NumPy sementara Qt masih menggunakan QImage, itu menyebabkan segmentation fault (crash).

**Solusi:** Panggil `.tobytes()` untuk membuat copy memory yang terisolasi dan aman:

```python
# numpy channel swap (lebih cepat dari cv2.cvtColor) + .tobytes() untuk safety
frame_rgb = frame_bgr[:, :, ::-1].copy()
qimage = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, Format_RGB888)
```

### 4.2 Pancaran Sinyal

Thread memancarkan enam signal:

1. **`frame_pair_ready(QImage, QImage)`** — gambar RGB dan depth untuk display
2. **`distance_info_ready(str, object, str)`** — label, jarak, zone untuk alert panel
3. **`obstacles_ready(list)`** — obstacle fused atau mentah untuk radar view
4. **`navigation_ready(dict)`** — steering angle, speed, status, gaps untuk alert panel + radar
5. **`light_mode_changed(bool)`** — flag is_dark untuk mode auto-switch view
6. **`error(str)`** — pesan error fatal dari acquisition thread (mis., kamera gagal dibuka)

Semua signal menyeberangi boundary thread via **signal-slot mechanism** Qt, yang thread-safe by design. Fungsi slot dijalankan di main thread.

### 4.3 Kontrol Frame Rate

Loop pemrosesan **tidak** menggunakan `msleep` — queue memberikan flow control alami. Acquisition thread menangkap pada kecepatan hardware (30 FPS), dan loop pemrosesan menarik frame secepat yang bisa memprosesnya. Jika queue penuh, frame tertua di-drop (backpressure).

---

## 5. Fase Rendering GUI

Main thread menangkap signal yang dipancarkan dan mendistribusikan data ke komponen visual.

### 5.1 DepthView (Display Kamera)

Mengkonversi `QImage` yang aman menjadi `QPixmap` hardware-accelerated dan merendernya. Optimisasi:
- `setScaledContents(True)` dipanggil sekali saat init (bukan per frame)
- Hanya memperbarui label untuk page yang sedang visible (RGB / Depth)
- Menangani depth map kosong (mode webcam) via pengecekan `.isNull()`

### 5.2 AlertPanel (Display Status)

Membaca jarak dan zone dari `distance_info_ready` dan memperbarui:
- Nama objek (dari kelas YOLO atau "OBSTACLE")
- Jarak dalam meter
- Zone (LEFT / CENTER / RIGHT)
- Rekomendasi aksi (STOP / SLOWDOWN / GO)
- Status berkode warna (DANGER / WARNING / SAFE)

**Optimisasi:** Stylesheet hanya diterapkan saat status **berubah** (mis., SAFE → DANGER). Pada kondisi tunak, nol kalkulasi ulang stylesheet per frame.

### 5.3 RadarView (Display Spasial 90° FOV)

Merender radar top-down 90° FOV wedge yang menunjukkan posisi obstacle.

**Optimisasi:** Background statis (ring, label, garis FOV, garis zone) di-**pre-render sekali** ke dalam `QPixmap` yang di-cache. Hanya garis sweep dan blip obstacle yang digambar ulang setiap frame. Ini mengurangi pekerjaan paint sebesar ~80%.

#### 5.3.1 Mapping Koordinat Polar

Pusat bbox setiap obstacle dipetakan ke sudut pada radar:

```
angle_deg = 135 - (bbox_center_x / frame_width) × 90
```

- Tepi kiri frame (0px) → 135° (kiri radar)
- Pusat frame (320px) → 90° (pusat radar)
- Tepi kanan frame (640px) → 45° (kanan radar)

#### 5.3.2 Konversi Cartesian

```
dist_frac = min(distance_m / RADAR_MAX_DEPTH, 1.0)
bx = cx + dist_frac × r × cos(angle_deg)
by = cy - dist_frac × r × sin(angle_deg)
```

#### 5.3.3 Steering Arrow pada Radar

Panah rekomendasi steering dari NavigationStage juga ditampilkan pada radar. Sudut steering (-45° hingga +45°) dipetakan ke sudut radar:

```
radar_angle = 90 - steering_angle_deg
```

- `steering = 0°` (lurus) → `radar_angle = 90°` (panah ke atas pada radar)
- `steering = -45°` (belok kiri) → `radar_angle = 135°` (panah ke kiri radar)
- `steering = +45°` (belok kanan) → `radar_angle = 45°` (panah ke kanan radar)

Panjang panah = `0.7 × r` (70% radius radar). Warna panah mengikuti status: merah (STOPPED/BLOCKED), oranye (AVOIDING), hijau (CLEAR).

### 5.4 Recording & Playback (Mode Validasi Offline)

Sistem mendukung perekaman sesi live dan playback rekaman melalui pipeline penuh — berguna untuk validasi ulang tanpa hardware atau analisis offline.

#### 5.4.1 VideoRecorder (Rekam Sesi)

`Vision/src/video_recorder.py` menulis RGB + depth ke disk untuk diputar ulang nanti. Dua mode tersedia:

**Mode GUI (non-blocking)** — dipanggil dari pipeline:
```python
from Vision.src.video_recorder import VideoRecorder
rec = VideoRecorder(save_dir="data/recordings")
rec.start_recording()
rec.record_frame(rgb_bgr, depth_raw_uint16, depth_filtered_uint16)
# ... ulangi untuk setiap frame ...
rec.stop_recording()
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

#### 5.4.2 VideoPlaybackThread (Putar Ulang)

`Vision/src/video_playback_thread.py` adalah `QThread` yang membaca folder recording dan menjalankan frame pair (RGB+depth) melalui pipeline 5-stage penuh. Sinyal yang dipancarkan identik dengan `CameraThread` (lihat Bagian 4.2) — MainWindow cukup menukar instance tanpa mengubah wiring.

**Kontrol playback (public API):**

| Method | Fungsi |
|---|---|
| `start_playback(recording_dir)` | Mulai playback dari folder recording |
| `stop_playback()` | Hentikan playback |
| `set_paused(bool)` | Pause / resume |
| `toggle_pause()` | Toggle pause state |
| `set_speed(multiplier)` | Set kecepatan (0.25–4.0x, di-clamp) |
| `set_loop(bool)` | Aktifkan/nonaktifkan loop otomatis di akhir rekaman |
| `is_paused` (property) | Status pause saat ini |
| `set_depth_thresholds(min_m, max_m)` | Update threshold depth (compat dengan CameraThread) |

**Format depth yang didukung:**
- Stacked `.npy` (cepat, preferensi utama): `depth.npy` shape `(N, H, W)` uint16
- Individual `.npy` (legacy): `depth/frame_00000.npy`, `depth/frame_00001.npy`, ...
- RGB-only: tanpa file depth → stage depth di-skip otomatis

#### 5.4.3 Input Source Switcher (Live Camera ↔ Video File)

`ControlsPanel` memiliki toggle **Live Camera** ↔ **Video File**. Saat user memilih Video File, `QFileDialog` terbuka dan `MainWindow`:
1. Menghentikan `CameraThread` (jika sedang jalan)
2. Membuat instance `VideoPlaybackThread` baru
3. Me-reconnect sinyal identik ke slot GUI yang sama
4. Memulai playback dari folder recording

Karena kontrak sinyal identik, semua widget (DepthView, RadarView, AlertPanel) bekerja tanpa modifikasi. Untuk kembali ke live, user memilih Live Camera dan `CameraThread` di-restart.

---

## 6. Ringkasan Performa

| Komponen | Latensi | Catatan |
|---|---|---|
| RealSense capture | ~33ms | Hardware-limited pada 30 FPS |
| DepthProcessingStage (LUT) | ~1–3ms | LUT indexing + obstacle detection |
| YOLODetectionStage | ~5–10ms (P50), ≤25ms P95 | FP16 GPU inference (RTX A4000, 320px) |
| FusionStage | <1ms | Perhitungan overlap + depth sampling |
| NavigationStage | <1ms | Polar histogram + gap selection |
| VisualAnnotationStage | ~1ms | OpenCV drawing pada RGB + depth |
| QImage conversion | ~0.5ms | numpy swap + tobytes() |
| **Total per frame** | **~10–20ms (P50), ~30ms P95** | Target: 30 FPS (budget 33ms) |

> **Catatan benchmark vs real-time:** Angka di atas adalah latensi pipeline murni (`FrameProcessor.process()` pada synthetic frames). Tiga mode pengukuran:
>
> | Mode | FPS | Kondisi | Sumber |
> |---|---|---|---|
> | Pipeline benchmark | 42.5 FPS (P50) | GPU RTX A4000, synthetic frames, tanpa GUI/Qt | `tests/benchmark.py` |
> | Real-time (GPU + GUI) | ~20 FPS | GPU + Qt signal-slot + QImage + GUI render | Observasi runtime |
> | Real-time (CPU-only) | 1.3–3.4 FPS | AMD Ryzen 5 6600H, tanpa GPU, RealSense | `Doc/field_test_report_role5.md` |
>
> Gap benchmark → real-time disebabkan oleh overhead yang **tidak diukur** di benchmark: konversi QImage (~0.5ms × 2 frame), emisi 6 Qt signal, render `DepthView` (scaled pixmap), `RadarView` (20 FPS paint timer), `AlertPanel` (stylesheet update), dan resize event dari `QScrollArea`. Gap CPU-only disebabkan oleh inferensi YOLO tanpa GPU (~200ms vs ~13ms di GPU).

---

## Ringkasan Alur

```
Kamera menangkap cahaya (30 FPS)
  → Acquisition thread capture + filter (spatial, temporal, hole-filling)
    → Depth tanpa filter dipertahankan (untuk model depth)
  → Queue mengantar frame ke loop pemrosesan
    → Depth dikonversi ke LUT colormap + deteksi obstacle
    → YOLO mengidentifikasi objek (dual-model swap: RGB/depth/CLAHE)
    → Fusion mencocokkan keduanya (PASS 1: direct sampling, PASS 2: overlap)
    → Navigation menghitung steering (polar histogram + gap selection)
    → Visual annotation menggambar HUD + steering arrow (in-place)
  → Signal mentransmisikan data (thread-safe QImage + typed signal)
    → DepthView merender gambar (visible-only updates)
    → AlertPanel menampilkan status (change-only stylesheets)
    → RadarView memplot posisi (cached background)
```

*(Seluruh siklus ini terjadi dalam waktu kurang dari ~30 milidetik, 30+ kali per detik.)*
