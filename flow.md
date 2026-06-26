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
4. **Signal Routing:** GUI menghubungkan signal output thread (mis., `frame_pair_ready`, `obstacles_ready`, `navigation_ready`) ke fungsi update miliknya sendiri.
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
depth_raw_unfiltered = np.asanyarray(depth_frame.get_data())  # BEFORE filters
# ... apply filters to depth_frame ...
depth_raw_filtered = np.asanyarray(depth_frame.get_data())    # AFTER filters
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
    depth_frame: Optional[np.ndarray]  # H×W uint16 (filtered, None for webcam)
    depth_frame_raw: Optional[np.ndarray]  # H×W uint16 (unfiltered, for depth model)
    depth_colormap: Optional[np.ndarray]   # H×W×3 uint8 (filtered, for display)
    depth_colormap_raw: Optional[np.ndarray]  # H×W×3 uint8 (unfiltered, for depth model)
    depth_scale: float                 # raw → meters conversion factor
    obstacles: List[Dict]              # from DepthProcessingStage
    detections: List[Detection]        # from YOLODetectionStage
    fused_output: List[Dict]           # from FusionStage
    metadata: Dict[str, Any]           # timestamp, is_dark, rgb_confidence, active_model
    errors: List[str]                  # pipeline error log
```

### 3.2 Stage A: `DepthProcessingStage` (Pemahaman Spasial)

Stage ini mengkonversi data depth mentah menjadi informasi obstacle terstruktur dan memvisualisasikannya sebagai colormap berwarna.

#### 3.2.1 Generasi Colormap Berbasis LUT

Alih-alih membuat beberapa boolean mask per frame (lambat), stage ini menggunakan **Lookup Table (LUT) pre-computed 256-entry** yang memetakan indeks depth ke warna BGR:

```python
# Build once at init (and rebuild on threshold change)
lut = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    depth_m = (i / 255.0) * max_distance
    if depth_m < min_distance or depth_m > max_distance:
        lut[i] = (0, 0, 0)        # Black = invalid
    elif depth_m < danger_threshold:
        lut[i] = (0, 0, 255)       # Red = danger
    elif depth_m < warning_threshold:
        lut[i] = (0, 255, 255)     # Yellow = warning
    else:
        lut[i] = (0, 255, 0)       # Green = safe

# Per frame: single indexing operation (~3x faster than mask approach)
depth_m = depth_frame.astype(np.float32) * depth_scale
idx = np.clip(depth_m * scale, 0, 255).astype(np.uint8)
colormap = self._depth_lut[idx]
```

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
# Hysteresis: enter dark at < 35, exit at > 50 (prevents flicker near threshold)
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

#### 3.3.5 Format Output Detection

```python
@dataclass
class Detection:
    class_id: int        # COCO class index (e.g., 0 = person)
    class_name: str      # Human-readable name (e.g., "person")
    confidence: float    # 0.0 – 1.0
    bbox: List[int]      # [x1, y1, x2, y2] in pixels (xyxy format)
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

Alih-alih mencocokkan box YOLO ke contour obstacle depth, PASS 1 menyampling depth **langsung dari depth frame** di dalam bbox YOLO:

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

| Condition | Threshold | Why |
|---|---|---|
| Normal (is_dark=False, rgb_confidence ≥ 0.5) | 0.5 (50%) | Strict matching when YOLO is reliable |
| Dark or low confidence | 0.3 (30%) | Relaxed matching when YOLO may be inaccurate |

#### 3.4.5 Matriks Prioritas

**PASS 1 (YOLO detections dengan depth):**

| Class | Distance | Priority | Action |
|---|---|---|---|
| person | < `danger_distance` | 0 | STOP |
| other | < `danger_distance` | 1 | None |
| person | < `warning_distance` | 2 | None |
| other | ≥ `danger_distance` | 3 | None |

**PASS 2 (obstacle depth-only):**

| Distance | Priority | Why |
|---|---|---|
| < 0.5m | 1 | Very close — demoted from 0 to avoid false STOP |
| < 1.0m | 2 | Close — warning level |
| ≥ 1.0m | 3 | Normal |

Threshold berasal dari `DetectionConfig` (dapat dikonfigurasi saat runtime via slider GUI).

#### 3.4.6 Format Output

```python
{
    "object_class":  str,      # "person", "chair", "obstacle"
    "distance_m":    float,    # meters
    "zone":          str,      # "left" | "center" | "right"
    "priority":      int,      # 0 = most dangerous
    "bbox":          [x1, y1, x2, y2],  # xyxy format
    "action":        str | None,         # "STOP" or None
}
```

### 3.5 Stage D: `NavigationStage` (Path Planning)

Stage ini menghitung rekomendasi steering menggunakan pendekatan **polar histogram + berbasis gap** (VFH-lite). Ia menjawab: **"Ke arah mana robot harus steering, dan seberapa cepat?"**

#### 3.5.1 Polar Histogram

Frame depth dibagi menjadi N sektor horizontal (default: 18 sektor ~5° masing-masing). Untuk setiap sektor, jarak percentile ke-10 dihitung (robust terhadap noise):

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

Sektor bebas yang berdekatan membentuk **gap**. Setiap gap diberi skor:

```
score = 0.5 × center_bias + 0.3 × width_score + 0.2 × clearance_score
```

- `center_bias` — lebih suka gap dekat pusat (0°), penalti sudut ekstrem
- `width_score` — gap yang lebih lebar lebih aman
- `clearance_score` — gap yang lebih dalam memungkinkan perjalanan lebih cepat

Sudut pusat gap dengan skor tertinggi menjadi sudut steering yang direkomendasikan.

#### 3.5.4 Hysteresis (Anti-Oscillation)

Untuk mencegah steering berpindah kiri/kanan setiap frame, stage ini tetap pada heading sebelumnya selama N frame (default: 5) jika masih dalam gap bebas.

#### 3.5.5 Safety Override

Jika FusionStage menemukan person pada priority 0 (di danger zone), navigation memaksa `STOPPED` terlepas dari gap yang tersedia. Ini memastikan robot tidak pernah steering menghindari person yang seharusnya ia hentikan.

#### 3.5.6 Mapping Kecepatan

Kecepatan naik secara linear berdasarkan jarak minimum pada sektor tengah:

```
if min_dist < danger_distance:    speed = 0.0 (stop)
if min_dist >= warning_distance:  speed = 1.0 (full)
else:                             speed = (min_dist - danger) / (warning - danger)
```

#### 3.5.7 Format Output

```python
{
    "steering_angle_deg": float,   # -45 (left) to +45 (right), 0 = straight
    "speed": float,                # 0.0 (stop) to 1.0 (full)
    "status": str,                 # "CLEAR" | "AVOIDING" | "BLOCKED" | "STOPPED"
    "gaps": List[Dict],            # Navigable gaps with angle, width, distance
    "histogram": List[float],      # Min distance per sector
    "blocked_sectors": List[bool], # Blocked flag per sector
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
# numpy channel swap (faster than cv2.cvtColor) + .tobytes() for safety
frame_rgb = frame_bgr[:, :, ::-1].copy()
qimage = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, Format_RGB888)
```

### 4.2 Pancaran Sinyal

Thread memancarkan lima signal:

1. **`frame_pair_ready(QImage, QImage)`** — gambar RGB dan depth untuk display
2. **`distance_info_ready(str, object, str)`** — label, jarak, zone untuk alert panel
3. **`obstacles_ready(list)`** — obstacle fused atau mentah untuk radar view
4. **`navigation_ready(dict)`** — steering angle, speed, status, gaps untuk alert panel + radar
5. **`light_mode_changed(bool)`** — flag is_dark untuk mode auto-switch view

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

### 5.3 RadarView (Display Spasial 180°)

Merender radar top-down semicircular yang menunjukkan posisi obstacle.

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

---

## 6. Ringkasan Performa

| Komponen | Latensi | Catatan |
|---|---|---|
| RealSense capture | ~33ms | Hardware-limited pada 30 FPS |
| DepthProcessingStage (LUT) | ~1–3ms | LUT indexing + obstacle detection |
| YOLODetectionStage | ~5–10ms | FP16 GPU inference (RTX A4000, 320px) |
| FusionStage | <1ms | Perhitungan overlap + depth sampling |
| NavigationStage | <1ms | Polar histogram + gap selection |
| VisualAnnotationStage | ~1ms | OpenCV drawing pada RGB + depth |
| QImage conversion | ~0.5ms | numpy swap + tobytes() |
| **Total per frame** | **~10–20ms** | Target: 30 FPS (budget 33ms) |

---

## Ringkasan Alur

```
Camera grabs light (30 FPS)
  → Acquisition thread captures + filters (spatial, temporal, hole-filling)
    → Unfiltered depth preserved (for depth model)
  → Queue delivers frames to processing loop
    → Depth converts to LUT colormap + obstacle detection
    → YOLO identifies objects (dual-model swap: RGB/depth/CLAHE)
    → Fusion matches them (PASS 1: direct sampling, PASS 2: overlap)
    → Navigation computes steering (polar histogram + gap selection)
    → Visual annotation draws HUD + steering arrow (in-place)
  → Signals transmit data (thread-safe QImage + typed signals)
    → DepthView renders images (visible-only updates)
    → AlertPanel shows status (change-only stylesheets)
    → RadarView plots positions (cached background)
```

*(Seluruh siklus ini terjadi dalam waktu kurang dari ~20 milidetik, 30+ kali per detik.)*