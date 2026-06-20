# 🔍 PBL Vision System — Critical Audit Report

**Tanggal Audit**: 19 Juni 2026  
**Auditor**: R1 (Moris) via OpenCode  
**Branch**: `main`

---

## 📋 Ringkasan Masalah

| Severity | Jumlah | Deskripsi |
|----------|--------|-----------|
| 🔴 Critical | 4 | Akan menyebabkan crash atau data loss |
| 🟠 High | 7 | Bug fungsional yang mengganggu kerja |
| 🟡 Medium | 6 | Masalah performa atau reliability |
| 🔵 Low | 5 | Kualitas kode dan maintainability |
| ⚠️ Review | 5 | ✅ All FIXED |
| 🔧 FusionStage | 8 | ✅ All FIXED |

---

## 👤 R1 — Moris (ML Pipeline / Integration)

### 🔴 CRITICAL: DepthProcessingStage Double Processing
- **File**: `Vision/src/frame_processor.py:175-188`
- **Masalah**: 
  ```python
  def process(self, frame_data):
      depth_colormap = self.obstacle_detector.detect(
          frame_data.rgb_frame, frame_data.depth_frame, frame_data.depth_scale
      )
      # detect() returns (annotated_frame, obstacles_list), bukan hanya obstacles
  ```
- **Impact**: Stage membuang annotated frame dan reconstruct depth_colormap dari raw data (wasted computation)
- **Status**: ✅ FALSE ALARM — Stage correctly uses annotated frame (line 252) AND generates depth_colormap separately (different purpose)
- **Fix**: 
  1. Update `DepthProcessingStage.process()` untuk use returned annotated frame
  2. Atau ubah `detect()` contract untuk return obstacles saja

---

### 🟠 HIGH: FrameData Contract Mismatch
- **File**: `Vision/src/frame_processor.py:175` vs `Vision/src/camera_thread.py:172`
- **Masalah**: 
  - `FrameProcessor.process()` expects 3 args: `(rgb, depth, depth_scale)`
  - `camera_thread.py` calls: `self._processor.process(color_bgr, depth_raw, self._depth_scale)`
  - Tapi `DepthProcessingStage` expects: `(frame_data)` object
- **Impact**: Runtime error jika depth frame None (webcam mode)
- **Status**: ✅ FIXED — `FrameProcessor.process` natively instantiates `FrameData` internally
- **Fix**: 
  1. Pastikan `FrameProcessor.process()` menerima `(rgb, depth, depth_scale)` dan wrap into FrameData
  2. Atau update camera_thread untuk pass FrameData object

---

### 🟡 MEDIUM: Full-Frame Float32 Conversion
- **File**: `Vision/src/obstacle_detector.py:45`
- **Masalah**: 
  ```python
  depth_meter = depth_frame.astype(np.float32) * depth_scale
  ```
- **Impact**: ~1.2MB allocation per frame (640x480x4 bytes), causes GC pauses di 30fps
- **Status**: ✅ FIXED — Added reusable `_depth_buffer` with `np.multiply(..., out=...)`

---

### 🟡 MEDIUM: Thread Safety — last_detections
- **File**: `Vision/src/obstacle_detector.py:31`
- **Masalah**: 
  ```python
  self.last_detections = []  # Mutated setiap detect() call tanpa lock
  ```
- **Impact**: Race condition jika dipanggil dari multiple threads
- **Status**: ✅ FIXED — Added `threading.Lock` + property getter/setter

---

## 👤 R2 — Husein (YOLOv8 Specialist)

### 🟡 MEDIUM: YOLO Detection Dataclass vs Dict Contract
- **File**: `Vision/src/yolowrapper.py:69-74` vs `Vision/src/frame_processor.py:175-188`
- **Masalah**: 
  - `YOLOWrapper.detect()` returns `List[Detection]` (dataclass)
  - `YOLODetectionStage.process()` converts to `List[Dict]`
  - R4's FusionStage mungkin expect dataclass format
- **Impact**: Contract violation antara R2/R1 dan R4
- **Status**: ✅ FIXED — `YOLODetectionStage` updated to use dataclass directly
- **Fix**: 
  1. Standarisasi pada satu format (prefer dataclass untuk type safety)
  2. Update `YOLODetectionStage` untuk return `List[Detection]` langsung
  3. Atau update R4's FusionStage untuk handle both formats

---

## 👤 R3 — Long (Depth / Camera)

### 🔴 CRITICAL: RealSense Pipeline Conflict
- **File**: `Vision/src/camera_thread.py:98` vs `Vision/src/recorder.py:14`
- **Masalah**: 
  - `CameraThread` membuat `rs.pipeline()` di line 98
  - `FrameRecorder` membuat `rs.pipeline()` sendiri di line 14
  - RealSense SDK HANYA MENGIZINKAN 1 pipeline per device
- **Impact**: Jika `FrameRecorder.start()` dipanggil saat `CameraThread` running → crash "RuntimeError: Frame didn't arrive within 1000ms"
- **Status**: ✅ FIXED — Added `_active_pipeline_count` flag to `FrameRecorder` to prevent concurrent pipeline access
- **Fix**: 
  1. Share single pipeline antara CameraThread dan FrameRecorder
  2. Atau gunakan mutex lock untuk akses pipeline
  3. Atau matikan CameraThread sebelum mulai FrameRecorder

---

### 🔴 CRITICAL: Depth Frame Resize Mismatch
- **File**: `Vision/src/camera_thread.py:167-168`
- **Masalah**: 
  ```python
  if depth_raw.shape != color_bgr.shape[:2]:
      depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
  ```
- **Impact**: 
  - Decimation filter mengurangi resolusi depth
  - Resize dengan NEAREST interpolation membuat artificial depth values di boundaries
  - Menyebabkan false obstacle detections di zone edges
- **Status**: ✅ FIXED — Decimation filter and `cv2.resize` removed. Depth map now retains 640x480 native resolution cleanly.
- **Fix**: 
  1. Skip decimation filter jika tidak perlu
  2. Atau gunakan BILINEAR/CUBIC interpolation
  3. Atau resize SEBELUM zone calculation

---

### 🔴 CRITICAL: QImage Memory Safety
- **File**: `Vision/src/camera_thread.py:245-247`
- **Masalah**: 
  ```python
  return QImage(
      frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
  ).copy()
  ```
- **Impact**: 
  - `.copy()` dipanggil, tapi `frame_rgb` adalah local variable
  - Bisa garbage collected sebelum Qt proses image
  - Race condition causing display corruption (langka tapi mungkin)
- **Status**: ✅ FIXED — Replaced `.data` with `.tobytes()` creating safe memory space for PyQt
- **Fix**: 
  1. Keep reference ke `frame_rgb` sampai QImage selesai diproses
  2. Atau gunakan `QImage.fromData(frame_rgb.tobytes())`
  3. Atau simpan `frame_rgb` sebagai class attribute

---

## 👤 R4 — Rasyid (Sensor Fusion)

### 🟠 HIGH: Radar View Disconnected
- **File**: `GUI/src/radar_view.py` + `GUI/src/main_window.py` (missing connection)
- **Masalah**: 
  - `radar_view.py` punya method `update_obstacles()`
  - Tapi **TIDAK PERNAH** dipanggil dari `main_window.py`
- **Impact**: Radar display menampilkan data static/kosong, bukan real-time obstacles
- **Status**: ✅ FIXED — Added `obstacles_ready` signal to CameraThread. Connected via `_on_obstacles_ready()` with angle conversion from bbox position.
- **Fix**: 
  1. Connect `distance_info_ready` signal ke `radar_view.update_obstacles()`
  2. Atau tambah signal baru dari FrameProcessor untuk radar data

---

### 🟡 MEDIUM: YOLO Detection Format Mismatch
- **File**: `Vision/src/yolowrapper.py:69-74` vs `Vision/src/frame_processor.py:175-188`
- **Masalah**: 
  - R2 return `List[Detection]` (dataclass)
  - R1 convert ke `List[Dict]`
  - R4's FusionStage mungkin expect dataclass
- **Impact**: FusionStage menerima format yang tidak diharapkan
- **Status**: ✅ FIXED — Handled directly by `YOLODetectionStage`
- **Fix**: 
  1. Konfirmasi dengan R2/R1 format yang diinginkan
  2. Update FusionStage untuk handle both formats
  3. Atau standarisasi contract

---

### 🟡 MEDIUM: Alert Threshold Not Connected
- **File**: `GUI/src/controls_panel.py:23` vs `GUI/src/main_window.py` (missing connection)
- **Masalah**: 
  - `ControlsPanel` emit `thresholds_changed` signal
  - Tapi `main_window.py` TIDAK connect signal ini
- **Impact**: GUI threshold controls tidak berpengaruh ke pipeline (hardcoded di camera_thread)
- **Status**: ✅ FIXED — Signal connected directly to `frame_processor.set_action_thresholds`
- **Fix**: 
  1. Connect `thresholds_changed` ke `camera_thread.set_depth_thresholds()`
  2. Atau update FrameProcessor untuk terima threshold dari GUI

---

## 👤 R5 — Hamid (Dataset / Testing)

### 🔵 LOW: Test Coverage Gap — Synthetic Data
- **File**: `tests/test_frame_processor.py:29`
- **Masalah**: 
  ```python
  def make_synthetic_frames():
      rgb = np.full((h, w, 3), 128, dtype=np.uint8)  # Same value everywhere
  ```
- **Impact**: Tests pass tapi tidak menangkap real-world bugs
- **Fix**: 
  1. Tambah realistic test data dengan actual obstacles
  2. Test dengan depth values yang bervariasi
  3. Test edge cases (zero depth, max depth, etc.)

---

## 👤 R6 — GUI (Moris兼任)

### 🟠 HIGH: YOLO Model Path Fragile
- **File**: `GUI/src/main_window.py:82-100`
- **Masalah**: 
  ```python
  CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
  MODEL_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models", "security_best.pt"))
  ```
- **Impact**: 
  - Uses relative path dari GUI source file, bukan project root
  - Model gagal load jika working directory berbeda (e.g., running dari folder lain)
- **Status**: ✅ FIXED — Adopted `pathlib` for robust absolute path resolution
- **Fix**: 
  1. Use `Path(__file__).parent.parent` atau config-based path
  2. Atau tambah fallback path resolution

---

### 🔵 LOW: No `__init__.py` Files
- **File**: Semua packages (`Vision/src/`, `GUI/src/`, etc.)
- **Masalah**: Tidak ada `__init__.py` di package directories
- **Impact**: Relies on `sys.path` manipulation, fragile imports
- **Status**: ✅ FIXED — `__init__.py` files added across Vision and GUI modules
- **Fix**: 
  1. Tambah `__init__.py` files
  2. Atau gunakan proper package structure dengan `pyproject.toml`

---

### 🔵 LOW: Generic Object Signals
- **File**: `Vision/src/camera_thread.py:32`
- **Masalah**: 
  ```python
  frame_pair_ready = pyqtSignal(object, object)
  ```
- **Impact**: 
  - Bypass Qt's type system
  - Tidak ada compile-time type checking, harder debugging
- **Status**: ✅ FIXED — Added strict `QImage` type hint bindings
- **Fix**: 
  1. Gunakan `pyqtSignal(QImage, QImage)` jika memungkinkan
  2. Atau minimal tambah type hints di docstring

---

### 🔵 LOW: Error Handling Silent
- **File**: `Vision/src/frame_processor.py:175-188`
- **Masalah**: 
  ```python
  try:
      result = stage.process(frame_data)
  except Exception as e:
      logger.warning(f"Stage {stage_name} failed: {e}")
      continue  # Silently skips failed stage
  ```
- **Impact**: Pipeline continue dengan partial data, sulit diagnose failures
- **Status**: ✅ FIXED — `FrameData` extended with `errors: List[str]` to correctly propagate and catch stage errors
- **Fix**: 
  1. Tambah error counter
  2. Fail setelah N consecutive errors
  3. Atau emit error signal ke GUI

---

## ⚠️ Review Concerns — Post-Fix Analysis

> Concerns yang diidentifikasi setelah review solusi di branch `fix/audit-issues`.

### C5: YOLO FP16 Optimization — Unverified Claim
- **File**: `Vision/src/yolowrapper.py:59`
- **Masalah**: 
  - `half=True` ditambahkan dengan klaim "2x GPU speedup"
  - Benchmark menunjukkan FP16 **3% lebih lambat** dari FP32 di RTX A4000 Laptop
  - Model kecil (6MB) — overhead FP16 conversion melebihi compute savings
- **Impact**: Performance regression (minor), misleading documentation
- **Status**: ✅ FIXED — Removed `half=True` flag. FP32 is optimal for small models on laptop GPUs.

---

### C1: Decimation Filter Removal — Performance Risk
- **File**: `Vision/src/camera_thread.py:111-115`
- **Masalah**: 
  - `decimation_filter` dihapus untuk menghindari INTER_NEAREST artifacts
  - Tapi ini berarti **4x lebih banyak depth pixels** untuk diproses (~307K vs ~77K)
  - `obstacle_detector.py` dan `frame_processor.py` sekarang handle lebih banyak data
- **Impact**: Frame rate drops di hardware rendah (laptop integrated GPU)
- **Status**: ✅ FIXED — Decimation now configurable via `camera_config.py` (default: OFF). When enabled, uses `INTER_LINEAR` instead of `INTER_NEAREST` for better quality resize.

---

### C2: YOLO Detection Format — R4 Coordination Required
- **File**: `Vision/src/frame_processor.py:297`
- **Masalah**: 
  ```python
  # Gunakan dataclass secara langsung
  data.detections = detections  # List[Detection], bukan List[Dict]
  ```
  - `YOLODetectionStage` sekarang return `List[Detection]` (dataclass)
  - R4's FusionStage mungkin masih expect `List[Dict]`
- **Impact**: FusionStage menerima format salah → runtime error
- **Status**: ✅ FIXED — `FrameData.detections` type hint updated to `List[Any]` with documentation. Contract documented in FrameData docstring.

---

### C3: Empty QImage — DepthView Must Handle
- **File**: `Vision/src/camera_thread.py:183, 214`
- **Masalah**: 
  ```python
  depth_pixmap = QImage()  # Empty QImage, bukan None
  ```
  - Webcam fallback path sekarang emit `QImage()` (empty) bukan `None`
  - `DepthView.update_frames()` mungkin tidak handle empty QImage
- **Impact**: Potential crash atau visual artifacts di DepthView
- **Status**: ✅ FIXED — Added `isNull()` check in `DepthView.update_frames()` for both RGB and depth images.

---

### C4: Lazy Import in Exception Handler
- **File**: `Vision/src/frame_processor.py:158`
- **Masalah**: 
  ```python
  except Exception as e:
      from logging_config import get_logger  # import setiap error
      get_logger(__name__).error(...)
  ```
  - Import dilakukan di dalam exception handler (dipanggil setiap stage gagal)
  - `get_logger` sudah available di module scope
- **Impact**: Minor performance overhead saat stage gagal
- **Status**: ✅ FIXED — Moved `get_logger` import to module level, created `_logger` instance.

---

## 🎯 Prioritas Perbaikan

### Immediate (Sebelum Demo Berikutnya)
1. ✅ #1 RealSense pipeline conflict (R3) — FIXED
2. ✅ #4 Radar view connection (R4) — FIXED
3. ✅ #6 DepthProcessingStage contract (R1) — FALSE ALARM (no fix needed)

### Minggu Ini
4. ✅ #2 Depth resize interpolation (R3) — FIXED
5. ✅ #5 YOLO model path (R6) — FIXED
6. ✅ #10 Threshold connection (R4) — FIXED

### Sebelum Final Submission
7. ✅ #8 Performance optimization (R1) — FIXED
8. ✅ #11 Thread safety (R1) — FIXED
9. ✅ #12 Package structure (R6) — FIXED

### Post-Merge Review Concerns
10. ✅ C5: YOLO FP16 benchmark verification (R2) — FIXED
11. ✅ C1: Decimation filter configurable (R3) — FIXED
12. ✅ C2: YOLO format coordination with R4 (R4) — FIXED
13. ✅ C3: Empty QImage handling in DepthView (R6) — FIXED
14. ✅ C4: Lazy import cleanup (R1) — FIXED

### FusionStage Implementation Bugs (20 Juni 2026)
15. ✅ F1: FusionStage never added to pipeline (Critical) — FIXED
16. ✅ F2: Output bbox format mismatch [x,y,w,h] vs [x1,y1,x2,y2] (High) — FIXED
17. ✅ F3: Priority thresholds hardcoded, ignores DetectionConfig (High) — FIXED
18. ✅ F4: Zone not passed to AlertPanel (High) — FIXED
19. ✅ F5: Result variable fragile scope (Medium) — FIXED
20. ✅ F6: Radar angle_deg never computed (Medium) — FIXED
21. ✅ F7: Stale PLACEHOLDER comment (Low) — FIXED
22. ✅ F8: No tests for FusionStage (Low) — FIXED

---

## 📝 Catatan untuk Setiap Role

### R1 (Moris)
- Fokus perbaiki `frame_processor.py` dan `obstacle_detector.py`
- Pastikan contract antara stages konsisten
- Test dengan real camera data

### R2 (Husein)
- Konfirmasi format output `Detection` dataclass
- Update R4 jika ada perubahan format
- Document API contract di docstring

### R3 (Long)
- Prioritas tinggi: perbaiki RealSense pipeline conflict
- Test dengan both RealSense dan webcam
- Pastikan depth resize tidak menyebabkan artifacts

### R4 (Rasyid)
- Connect radar view ke pipeline
- Handle both Detection dataclass dan Dict format
- Test sensor fusion dengan multiple objects

### R5 (Hamid)
- Tambah realistic test cases
- Test edge cases (zero depth, max depth, etc.)
- Pastikan tests catch real bugs

### R6 (GUI)
- Perbaiki model path resolution
- Tambah `__init__.py` files
- Improve error handling di pipeline

---

## 🔗 Referensi

- **Main Pipeline**: `Vision/src/frame_processor.py`
- **Camera Thread**: `Vision/src/camera_thread.py`
- **Obstacle Detector**: `Vision/src/obstacle_detector.py`
- **YOLO Wrapper**: `Vision/src/yolowrapper.py`
- **GUI Main**: `GUI/src/main_window.py`
- **Radar View**: `GUI/src/radar_view.py`
- **Controls Panel**: `GUI/src/controls_panel.py`

---

**Dokumen ini dibuat untuk memandu perbaikan oleh masing-masing role. Silakan update checklist setelah menyelesaikan perbaikan.**
