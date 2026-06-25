"""
Vision/src/frame_processor.py — Pipeline orkestrator untuk pemrosesan frame.

Role: ML Pipeline Architect / Frame Processor Lead (Role 1)

Arsitektur: Chain of Responsibility
  Raw Frame -> DepthProcessing -> YOLODetection -> SensorFusion -> Annotated Output

Setiap stage mengimplementasikan PipelineStage (ABC).
Stage bisa di-enable/disable secara modular.
Kontrak antar stage: FrameData (dataclass).

Dependency:
  - obstacle_detector.ObstacleDetector  (existing, Role 3 & 4)
  - detection_config.DetectionConfig    (existing, Role 1)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from Vision.inc.detection_config import DetectionConfig
from Vision.src.obstacle_detector import ObstacleDetector

try:
    from Vision.src.yolowrapper import YOLOWrapper
except ImportError:
    YOLOWrapper = None  # type: ignore[misc,assignment]

from Vision.inc.logging_config import get_logger

_logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FrameData — struktur data yang mengalir antar stage
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FrameData:
    """Struktur data tunggal yang membawa semua informasi sepanjang pipeline.

    Attributes:
        rgb_frame:      Frame RGB/BGR dari kamera (numpy uint8, HxWx3).
                        DIISI OLEH: CameraThread.

        depth_frame:    Frame depth mentah (numpy uint16, HxW). None jika webcam.
                        DIISI OLEH: CameraThread.

        depth_colormap: Visualisasi depth sebagai BGR berwarna (numpy uint8, HxWx3).
                        DIISI OLEH: DepthProcessingStage (R3).

        depth_scale:    Faktor konversi depth_raw ke meter (default 0.001).

        obstacles:      Daftar obstacle dari depth processing.
                        DIISI OLEH: DepthProcessingStage (R3).
                        DIKONSUMSI OLEH: FusionStage (R4).

                        Format per obstacle:
                        {
                            "bbox":        [x, y, w, h],    # bounding box di frame
                            "distance_m":  float,            # jarak dalam meter
                            "zone":        "left"|"center"|"right",  # sektor horizontal
                            "area_px":     int,              # luas kontur dalam pixel
                        }

        detections:     Daftar deteksi objek dari YOLO.
                        DIISI OLEH: YOLODetectionStage (R2).
                        DIKONSUMSI OLEH: FusionStage (R4).

                        Format: List[Detection] (dataclass dari yolowrapper.py)
                        {
                            "bbox":        [x1, y1, x2, y2],  # format xyxy
                            "class_id":    int,                # indeks kelas COCO
                            "class_name":  str,                # nama kelas (e.g. "person")
                            "confidence":  float,              # 0.0 - 1.0
                        }

        fused_output:   Hasil fusi RGB+Depth.
                        DIISI OLEH: FusionStage (R4).
                        DIKONSUMSI OLEH: GUI Console (R6).

                        Format per item:
                        {
                            "object_class":  str,               # nama kelas
                            "distance_m":    float,             # jarak dalam meter
                            "zone":          "left"|"center"|"right",
                            "priority":      int,               # 0=tertinggi
                            "bbox":          [x1, y1, x2, y2],
                            "action":        str | None,        # rekomendasi aksi
                        }

        metadata:       Informasi tambahan (timestamp, FPS, status).
    """

    rgb_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    depth_frame_raw: Optional[np.ndarray] = None  # Unfiltered depth (for depth model)
    depth_colormap: Optional[np.ndarray] = None
    depth_colormap_raw: Optional[np.ndarray] = None  # Unfiltered colormap (for depth model)
    depth_scale: float = 0.001

    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    detections: List[Any] = field(default_factory=list)
    fused_output: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def has_depth(self) -> bool:
        return self.depth_frame is not None


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineStage — Abstract Base Class untuk semua stage
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineStage(ABC):
    """Kontrak untuk setiap stage dalam pipeline.

    Subclass cukup implementasikan .process().
    Setiap stage menerima FrameData dan mengembalikan FrameData
    (bisa objek yang sama atau objek baru).
    """

    def __init__(self, name: str = "") -> None:
        self.name: str = name or self.__class__.__name__
        self._enabled: bool = True
        self._last_latency_ms: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def last_latency_ms(self) -> float:
        """Latency eksekusi terakhir dalam milidetik (untuk benchmark)."""
        return self._last_latency_ms

    @abstractmethod
    def process(self, data: FrameData) -> FrameData:
        """Proses frame data. Harus di-override oleh subclass."""
        ...

    def _measure(self, data: FrameData) -> FrameData:
        """Wrapper internal: mengukur latency dan memanggil process()."""
        if not self._enabled:
            return data

        t0 = time.perf_counter()
        try:
            result = self.process(data)
        except Exception as e:
            _logger.error(f"Stage {self.name} failed: {e}")
            data.errors.append(f"{self.name} failed: {e}")
            result = data
            
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DepthProcessingStage — stage pertama dalam pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class DepthProcessingStage(PipelineStage):
    """Stage pemrosesan depth: colormap (Merah/Kuning/Hijau) + multi-zone detection.

    Menghasilkan:
        FrameData.depth_colormap — visualisasi zona bahaya
        FrameData.obstacles — daftar obstacle terstruktur untuk Role 4
    """

    def __init__(
        self,
        config: DetectionConfig,
        depth_min_m: float = 0.30,
        depth_max_m: float = 5.00,
    ) -> None:
        super().__init__("DepthProcessingStage")
        self._config = config
        self._depth_min_m = depth_min_m
        self._depth_max_m = depth_max_m

        # Threshold bahaya untuk pewarnaan dan navigasi
        self.danger_threshold = 1.0
        self.warning_threshold = 3.0

        # Gunakan ObstacleDetector yang baru, tapi dibatasi hanya untuk
        # "Immediate Collision Fallback" (max 1.5 meter, area besar)
        self._detector = ObstacleDetector(
            max_distance_m=min(1.5, depth_max_m),
            min_distance_m=depth_min_m,
            min_area=3000,
        )

        # Pre-compute depth-to-color LUT (256 entries) for fast colormap generation
        self._build_depth_lut()

    def _build_depth_lut(self) -> None:
        """Pre-compute 256-entry BGR LUT: index = depth_m * 255 / max_distance."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        max_m = self._depth_max_m if self._depth_max_m > 0 else 5.0
        for i in range(256):
            depth_m = (i / 255.0) * max_m
            if depth_m < self._depth_min_m or depth_m > self._depth_max_m:
                lut[i] = (0, 0, 0)        # Black = invalid
            elif depth_m < self.danger_threshold:
                lut[i] = (0, 0, 255)       # Red = danger
            elif depth_m < self.warning_threshold:
                lut[i] = (0, 255, 255)     # Yellow = warning
            else:
                lut[i] = (0, 255, 0)       # Green = safe
        self._depth_lut = lut
        self._depth_lut_scale = 255.0 / max_m

    def _depth_to_colormap(self, depth_frame: np.ndarray) -> np.ndarray:
        """Fast depth→BGR colormap via pre-computed LUT. ~3x faster than mask approach."""
        depth_m = depth_frame.astype(np.float32) * self._depth_scale
        # Map depth_m (0..max_m) → index (0..255)
        idx = np.clip(depth_m * self._depth_lut_scale, 0, 255).astype(np.uint8)
        return self._depth_lut[idx]

    def set_thresholds(self, depth_min_m: float, depth_max_m: float) -> None:
        """Update threshold depth (dipanggil dari GUI)."""
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return
        self._depth_min_m = depth_min_m
        self._depth_max_m = depth_max_m
        # Tetap batasi obstacle fallback ke 1.5m maksimal
        self._detector.max_distance_m = min(1.5, depth_max_m)
        self._build_depth_lut()

    def set_action_thresholds(self, warning: float, danger: float) -> None:
        """Update threshold aksi (dipanggil dari GUI)."""
        self.warning_threshold = warning
        self.danger_threshold = danger
        self._build_depth_lut()

    def process(self, data: FrameData) -> FrameData:
        if not data.has_depth():
            return data

        self._depth_scale = data.depth_scale
        height, width = data.depth_frame.shape

        # 1. Depth Colormap via pre-computed LUT (~3x faster than mask approach)
        data.depth_colormap = self._depth_to_colormap(data.depth_frame)

        # 2. Unfiltered colormap (for depth model — trained on raw depth)
        if data.depth_frame_raw is not None:
            data.depth_colormap_raw = self._depth_to_colormap(data.depth_frame_raw)

        _, obstacles_list = self._detector.detect(
            data.rgb_frame,
            data.depth_frame,
            data.depth_scale,
            self.danger_threshold,
            self.warning_threshold,
        )

        # Menghapus label "TODO(R3)" dan mengisi data asli dari detektor!
        data.obstacles = obstacles_list

        return data


# ═══════════════════════════════════════════════════════════════════════════════
# YOLODetectionStage — Role 2 (YOLOv8 Specialist)
# ═══════════════════════════════════════════════════════════════════════════════


class YOLODetectionStage(PipelineStage):
    """Stage deteksi objek menggunakan YOLOv8.

    Kontrak input:   FrameData.rgb_frame (numpy BGR uint8)
    Kontrak output:  FrameData.detections (List[Dict])

    Format per detection:
        {
            "bbox":        [x1, y1, x2, y2],  # format xyxy, int
            "class_id":    int,                # indeks kelas
            "class_name":  str,                # contoh: "person", "mobil"
            "confidence":  float,              # 0.0 sampai 1.0
        }
    """

    def __init__(self, model_path: str = "yolov8n.pt", depth_model_path: str = None, conf_threshold: float = 0.25, input_size: int = 320) -> None:
        super().__init__("YOLODetectionStage")
        self._model_path = model_path
        self._depth_model_path = depth_model_path
        self._conf_threshold = conf_threshold
        self._input_size = input_size
        self._wrapper_rgb = None
        self._wrapper_depth = None
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        if YOLOWrapper is not None and model_path:
            try:
                self._wrapper_rgb = YOLOWrapper(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    input_size=input_size,
                )
            except Exception as e:
                _logger.warning(f"YOLO RGB model failed to load: {e}")

        # Depth model is lazy-loaded on first dark frame (saves VRAM at startup)

    def _enhance_dark_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to dark frames to improve YOLO detection in low light."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def process(self, data: FrameData) -> FrameData:
        # Detect low light (always, even if YOLO model unavailable)
        brightness = np.mean(data.rgb_frame)
        rgb_confidence = min(brightness / 128.0, 1.0)
        is_dark = brightness < 40

        # Store confidence metadata for FusionStage
        data.metadata["rgb_confidence"] = rgb_confidence
        data.metadata["is_dark"] = is_dark

        # Decide which model to use
        if is_dark and self._depth_model_path and self._wrapper_depth is None:
            # Lazy-load depth model on first dark frame
            try:
                self._wrapper_depth = YOLOWrapper(
                    model_path=self._depth_model_path,
                    conf_threshold=self._conf_threshold,
                    input_size=self._input_size,
                )
                _logger.info(f"YOLO depth model lazy-loaded: {self._depth_model_path}")
            except Exception as e:
                _logger.warning(f"YOLO depth model failed to lazy-load: {e}")
                self._depth_model_path = None  # Don't retry

        if is_dark and self._wrapper_depth is not None and data.depth_colormap_raw is not None:
            # Dark mode: run YOLO on unfiltered depth colormap (model was trained on raw depth)
            detections = self._wrapper_depth.detect(data.depth_colormap_raw)
            data.metadata["active_model"] = "depth"
            _logger.debug("Using DEPTH model (dark mode, unfiltered)")
        elif is_dark and self._wrapper_depth is not None and data.depth_colormap is not None:
            # Dark mode fallback: no raw depth available, use filtered
            detections = self._wrapper_depth.detect(data.depth_colormap)
            data.metadata["active_model"] = "depth_filtered"
            _logger.debug("Using DEPTH model (dark mode, filtered fallback)")
        elif self._wrapper_rgb is not None:
            # Normal or dim mode: run YOLO on RGB (with CLAHE if dark)
            frame_for_yolo = self._enhance_dark_frame(data.rgb_frame) if is_dark else data.rgb_frame
            detections = self._wrapper_rgb.detect(frame_for_yolo)
            data.metadata["active_model"] = "rgb_clahe" if is_dark else "rgb"
        else:
            detections = []
            data.metadata["active_model"] = "none"

        # Gunakan dataclass secara langsung
        data.detections = detections

        return data


# ═══════════════════════════════════════════════════════════════════════════════
# FusionStage — placeholder untuk Role 4
# ═══════════════════════════════════════════════════════════════════════════════


class FusionStage(PipelineStage):
    """Stage fusi RGB (YOLO) + Depth.

    Menggabungkan deteksi YOLO (class_name) dengan obstacle depth (distance_m)
    menggunakan overlap ratio untuk mencocokkan blob depth ke object YOLO.

    Kontrak input:   FrameData.detections (dari R2) + FrameData.obstacles (dari R3)
    Kontrak output:  FrameData.fused_output (List[Dict])

    Format per item:
        {
            "object_class":  str,               # dari R2 (e.g. "person")
            "distance_m":    float,             # dari R3 depth (meter)
            "zone":          "left"|"center"|"right",
            "priority":      int,               # 0 = paling bahaya (person dekat)
            "bbox":          [x1, y1, x2, y2],  # bounding box final (xyxy)
            "action":        str | None,         # "STOP" atau None
        }

    Aturan prioritas:
        - Person dalam jarak < danger_distance  -> priority 0 (STOP)
        - Obstacle dalam jarak < danger_distance -> priority 1
        - Person dalam jarak < 3m               -> priority 2
        - Lainnya                                -> priority 3+
        - Jika tidak ada obstacle                -> list kosong []
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        super().__init__("FusionStage")
        self._config = config

    def _calculate_overlap_ratio(
        self,
        depth_box: List[int],
        yolo_box: List[int],
        depth_area_px: Optional[int] = None,
    ) -> float:
        """Hitung overlap antara depth_box dan yolo_box.

        Menggunakan area terkecil (min) sebagai denominator agar:
        - YOLO box kecil di dalam depth blob besar → overlap tinggi
        - Depth blob kecil di dalam YOLO box besar → overlap tinggi

        Args:
            depth_box:     [x1, y1, x2, y2] depth obstacle bbox
            yolo_box:      [x1, y1, x2, y2] YOLO detection bbox
            depth_area_px: luas aktual kontur depth (dari ObstacleDetector.area_px)
        """
        xA = max(depth_box[0], yolo_box[0])
        yA = max(depth_box[1], yolo_box[1])
        xB = min(depth_box[2], yolo_box[2])
        yB = min(depth_box[3], yolo_box[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        # Area depth box (gunakan contour area jika tersedia)
        if depth_area_px and depth_area_px > 0:
            depthBoxArea = depth_area_px
        else:
            depthBoxArea = (depth_box[2] - depth_box[0]) * (depth_box[3] - depth_box[1])

        # Area YOLO box
        yoloBoxArea = (yolo_box[2] - yolo_box[0]) * (yolo_box[3] - yolo_box[1])

        # Gunakan area terkecil sebagai denominator
        minArea = min(depthBoxArea, yoloBoxArea)
        if minArea == 0:
            return 0.0

        return interArea / float(minArea)

    def _sample_depth_in_bbox(
        self,
        depth_frame: np.ndarray,
        depth_scale: float,
        bbox: List[int],
        min_distance_m: float = 0.3,
        max_distance_m: float = 5.0,
    ) -> Optional[float]:
        """Langsung sampling kedalaman dari depth frame di dalam YOLO bbox.

        Menggunakan region tengah 60% dari bbox untuk menghindari
        piksel latar belakang di tepi bounding box.

        Returns:
            Jarak dalam meter (25th percentile), atau None jika tidak ada depth valid.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_frame.shape[:2]

        # Clamp ke batas frame
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        # Gunakan region tengah 60% untuk menghindari piksel tepi
        bw, bh = x2 - x1, y2 - y1
        margin_x = int(bw * 0.2)
        margin_y = int(bh * 0.2)
        cx1 = x1 + margin_x
        cy1 = y1 + margin_y
        cx2 = x2 - margin_x
        cy2 = y2 - margin_y

        # Fallback ke full bbox jika region terlalu kecil
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2

        region = depth_frame[cy1:cy2, cx1:cx2].astype(np.float32) * depth_scale
        valid = region[(region >= min_distance_m) & (region <= max_distance_m)]

        if valid.size == 0:
            return None

        # 25th percentile — lebih stabil dari min, lebih akurat dari median
        return float(np.percentile(valid, 25))

    def _determine_zone(self, bbox: List[int], frame_width: int) -> str:
        """Tentukan zona horizontal (left/center/right) dari bounding box."""
        center_x = (bbox[0] + bbox[2]) // 2
        zone_width = frame_width // 3
        if center_x < zone_width:
            return "left"
        elif center_x < zone_width * 2:
            return "center"
        return "right"

    def process(self, data: FrameData) -> FrameData:
        fused_results = []

        # Ambil threshold dari config
        danger_dist = self._config.danger_distance if self._config else 1.0
        warning_dist = self._config.warning_distance if self._config else 3.0

        # Adaptive confidence untuk dark mode
        rgb_confidence = data.metadata.get("rgb_confidence", 1.0)
        is_dark = data.metadata.get("is_dark", False)
        overlap_threshold = 0.3 if (is_dark or rgb_confidence < 0.5) else 0.5

        has_depth = data.depth_frame is not None
        frame_width = data.rgb_frame.shape[1] if data.rgb_frame is not None else 640

        # Track YOLO boxes yang sudah di-fuse (untuk avoid double-counting)
        matched_yolo_indices = set()

        # ── PASS 1: YOLO-first — setiap deteksi YOLO langsung sampling depth ──
        if not is_dark:
            for i, det in enumerate(data.detections):
                yolo_bbox = det.bbox  # [x1, y1, x2, y2]
                class_name = det.class_name

                # Sampling depth langsung dari depth frame
                dist = None
                if has_depth:
                    dist = self._sample_depth_in_bbox(
                        data.depth_frame, data.depth_scale, yolo_bbox
                    )

                if dist is None:
                    continue  # Skip YOLO detection tanpa depth data valid

                matched_yolo_indices.add(i)

                zone = self._determine_zone(yolo_bbox, frame_width)

                # Safety Matrix Priority
                priority = 3
                if class_name == "person":
                    if dist < danger_dist:
                        priority = 0
                    elif dist < warning_dist:
                        priority = 2
                else:
                    if dist < danger_dist:
                        priority = 1
                    else:
                        priority = 3

                fused_results.append({
                    "object_class": class_name,
                    "distance_m": dist,
                    "zone": zone,
                    "priority": priority,
                    "bbox": list(yolo_bbox),
                    "action": "STOP" if priority == 0 else None,
                })

        # ── PASS 2: Depth-only obstacles — untuk hal yang YOLO tidak deteksi ──
        for obs in data.obstacles:
            x, y, w, h = obs["bbox"]
            obs_box = [x, y, x + w, y + h]
            dist = obs["distance_m"]
            zone = obs["zone"]
            area_px = obs.get("area_px")

            # Cek apakah obstacle ini sudah ter-cover oleh YOLO detection
            already_covered = False
            for i, det in enumerate(data.detections):
                if i in matched_yolo_indices:
                    overlap = self._calculate_overlap_ratio(
                        obs_box, det.bbox, depth_area_px=area_px
                    )
                    if overlap > overlap_threshold:
                        already_covered = True
                        break

            if already_covered:
                continue  # Sudah di-cover oleh YOLO detection di PASS 1

            # Fallback distance limit: abaikan obstacle generik > 1.5m
            if dist > 1.5:
                continue

            # Obstacle depth-only (YOLO tidak mengenali)
            # Demote priority: jangan spam DANGER kecuali sangat dekat
            if dist < 0.5:
                priority = 1
            elif dist < 1.0:
                priority = 2
            else:
                priority = 3

            fused_results.append({
                "object_class": "obstacle",
                "distance_m": dist,
                "zone": zone,
                "priority": priority,
                "bbox": [x, y, x + w, y + h],
                "action": None,
            })

        data.fused_output = fused_results

        return data

# ═══════════════════════════════════════════════════════════════════════════════
# VisualAnnotationStage — Role 5 (Rendering)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualAnnotationStage(PipelineStage):
    """Stage terakhir untuk menggambar bounding box dan HUD status ke rgb_frame.
    
    Kontrak:
    - Membaca FrameData.fused_output (atau obstacles jika fusion kosong).
    - Memodifikasi FrameData.rgb_frame in-place.
    """

    def __init__(self, config: DetectionConfig = None) -> None:
        super().__init__("VisualAnnotationStage")
        self._config = config or DetectionConfig()

    def process(self, data: FrameData) -> FrameData:
        if data.rgb_frame is None:
            return data

        # Pilih sumber data (Fusion prioritas utama, fallback ke obstacles raw, fallback terakhir YOLO)
        items_to_draw = data.fused_output if data.fused_output else data.obstacles
        source_is_xyxy = bool(data.fused_output)
        
        # Mode fallback YOLO only jika depth tidak aktif
        yolo_fallback_mode = False
        if not items_to_draw and data.detections:
            # Konversi YOLO detections ke dict format
            items_to_draw = []
            yolo_fallback_mode = True
            source_is_xyxy = True
            for det in data.detections:
                items_to_draw.append({
                    "bbox": det.bbox, # xyxy
                    "object_class": det.class_name,
                    "priority": 3.0,
                    "distance_m": 99.0, # Unknown
                    "zone": "center"
                })

        global_status = "SAFE"
        
        # 1. Gambar Bounding Boxes
        for item in items_to_draw:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            
            # Normalize to xywh for drawing
            if source_is_xyxy:
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                x, y, w, h = bbox

            distance = item.get("distance_m", 99.0)
            priority = item.get("priority", 3.0)
            zone_str = item.get("zone", "center")
            
            # Tentukan warna dan status global
            if priority <= 1.0 or distance < self._config.danger_distance:
                color = (60, 60, 255)  # Soft Red
                global_status = "DANGER"
            elif priority <= 2.0 or distance < self._config.warning_distance:
                color = (0, 165, 255)  # Amber
                if global_status != "DANGER":
                    global_status = "WARN"
            else:
                color = (50, 205, 50)  # Lime Green

            label = f"[{zone_str.upper()[0]}] {distance:.2f}m"
            if "object_class" in item:
                label = f"{item['object_class']} {label}"

            # Gambar HUD Corner Brackets
            bracket_len = max(5, min(w, h, 40) // 4)
            thick = 3
            cv2.line(data.rgb_frame, (x, y), (x + bracket_len, y), color, thick)
            cv2.line(data.rgb_frame, (x, y), (x, y + bracket_len), color, thick)
            cv2.line(data.rgb_frame, (x + w, y), (x + w - bracket_len, y), color, thick)
            cv2.line(data.rgb_frame, (x + w, y), (x + w, y + bracket_len), color, thick)
            cv2.line(data.rgb_frame, (x, y + h), (x + bracket_len, y + h), color, thick)
            cv2.line(data.rgb_frame, (x, y + h), (x, y + h - bracket_len), color, thick)
            cv2.line(data.rgb_frame, (x + w, y + h), (x + w - bracket_len, y + h), color, thick)
            cv2.line(data.rgb_frame, (x + w, y + h), (x + w, y + h - bracket_len), color, thick)

            # Teks Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
            text_y = max(y - 8, text_h + 4)
            
            cv2.rectangle(data.rgb_frame, (x, text_y - text_h - 4), (x + text_w, text_y + 4), (30, 30, 30), -1)
            cv2.putText(data.rgb_frame, label, (x, text_y), font, scale, color, thickness, cv2.LINE_AA)

        # 2. Gambar Status Global HUD
        if global_status == "DANGER":
            status_color = (60, 60, 255)
        elif global_status == "WARN":
            status_color = (0, 165, 255)
        else:
            status_color = (50, 205, 50)

        status_text = f"SYS: {global_status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(status_text, font, scale, thickness)
        sx, sy = 25, 45
        
        cv2.rectangle(data.rgb_frame, (sx - 8, sy - text_h - 10), (sx + text_w + 8, sy + 10), (30, 30, 30), -1)
        cv2.rectangle(data.rgb_frame, (sx - 8, sy - text_h - 10), (sx + text_w + 8, sy + 10), status_color, 1)
        cv2.putText(data.rgb_frame, status_text, (sx, sy), font, scale, status_color, thickness, cv2.LINE_AA)

        return data



# ═══════════════════════════════════════════════════════════════════════════════
# FrameProcessor — orkestrator pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class FrameProcessor:
    """Orkestrator pipeline vision.

    Menerima raw frame (numpy), menjalankan rantai stage yang aktif,
    dan mengembalikan FrameData yang sudah diproses.

    Usage:
        config = DetectionConfig()
        processor = FrameProcessor(config)

        # Later, when Role 2 and 4 are ready:
        # processor.add_stage(YOLODetectionStage("yolov8n.pt"))
        # processor.add_stage(FusionStage())

        while camera_running:
            rgb, depth = get_frames()
            result = processor.process(rgb, depth, depth_scale=0.001)
            # result.rgb_frame sudah teranotasi
            # result.depth_colormap siap ditampilkan
            # result.obstacles berisi daftar obstacle
    """

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._stages: List[PipelineStage] = []

        # Stage default: depth processing (selalu aktif)
        self._depth_stage = DepthProcessingStage(config)
        self.add_stage(self._depth_stage)

    # ── stage management ──────────────────────────────────────────────────

    def add_stage(self, stage: PipelineStage) -> None:
        """Tambah stage ke pipeline (dipanggil oleh Role 2, 4, dsb)."""
        self._stages.append(stage)

    def remove_stage(self, stage_name: str) -> bool:
        """Hapus stage berdasarkan nama. Return True jika berhasil."""
        for i, s in enumerate(self._stages):
            if s.name == stage_name:
                self._stages.pop(i)
                return True
        return False

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Cari stage berdasarkan nama."""
        for s in self._stages:
            if s.name == name:
                return s
        return None

    def set_stage_enabled(self, name: str, enabled: bool) -> bool:
        """Enable/disable stage berdasarkan nama."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = enabled
            return True
        return False

    @property
    def stages(self) -> List[PipelineStage]:
        return list(self._stages)

    # ── threshold management ──────────────────────────────────────────────

    def set_depth_thresholds(self, depth_min_m: float, depth_max_m: float) -> None:
        """Update threshold depth untuk depth stage (dipanggil dari GUI)."""
        self._depth_stage.set_thresholds(depth_min_m, depth_max_m)

    def set_action_thresholds(self, warning: float, danger: float) -> None:
        """Update action thresholds untuk bahaya (dipanggil dari GUI)."""
        self._depth_stage.set_action_thresholds(warning, danger)
        # Update FusionStage thresholds
        fusion_stage = self.get_stage("FusionStage")
        if fusion_stage and hasattr(fusion_stage, "set_action_thresholds"):
            fusion_stage.set_action_thresholds(warning, danger)
        # Update VisualAnnotationStage thresholds (OpenCV view colors)
        annotation_stage = self.get_stage("VisualAnnotationStage")
        if annotation_stage:
            annotation_stage._config.danger_distance = danger
            annotation_stage._config.warning_distance = warning

    # ── main processing ───────────────────────────────────────────────────

    def process(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        depth_scale: float = 0.001,
        depth_frame_raw: Optional[np.ndarray] = None,
    ) -> FrameData:
        """Jalankan seluruh pipeline pada satu pasangan frame.

        Args:
            rgb_frame:      Frame RGB/BGR dari kamera.
            depth_frame:    Frame depth filtered (None jika webcam).
            depth_scale:    Konversi depth ke meter.
            depth_frame_raw: Frame depth unfiltered (untuk depth model).

        Returns:
            FrameData yang sudah diproses oleh semua stage aktif.
        """
        data = FrameData(
            rgb_frame=rgb_frame,
            depth_frame=depth_frame,
            depth_frame_raw=depth_frame_raw,
            depth_scale=depth_scale,
            metadata={"timestamp": time.time()},
        )

        for stage in self._stages:
            data = stage._measure(data)

        return data

    # ── benchmarking ──────────────────────────────────────────────────────

    def get_latency_report(self) -> Dict[str, float]:
        """Kembalikan laporan latency per stage (untuk Role 5)."""
        return {s.name: s.last_latency_ms for s in self._stages}
