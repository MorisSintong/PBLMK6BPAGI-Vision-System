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
from detection_config import DetectionConfig
from obstacle_detector import ObstacleDetector

try:
    from yolowrapper import YOLOWrapper
except ImportError:
    YOLOWrapper = None  # type: ignore[misc,assignment]

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

                        Format per detection:
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
    depth_colormap: Optional[np.ndarray] = None
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
            from logging_config import get_logger
            get_logger(__name__).error(f"Stage {self.name} failed: {e}")
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

        # Gunakan ObstacleDetector yang baru
        self._detector = ObstacleDetector(
            max_distance_m=depth_max_m,
            min_distance_m=depth_min_m,
            min_area=800,
        )

    def set_thresholds(self, depth_min_m: float, depth_max_m: float) -> None:
        """Update threshold depth (dipanggil dari GUI)."""
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return
        self._depth_min_m = depth_min_m
        self._depth_max_m = depth_max_m
        self._detector.max_distance_m = depth_max_m

    def set_action_thresholds(self, warning: float, danger: float) -> None:
        """Update threshold aksi (dipanggil dari GUI)."""
        self.warning_threshold = warning
        self.danger_threshold = danger

    def process(self, data: FrameData) -> FrameData:
        if not data.has_depth():
            return data

        depth_m = data.depth_frame.astype(np.float32) * data.depth_scale
        height, width = depth_m.shape

        # 1. Depth Colormap dengan warna zona bahaya (Merah, Kuning, Hijau)
        depth_colormap = np.zeros((height, width, 3), dtype=np.uint8)

        valid_mask = (depth_m >= self._depth_min_m) & (depth_m <= self._depth_max_m)
        danger_mask = valid_mask & (depth_m < self.danger_threshold)
        warning_mask = (
            valid_mask
            & (depth_m >= self.danger_threshold)
            & (depth_m < self.warning_threshold)
        )
        safe_mask = valid_mask & (depth_m >= self.warning_threshold)

        depth_colormap[danger_mask] = (0, 0, 255)  # Merah
        depth_colormap[warning_mask] = (0, 255, 255)  # Kuning
        depth_colormap[safe_mask] = (0, 255, 0)  # Hijau

        data.depth_colormap = depth_colormap

        # 2. Obstacle detection
        annotated, obstacles_list = self._detector.detect(
            data.rgb_frame,
            data.depth_frame,
            data.depth_scale,
            self.danger_threshold,
            self.warning_threshold,
        )

        if annotated is not None:
            data.rgb_frame = annotated

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

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25, input_size: int = 416) -> None:
        super().__init__("YOLODetectionStage")
        self._model_path = model_path
        self._wrapper = None

        if YOLOWrapper is not None and model_path:
            try:
                self._wrapper = YOLOWrapper(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    input_size=input_size,
                )
            except Exception as e:
                from logging_config import get_logger
                get_logger(__name__).warning(f"YOLO model failed to load: {e}")

    def process(self, data: FrameData) -> FrameData:
        if self._wrapper is None:
            return data

        detections = self._wrapper.detect(data.rgb_frame)

        # Gunakan dataclass secara langsung
        data.detections = detections

        return data


# ═══════════════════════════════════════════════════════════════════════════════
# FusionStage — placeholder untuk Role 4
# ═══════════════════════════════════════════════════════════════════════════════


class FusionStage(PipelineStage):
    """Stage fusi RGB (YOLO) + Depth.

    PLACEHOLDER — akan diimplementasikan oleh Role 4 (Sensor Fusion Engineer).

    Kontrak input:   FrameData.detections (dari R2) + FrameData.obstacles (dari R3)
                     + FrameData.depth_frame
    Kontrak output:  FrameData.fused_output (List[Dict])

    Format per item (WAJIB):
        {
            "object_class":  str,               # dari R2 (e.g. "person")
            "distance_m":    float,             # dari R3 depth (meter)
            "zone":          "left"|"center"|"right",
            "priority":      int,               # 0 = paling bahaya (person dekat)
            "bbox":          [x1, y1, x2, y2],  # bounding box final
            "action":        str | None,         # "STOP", "BELOK KANAN", "BELOK KIRI", None
        }

    Aturan prioritas (WAJIB):
        - Person dalam jarak < 1m  -> priority 0 (STOP)
        - Obstacle dalam jarak < 1m -> priority 1
        - Person dalam jarak < 3m  -> priority 2
        - Lainnya                   -> priority 3+
        - Jika tidak ada obstacle   -> list kosong []
    """

    def __init__(self) -> None:
        super().__init__("FusionStage")

    def process(self, data: FrameData) -> FrameData:
        # Placeholder — tidak mengubah data
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
        # Jika FusionStage juga butuh, pass ke sana
        fusion_stage = self.get_stage("FusionStage")
        if fusion_stage and hasattr(fusion_stage, "set_action_thresholds"):
            fusion_stage.set_action_thresholds(warning, danger)

    # ── main processing ───────────────────────────────────────────────────

    def process(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        depth_scale: float = 0.001,
    ) -> FrameData:
        """Jalankan seluruh pipeline pada satu pasangan frame.

        Args:
            rgb_frame:   Frame RGB/BGR dari kamera.
            depth_frame: Frame depth mentah (None jika webcam).
            depth_scale: Konversi depth ke meter.

        Returns:
            FrameData yang sudah diproses oleh semua stage aktif.
        """
        data = FrameData(
            rgb_frame=rgb_frame,
            depth_frame=depth_frame,
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
