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

# ═══════════════════════════════════════════════════════════════════════════════
# FrameData — struktur data yang mengalir antar stage
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FrameData:
    """Struktur data tunggal yang membawa semua informasi sepanjang pipeline.

    Attributes:
        rgb_frame:      Frame RGB mentah dari kamera (numpy BGR, uint8).
        depth_frame:    Frame depth mentah dalam satuan asli kamera (uint16).
        depth_colormap: Visualisasi depth sebagai gambar BGR berwarna.
        depth_scale:    Faktor konversi depth_raw -> meter (default 0.001).
        obstacles:      Daftar obstacle yang terdeteksi oleh depth stage.
        detections:     Daftar deteksi dari YOLO stage (diisi Role 2).
        fused_output:   Hasil fusi RGB+Depth (diisi Role 4).
        metadata:       Informasi tambahan (timestamp, FPS, status).
    """

    rgb_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    depth_colormap: Optional[np.ndarray] = None
    depth_scale: float = 0.001

    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    fused_output: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

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
        result = self.process(data)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DepthProcessingStage — stage pertama dalam pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class DepthProcessingStage(PipelineStage):
    """Stage pemrosesan depth: colormap + obstacle detection.

    Menggunakan ObstacleDetector yang sudah ada.
    Logika depth-to-colormap direplikasi dari CameraThread._depth_to_colormap()
    agar pipeline self-contained (tidak bergantung pada CameraThread).
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

        # Gunakan ObstacleDetector yang sudah ada
        self._detector = ObstacleDetector(
            max_distance_m=depth_max_m,
            min_distance_m=config.min_distance,
            min_area=800,
            roi_ratio=0.7,
        )

    def set_thresholds(self, depth_min_m: float, depth_max_m: float) -> None:
        """Update threshold depth (dipanggil dari GUI)."""
        if depth_min_m <= 0 or depth_max_m <= 0 or depth_min_m >= depth_max_m:
            return
        self._depth_min_m = depth_min_m
        self._depth_max_m = depth_max_m
        self._detector.max_distance_m = depth_max_m

    def process(self, data: FrameData) -> FrameData:
        if not data.has_depth():
            return data

        # 1. Depth colormap (replikasi logika CameraThread)
        depth_m = data.depth_frame.astype(np.float32) * data.depth_scale
        valid_mask = (depth_m >= self._depth_min_m) & (depth_m <= self._depth_max_m)

        normalized = np.zeros_like(depth_m, dtype=np.float32)
        normalized[valid_mask] = (depth_m[valid_mask] - self._depth_min_m) / (
            self._depth_max_m - self._depth_min_m
        )
        depth_8u = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        data.depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
        data.depth_colormap[~valid_mask] = (0, 0, 0)

        # 2. Obstacle detection
        annotated, detected, closest_dist = self._detector.detect(
            data.rgb_frame, data.depth_frame, data.depth_scale
        )

        if annotated is not None:
            data.rgb_frame = annotated

        if detected and closest_dist is not None:
            data.obstacles = [
                {
                    "distance_m": closest_dist,
                    "status": "detected",
                }
            ]

        return data


# ═══════════════════════════════════════════════════════════════════════════════
# YOLODetectionStage — placeholder untuk Role 2
# ═══════════════════════════════════════════════════════════════════════════════


class YOLODetectionStage(PipelineStage):
    """Stage deteksi objek menggunakan YOLOv8.

    PLACEHOLDER — akan diimplementasikan oleh Role 2 (YOLOv8 Specialist).

    Kontrak input:  FrameData.rgb_frame
    Kontrak output: FrameData.detections  (List[Dict])
                    Format per detection: {bbox, class_id, class_name, confidence}
    """

    def __init__(self, model_path: str = "") -> None:
        super().__init__("YOLODetectionStage")
        self._model_path = model_path
        self._model = None  # Akan diisi Role 2

    def process(self, data: FrameData) -> FrameData:
        # Placeholder — tidak mengubah data
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# FusionStage — placeholder untuk Role 4
# ═══════════════════════════════════════════════════════════════════════════════


class FusionStage(PipelineStage):
    """Stage fusi RGB (YOLO) + Depth.

    PLACEHOLDER — akan diimplementasikan oleh Role 4 (Sensor Fusion Engineer).

    Kontrak input:  FrameData.detections + FrameData.depth_frame
    Kontrak output: FrameData.fused_output  (List[Dict])
                    Format per item: {object_class, distance_m, zone, priority, bbox}
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
