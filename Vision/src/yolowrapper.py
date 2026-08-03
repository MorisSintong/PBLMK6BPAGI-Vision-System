"""
Vision/src/yolowrapper.py — YOLOv8 Object Detection Wrapper.

Role 2 (Husein) — YOLOv8 Specialist.

Wraps ultralytics YOLO model for inference. Outputs detection results
in the contract format agreed with R1 (Moris) and R4 (Rasyid).

Features:
  - Auto-detects CUDA via torch.cuda.is_available(); falls back to CPU
  - FP16 inference when GPU is available (half=True on Tensor Cores)
  - GPU warm-up: dummy inference on load to pre-compile CUDA kernels
  - Reduced input size (default 320) for ~40% fewer pixels vs 416
  - Batch tensor transfer: one .cpu().numpy() call for all boxes
  - Output: List[Detection] dataclass with class_id, class_name, confidence, bbox
"""

from dataclasses import dataclass
from typing import List
import time
import os
import warnings

import numpy as np
import torch

# Suppress ultralytics' torch.load FutureWarning (upstream issue, not ours)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

from ultralytics import YOLO

from Vision.inc.logging_config import get_logger

logger = get_logger(__name__)

# Expected class contract for the security robot pipeline.
# Both ModelRGB_V4.2.pt and ModelDepth_V4.pt are trained on these 3 classes
# and must expose exactly this name set so down-stream stages (FusionStage,
# AlertPanel) can map class_id -> semantic label reliably.
EXPECTED_CLASS_NAMES = ("mobil", "motor", "person")

# Tasks whose output includes a `.boxes` attribute compatible with detect().
# YOLOv8/v11 detect + segment both expose boxes; pose exposes keypoints+boxes
# (boxes are present but the model is not our intended contract) and classify
# has NO boxes at all and is therefore rejected outright.
SUPPORTED_TASKS = ("detect", "segment")


class ModelValidationError(Exception):
    """Raised when a loaded YOLO weight file does not match the pipeline contract."""


@dataclass
class Detection:
    """Detection output contract (compatible with FrameData.detections)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [xmin, ymin, xmax, ymax]


class YOLOWrapper:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, input_size: int = 320):
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        if not os.path.exists(model_path):
            raise ModelValidationError(f"Model file not found: {model_path}")

        self._has_gpu = torch.cuda.is_available()
        self._device = "0" if self._has_gpu else "cpu"
        self._fp16 = self._has_gpu and self._device != "cpu"

        logger.info(f"Loading YOLO model from: {model_path} (device: {self._device}, fp16: {self._fp16})")
        try:
            self.model = YOLO(model_path)
        except Exception as e:  # wrong architecture for this ultralytics version, corrupt file, etc.
            raise ModelValidationError(
                f"Failed to load YOLO model '{model_path}': {e}. "
                f"The weight file may target a newer/different YOLO architecture "
                f"(e.g. YOLOv26) than the installed ultralytics {__import__('ultralytics').__version__}."
            ) from e

        self._validate_model_contract(model_path)

        self.class_mapping = self.model.names

        # Warm-up: run one dummy inference to pre-compile CUDA kernels
        if self._has_gpu:
            logger.info("Warming up GPU (first inference is slow)...")
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            try:
                self.model.predict(source=dummy, imgsz=self.input_size, verbose=False)
                torch.cuda.synchronize()
                logger.info("GPU warm-up complete.")
            except Exception as e:
                logger.warning(f"GPU warm-up failed (will retry on real frames): {e}")
                self._has_gpu = False
                self._device = "cpu"
                self._fp16 = False

        logger.info(f"YOLO model ready. Classes: {len(self.class_mapping)} | GPU: {self._has_gpu}")

    def _validate_model_contract(self, model_path: str) -> None:
        """Fail loudly if the loaded weight file does not match the pipeline contract.

        Guards against three silent-failure modes:
          1. A weight trained for an unsupported task (e.g. classify has no .boxes,
             which would make detect() return [] for every frame with no error).
          2. A weight whose class set differs from EXPECTED_CLASS_NAMES, which would
             desync FusionStage / AlertPanel label mapping.
          3. A weight targeting a newer architecture (e.g. YOLOv26) that this
             ultralytics version cannot even parse — surfaced here instead of at
             first predict() call mid-run.
        """
        task = getattr(self.model, "task", None)
        if task is None:
            raise ModelValidationError(
                f"Cannot determine task for model '{model_path}'. "
                f"Expected one of {SUPPORTED_TASKS}."
            )
        if task not in SUPPORTED_TASKS:
            raise ModelValidationError(
                f"Model '{model_path}' has task='{task}', which is not supported. "
                f"Expected one of {SUPPORTED_TASKS} (a '{task}' model exposes no "
                f"compatible bounding boxes for this pipeline)."
            )

        names = list(getattr(self.model, "names", {}).values())
        if sorted(names) != sorted(EXPECTED_CLASS_NAMES):
            raise ModelValidationError(
                f"Model '{model_path}' class set {names} does not match the expected "
                f"pipeline contract {list(EXPECTED_CLASS_NAMES)}. Class-id to label "
                f"mapping in FusionStage/AlertPanel would be incorrect."
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            device=self._device,
            half=self._fp16,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Batch tensor transfer: move all at once instead of per-box
        xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy().astype(int)

        detections = []
        for i in range(len(boxes)):
            detections.append(Detection(
                class_id=int(cls_all[i]),
                class_name=self.class_mapping.get(int(cls_all[i]), f"unknown-{cls_all[i]}"),
                confidence=float(conf_all[i]),
                bbox=xyxy_all[i].tolist(),
            ))

        return detections


if __name__ == "__main__":
    print("\n=== YOLOv8 Wrapper Standalone Test ===\n")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models", "security_best.pt"))

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Download 'security_best.pt' to Vision/models/")
    else:
        try:
            wrapper = YOLOWrapper(model_path=MODEL_PATH, conf_threshold=0.25, input_size=320)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            print("\n[TEST 1] Warm-up inference...")
            wrapper.detect(dummy_frame)

            print("\n[TEST 2] Latency benchmark (20 iterations)...")
            total_latency = 0
            for i in range(20):
                start = time.time()
                detections = wrapper.detect(dummy_frame)
                latency = (time.time() - start) * 1000
                total_latency += latency
                print(f"   Loop {i+1}: {latency:.2f} ms | detections: {len(detections)}")

            avg = total_latency / 20
            print(f"\n[RESULT] Average latency: {avg:.2f} ms")
            if avg <= 30:
                print("STATUS: EXCELLENT (target <= 30ms)")
            elif avg <= 50:
                print("STATUS: PASS (target <= 50ms)")
            elif avg <= 100:
                print("STATUS: OK (CPU target <= 100ms)")
            else:
                print("STATUS: TOO SLOW (> 100ms)")

        except Exception as e:
            print(f"[ERROR] {e}")
