"""
Vision/src/yolowrapper.py — YOLOv8 Object Detection Wrapper.

Role 2 (Husein) — YOLOv8 Specialist.

Wraps ultralytics YOLO model for inference. Outputs detection results
in the contract format agreed with R1 (Moris) and R4 (Rasyid).
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

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Detection output contract (compatible with FrameData.detections)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [xmin, ymin, xmax, ymax]


class YOLOWrapper:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, input_size: int = 416):
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        self._device = "0" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading YOLO model from: {model_path} (device: {self._device})")
        self.model = YOLO(model_path)
        self.class_mapping = self.model.names
        logger.info(f"YOLO model loaded. Classes: {len(self.class_mapping)} | GPU: {torch.cuda.is_available()}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []

        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            device=self._device,
            half=True if self._device != "cpu" else False,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_mapping.get(class_id, f"unknown-{class_id}")

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox=xyxy,
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
            wrapper = YOLOWrapper(model_path=MODEL_PATH, conf_threshold=0.25, input_size=416)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            print("\n[TEST 1] Warm-up inference...")
            wrapper.detect(dummy_frame)

            print("\n[TEST 2] Latency benchmark (10 iterations)...")
            total_latency = 0
            for i in range(10):
                start = time.time()
                detections = wrapper.detect(dummy_frame)
                latency = (time.time() - start) * 1000
                total_latency += latency
                print(f"   Loop {i+1}: {latency:.2f} ms | detections: {len(detections)}")

            avg = total_latency / 10
            print(f"\n[RESULT] Average latency: {avg:.2f} ms")
            if avg <= 50:
                print("STATUS: PASS (GPU target <= 50ms)")
            elif avg <= 100:
                print("STATUS: OK (CPU target <= 100ms)")
            else:
                print("STATUS: TOO SLOW (> 100ms)")

        except Exception as e:
            print(f"[ERROR] {e}")
