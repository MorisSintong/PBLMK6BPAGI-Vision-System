"""
Tests for YOLOWrapper model-contract validation guard.

These tests verify that YOLOWrapper rejects incompatible weight files
(missing, corrupt, wrong task, or mismatched class set) with a clear
ModelValidationError instead of failing silently mid-run.

Real-model loading is covered by an integration test that only runs when
the model files are present; unit tests use a lightweight fake object so
they execute without a GPU or the actual ultralytics weights.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Vision.src.yolowrapper import (
    EXPECTED_CLASS_NAMES,
    SUPPORTED_TASKS,
    ModelValidationError,
    YOLOWrapper,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake model object + monkeypatched YOLO loader (no real weights / GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeModel:
    """Mimics an ultralytics YOLO model with a configurable task + class names."""

    def __init__(self, task="segment", names=None):
        self.task = task
        self.names = {i: n for i, n in enumerate(names or list(EXPECTED_CLASS_NAMES))}

    def predict(self, **kwargs):
        # Not exercised by the validation tests.
        raise AssertionError("predict should not be called in validation tests")


@pytest.fixture
def patch_yolo(monkeypatch):
    """Patch ultralytics.YOLO to return a controllable fake model."""
    captured = {}

    def _fake_yolo(path):
        spec = captured.get("spec", {})
        if "raise" in spec:
            raise spec["raise"]
        return _FakeModel(task=spec.get("task", "segment"),
                          names=spec.get("names", list(EXPECTED_CLASS_NAMES)))

    monkeypatch.setattr("Vision.src.yolowrapper.YOLO", _fake_yolo)
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# Guard: supported contract passes
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_segment_model_passes(patch_yolo):
    """A segment model with the exact expected classes should construct fine."""
    patch_yolo["spec"] = {"task": "segment", "names": list(EXPECTED_CLASS_NAMES)}
    wrapper = YOLOWrapper.__new__(YOLOWrapper)  # bypass __init__ GPU/warm-up
    wrapper.model = _FakeModel(task="segment", names=list(EXPECTED_CLASS_NAMES))
    # _validate_model_contract should not raise
    wrapper._validate_model_contract("fake.pt")


def test_valid_detect_model_passes(patch_yolo):
    """A detect model with the expected classes is also supported."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="detect", names=list(EXPECTED_CLASS_NAMES))
    wrapper._validate_model_contract("fake.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: unsupported task is rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_task_rejected():
    """A 'classify' model exposes no boxes -> must be rejected."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="classify", names=list(EXPECTED_CLASS_NAMES))
    with pytest.raises(ModelValidationError, match="task='classify'"):
        wrapper._validate_model_contract("fake.pt")


def test_pose_task_rejected():
    """A 'pose' model is not the intended contract -> rejected."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="pose", names=list(EXPECTED_CLASS_NAMES))
    with pytest.raises(ModelValidationError, match="task='pose'"):
        wrapper._validate_model_contract("fake.pt")


def test_unknown_task_rejected():
    """A model with no determinable task must be rejected."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="segment", names=list(EXPECTED_CLASS_NAMES))
    wrapper.model.task = None  # simulate undetectable task
    with pytest.raises(ModelValidationError, match="Cannot determine task"):
        wrapper._validate_model_contract("fake.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: wrong class set is rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_mismatched_classes_rejected():
    """A model trained on different classes must be rejected."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="segment", names=["cat", "dog", "bird"])
    with pytest.raises(ModelValidationError, match="class set"):
        wrapper._validate_model_contract("fake.pt")


def test_partial_classes_rejected():
    """A subset of the expected classes is still a contract violation."""
    wrapper = YOLOWrapper.__new__(YOLOWrapper)
    wrapper.model = _FakeModel(task="segment", names=["person", "motor"])
    with pytest.raises(ModelValidationError, match="class set"):
        wrapper._validate_model_contract("fake.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: missing file raises before any load attempt
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file_raises(tmp_path, patch_yolo):
    missing = tmp_path / "nope.pt"
    with pytest.raises(ModelValidationError, match="not found"):
        YOLOWrapper(str(missing))


# ─────────────────────────────────────────────────────────────────────────────
# Guard: corrupt / unparseable weight file raises ModelValidationError
# (simulates a .pt trained for a newer arch that this ultralytics can't read)
# ─────────────────────────────────────────────────────────────────────────────

def test_corrupt_file_raises(tmp_path, patch_yolo):
    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"this is not a valid torch model")
    patch_yolo["spec"] = {"raise": RuntimeError("Unknown model architecture")}
    with pytest.raises(ModelValidationError, match="Failed to load"):
        YOLOWrapper(str(bad))


# ─────────────────────────────────────────────────────────────────────────────
# Integration: real models present in the repo load and pass validation
# (skipped silently when weights are absent, e.g. on CI without models/)
# ─────────────────────────────────────────────────────────────────────────────

REAL_MODELS = [
    os.path.join("Vision", "models", "ModelRGB_V4.2.pt"),
    os.path.join("Vision", "models", "ModelDepth_V4.pt"),
]


@pytest.mark.parametrize("model_rel", REAL_MODELS)
def test_real_models_pass_validation(model_rel):
    if not os.path.exists(model_rel):
        pytest.skip(f"Model not present: {model_rel}")
    wrapper = YOLOWrapper(model_rel)
    assert wrapper.model.task in SUPPORTED_TASKS
    assert list(wrapper.model.names.values()) == list(EXPECTED_CLASS_NAMES)
