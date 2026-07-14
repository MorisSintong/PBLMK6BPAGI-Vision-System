"""GPU detection utilities that work around battery/power-state limitations.

On some laptops, ``torch.cuda.is_available()`` returns ``False`` when the
system is running on battery, even if an NVIDIA GPU is physically present.
This is because Windows battery saver / NVIDIA Battery Boost can throttle
or hide the GPU from CUDA.

This module provides a ``force_cuda_init()`` helper that:
1. Forces CUDA initialization via the private ``_lazy_init()`` API
2. Returns ``True`` if the GPU hardware is present, regardless of power state
3. Optionally sets ``CUDA_VISIBLE_DEVICES`` to force the GPU to be visible

Call ``force_cuda_init()`` early in the application startup (before any
``torch.cuda.is_available()`` check) to ensure GPU is always used when
available.
"""
import os
import logging

logger = logging.getLogger(__name__)


def force_cuda_init() -> bool:
    """Force CUDA initialization and return whether GPU is usable.

    This works around the battery/power-state issue where
    ``torch.cuda.is_available()`` returns ``False`` on battery.

    Returns:
        True if a CUDA-capable GPU is detected and can be used, False otherwise.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU")
        return False

    # Set environment variable to force GPU visibility
    # This must be set before CUDA initialization
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Try to force CUDA initialization
    # _lazy_init() is a private API but is the most reliable way to
    # force CUDA context creation even when is_available() returns False
    try:
        if hasattr(torch.cuda, "_lazy_init"):
            torch.cuda._lazy_init()
    except Exception as e:
        logger.debug(f"CUDA _lazy_init() failed (non-fatal): {e}")

    # Check if GPU hardware is present
    try:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            logger.info(f"GPU detected: {device_count} device(s)")
            for i in range(device_count):
                try:
                    name = torch.cuda.get_device_name(i)
                    logger.info(f"  Device {i}: {name}")
                except Exception:
                    pass
            return True
    except Exception as e:
        logger.debug(f"torch.cuda.device_count() failed: {e}")

    return False


def get_device(prefer_gpu: bool = True) -> str:
    """Get the best available device string for ultralytics/pytorch.

    Args:
        prefer_gpu: If True, prefer GPU when available. If False, always use CPU.

    Returns:
        Device string: "0" for first GPU, or "cpu".
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if not prefer_gpu:
        return "cpu"

    # Try to force CUDA init first (works around battery issue)
    force_cuda_init()

    if torch.cuda.is_available():
        return "0"

    # Fallback: check if GPU hardware exists even if not "available"
    try:
        if torch.cuda.device_count() > 0:
            logger.warning(
                "GPU hardware detected but torch.cuda.is_available() returned False. "
                "This usually means running on battery power. "
                "Forcing GPU usage anyway."
            )
            return "0"
    except Exception:
        pass

    logger.info("No GPU available, using CPU")
    return "cpu"


def should_use_fp16(device: str) -> bool:
    """Determine if FP16 (half precision) should be used.

    Args:
        device: Device string ("0" for GPU, "cpu" for CPU).

    Returns:
        True if FP16 should be enabled.
    """
    if device == "cpu":
        return False
    try:
        import torch
        if torch.cuda.is_available():
            return True
        # Even if is_available() returns False, if device is "0" we want to try
        return True
    except ImportError:
        return False
