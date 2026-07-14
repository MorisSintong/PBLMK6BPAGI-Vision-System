"""GPU detection and performance utilities for laptop power-state bypass.

On laptops with NVIDIA GPUs, Windows power management throttles the GPU
when running on battery, causing dramatic FPS drops. The GPU clock can be
reduced to ~10% of max (e.g., 210 MHz vs 1875 MHz on RTX A4000 Laptop).

This module provides utilities to:
1. Force CUDA initialization (bypass torch.cuda.is_available() False on battery)
2. Detect battery state via Windows GetSystemPowerStatus API
3. Lock GPU clocks to maximum using nvidia-smi (bypass Windows power throttling)
4. One-call ``setup_gpu_for_max_performance()`` for application startup

Usage:
    # In main.py or yolowrapper.py, AFTER import torch:
    from Vision.src.gpu_utils import setup_gpu_for_max_performance
    info = setup_gpu_for_max_performance(lock_clocks=True)
    if info["clocks_locked"]:
        logger.info(f"GPU locked at {info['current_graphics_mhz']} MHz")
"""
import os
import sys
import logging
import subprocess
import ctypes
from ctypes import wintypes

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Battery detection (Windows)
# ═══════════════════════════════════════════════════════════════════════════


def is_on_battery() -> bool:
    """Check if the system is running on battery power.

    Returns:
        True if on battery, False if plugged in (AC) or unknown.
    """
    if sys.platform != "win32":
        return False  # On Linux/macOS, assume plugged in

    try:
        # Use Windows GetSystemPowerStatus API
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("Reserved1", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            # ACLineStatus: 0 = battery, 1 = AC, 255 = unknown
            return status.ACLineStatus == 0
    except Exception as e:
        logger.debug(f"Could not determine power state: {e}")

    return False


# ═══════════════════════════════════════════════════════════════════════════
# NVIDIA GPU performance controls
# ═══════════════════════════════════════════════════════════════════════════


def _run_nvidia_smi(args: list) -> tuple:
    """Run nvidia-smi with given arguments.

    Returns:
        (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"] + args,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", "nvidia-smi not found in PATH"
    except subprocess.TimeoutExpired:
        return -2, "", "nvidia-smi timed out"
    except Exception as e:
        return -3, "", str(e)


def get_gpu_clocks() -> dict:
    """Get current and max GPU clocks via nvidia-smi.

    Returns:
        Dict with keys: current_graphics_mhz, current_memory_mhz, max_graphics_mhz
    """
    rc, stdout, stderr = _run_nvidia_smi([
        "--query-gpu=clocks.current.graphics,clocks.current.memory,clocks.max.graphics",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return {}

    try:
        parts = stdout.strip().split(",")
        return {
            "current_graphics_mhz": int(parts[0].strip()),
            "current_memory_mhz": int(parts[1].strip()),
            "max_graphics_mhz": int(parts[2].strip()),
        }
    except (IndexError, ValueError) as e:
        logger.debug(f"Failed to parse nvidia-smi output: {e}")
        return {}


def lock_gpu_clocks(max_graphics_mhz: int) -> bool:
    """Lock GPU graphics clock to a specific frequency.

    This overrides Windows power management throttling. Requires admin
    privileges on most systems.

    Args:
        max_graphics_mhz: Target graphics clock in MHz.

    Returns:
        True if successful, False otherwise.
    """
    # Set persistence mode first (required for clock locking)
    rc, _, _ = _run_nvidia_smi(["-pm", "1"])
    if rc != 0:
        logger.warning("Failed to set persistence mode (admin required?)")

    # Lock graphics clocks
    rc, stdout, stderr = _run_nvidia_smi([
        "-lgc", str(max_graphics_mhz)
    ])

    if rc == 0:
        logger.info(f"GPU graphics clock locked to {max_graphics_mhz} MHz")
        return True
    else:
        logger.warning(
            f"Failed to lock GPU clocks (rc={rc}): {stderr.strip()}. "
            "This usually requires administrator privileges."
        )
        return False


def unlock_gpu_clocks() -> bool:
    """Unlock GPU clocks (reset to default auto-management).

    Returns:
        True if successful.
    """
    rc, _, _ = _run_nvidia_smi(["-rgc"])
    if rc == 0:
        logger.info("GPU clocks unlocked (back to auto)")
        return True
    return False


def set_gpu_power_limit(watts: int) -> bool:
    """Set GPU power limit (TDP cap).

    Args:
        watts: Power limit in watts.

    Returns:
        True if successful.
    """
    rc, _, _ = _run_nvidia_smi(["-pl", str(watts)])
    return rc == 0


# ═══════════════════════════════════════════════════════════════════════════
# CUDA initialization
# ═══════════════════════════════════════════════════════════════════════════


def force_cuda_init() -> bool:
    """Force CUDA initialization and return whether GPU is usable.

    This works around the battery/power-state issue where
    ``torch.cuda.is_available()`` returns False on battery.

    Returns:
        True if a CUDA-capable GPU is detected and can be used, False otherwise.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU")
        return False

    # Set environment variable to force GPU visibility
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Try to force CUDA initialization
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


# ═══════════════════════════════════════════════════════════════════════════
# Combined setup function
# ═══════════════════════════════════════════════════════════════════════════


def setup_gpu_for_max_performance(lock_clocks: bool = True) -> dict:
    """One-call setup to ensure maximum GPU performance.

    This function:
    1. Checks if running on battery
    2. Forces CUDA initialization
    3. If on battery AND nvidia-smi is available, locks GPU clocks to max
    4. Logs current vs max clock speeds for diagnostics

    Args:
        lock_clocks: If True (default), attempt to lock GPU clocks when on battery.

    Returns:
        Dict with diagnostic info:
        - on_battery: bool
        - gpu_available: bool
        - clocks_locked: bool
        - current_graphics_mhz: int (if available)
        - max_graphics_mhz: int (if available)
    """
    result = {
        "on_battery": False,
        "gpu_available": False,
        "clocks_locked": False,
        "current_graphics_mhz": None,
        "max_graphics_mhz": None,
    }

    # Check power state
    on_battery = is_on_battery()
    result["on_battery"] = on_battery
    if on_battery:
        logger.warning("System is on BATTERY power. GPU may be throttled.")
    else:
        logger.info("System is on AC power (or power state unknown).")

    # Force CUDA init
    gpu_ok = force_cuda_init()
    result["gpu_available"] = gpu_ok

    if not gpu_ok:
        logger.warning("No CUDA GPU available, skipping clock locking.")
        return result

    # Get current clock info
    clocks = get_gpu_clocks()
    if clocks:
        result["current_graphics_mhz"] = clocks.get("current_graphics_mhz")
        result["max_graphics_mhz"] = clocks.get("max_graphics_mhz")

        cur = clocks.get("current_graphics_mhz", 0)
        mx = clocks.get("max_graphics_mhz", 0)
        if mx > 0:
            pct = 100.0 * cur / mx
            logger.info(
                f"GPU clock: {cur}/{mx} MHz ({pct:.0f}%)"
            )
            if on_battery and cur < mx * 0.5:
                logger.warning(
                    f"GPU is throttled to {pct:.0f}% of max! "
                    f"Consider plugging in or locking clocks."
                )

    # Attempt clock locking if on battery
    if lock_clocks and on_battery and clocks.get("max_graphics_mhz"):
        max_mhz = clocks["max_graphics_mhz"]
        if lock_gpu_clocks(max_mhz):
            result["clocks_locked"] = True
            # Re-read clocks to verify
            new_clocks = get_gpu_clocks()
            if new_clocks:
                result["current_graphics_mhz"] = new_clocks.get("current_graphics_mhz")
                logger.info(
                    f"After lock: GPU clock = {new_clocks.get('current_graphics_mhz')} MHz"
                )
        else:
            logger.warning(
                "Could not lock GPU clocks. "
                "Run the application as Administrator, or plug in the charger."
            )

    return result


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
        return True
    except ImportError:
        return False
