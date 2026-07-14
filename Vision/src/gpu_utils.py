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
# Windows power management (powercfg)
# ═══════════════════════════════════════════════════════════════════════════


def _run_powercfg(args: list) -> tuple:
    """Run powercfg with given arguments.

    Returns:
        (returncode, stdout, stderr)
    """
    if sys.platform != "win32":
        return -1, "", "powercfg only available on Windows"
    try:
        result = subprocess.run(
            ["powercfg"] + args,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", "powercfg not found"
    except Exception as e:
        return -2, "", str(e)


def disable_pci_express_power_management() -> bool:
    """Disable PCI Express Link State Power Management.

    This is one of the main Windows settings that causes GPU throttling
    on battery. It can be disabled without admin privileges (it modifies
    the current user's power settings, not system-wide).

    The relevant GUIDs:
    - Subgroup: 501a4d13-42af-4429-9fd1-a8218c268e20 (PCI Express)
    - Setting:  ee12f906-d277-404b-b6da-e5fa1a576df5 (Link State Power Management)
    - Values:   0=Off, 1=Moderate, 2=Maximum power savings

    Returns:
        True if successful.
    """
    PCI_EXPRESS_SUBGROUP = "501a4d13-42af-4429-9fd1-a8218c268e20"
    LINK_STATE_SETTING = "ee12f906-d277-404b-b6da-e5fa1a576df5"

    rc, _, stderr = _run_powercfg([
        "/setacvalueindex", "SCHEME_CURRENT",
        PCI_EXPRESS_SUBGROUP, LINK_STATE_SETTING, "0"
    ])
    if rc != 0:
        logger.warning(f"Failed to disable PCI Express power management (AC): {stderr.strip()}")
        return False

    rc, _, stderr = _run_powercfg([
        "/setdcvalueindex", "SCHEME_CURRENT",
        PCI_EXPRESS_SUBGROUP, LINK_STATE_SETTING, "0"
    ])
    if rc != 0:
        logger.warning(f"Failed to disable PCI Express power management (DC): {stderr.strip()}")
        return False

    # Apply the changes
    rc, _, _ = _run_powercfg(["/setactive", "SCHEME_CURRENT"])
    if rc != 0:
        logger.warning("Failed to apply power scheme changes")

    logger.info("Disabled PCI Express Link State Power Management")
    return True


def set_processor_max_state(percent: int = 100) -> bool:
    """Set maximum processor state to a specific percentage.

    Args:
        percent: 0-100. Use 100 for full performance.

    Returns:
        True if successful.
    """
    PROCESSOR_SUBGROUP = "54533251-82be-4824-96c1-47b60b740d00"
    MAX_STATE_SETTING = "bc5038f7-23e0-4960-96da-33abaf5935ec"
    # 0x64 = 100, convert to hex
    value_hex = format(percent, "x")

    rc, _, stderr = _run_powercfg([
        "/setacvalueindex", "SCHEME_CURRENT",
        PROCESSOR_SUBGROUP, MAX_STATE_SETTING, value_hex
    ])
    if rc != 0:
        logger.warning(f"Failed to set AC max processor state: {stderr.strip()}")
        return False

    rc, _, stderr = _run_powercfg([
        "/setdcvalueindex", "SCHEME_CURRENT",
        PROCESSOR_SUBGROUP, MAX_STATE_SETTING, value_hex
    ])
    if rc != 0:
        logger.warning(f"Failed to set DC max processor state: {stderr.strip()}")
        return False

    rc, _, _ = _run_powercfg(["/setactive", "SCHEME_CURRENT"])
    logger.info(f"Set processor max state to {percent}%")
    return rc == 0


def set_high_performance_power_plan() -> bool:
    """Switch to the built-in 'High Performance' power plan.

    The 'High Performance' plan (GUID 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c)
    prevents CPU/GPU throttling on battery. Requires admin to switch.

    Returns:
        True if successful.
    """
    HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    rc, _, stderr = _run_powercfg(["/setactive", HIGH_PERF_GUID])
    if rc == 0:
        logger.info("Switched to High Performance power plan")
        return True
    logger.debug(f"Could not switch to High Performance plan: {stderr.strip()}")
    return False


def apply_windows_gpu_unthrottle() -> bool:
    """Apply Windows-side fixes to prevent GPU throttling on battery.

    This function:
    1. Disables PCI Express Link State Power Management (no admin needed)
    2. Sets processor max state to 100% for both AC and DC (no admin for
       current user scheme)

    The function uses the current user's power scheme (SCHEME_CURRENT) so
    it does NOT require admin privileges for these specific changes.

    Returns:
        True if all changes were applied successfully.
    """
    success = True
    if not disable_pci_express_power_management():
        success = False
    if not set_processor_max_state(100):
        success = False
    # Also disable USB selective suspend (affects RealSense camera on battery)
    USB_SUBGROUP = "2a737441-1930-4402-8d77-b2bebba308a3"
    USB_SUSPEND_SETTING = "48e6b7a6-50f5-4782-a5d4-53bb8f07e226"
    _run_powercfg(["/setacvalueindex", "SCHEME_CURRENT", USB_SUBGROUP, USB_SUSPEND_SETTING, "0"])
    _run_powercfg(["/setdcvalueindex", "SCHEME_CURRENT", USB_SUBGROUP, USB_SUSPEND_SETTING, "0"])
    _run_powercfg(["/setactive", "SCHEME_CURRENT"])
    return success


def _find_nvidia_registry_key() -> str:
    """Find the registry subkey for the NVIDIA GPU adapter.

    On laptops with dual GPUs (Intel iGPU + NVIDIA dGPU), the NVIDIA
    adapter may be in \\0000 or \\0001 depending on enumeration order.
    This function scans all subkeys and returns the one whose
    DriverDesc contains 'NVIDIA'.

    Returns:
        Registry path string, or empty string if not found.
    """
    if sys.platform != "win32":
        return ""

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as parent:
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(parent, i)
                    i += 1
                    subkey_path = f"{base}\\{subkey_name}"
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey_path) as sk:
                            desc, _ = winreg.QueryValueEx(sk, "DriverDesc")
                            if "NVIDIA" in str(desc):
                                return subkey_path
                    except (FileNotFoundError, OSError):
                        pass
                except OSError:
                    break
    except Exception:
        pass
    return ""


def try_set_nvidia_registry_keys() -> bool:
    """Try to set NVIDIA driver registry keys for maximum performance.

    These keys control NVIDIA PowerMizer and performance levels:
    - PerfLevelSrc: 0x33222220 = prefer max performance on all levels
    - PowerMizerEnable: 0 = disable power saving
    - PowerMizerLevel: 1 = max performance (AC)
    - PowerMizerLevelDC: 1 = max performance (battery/DC)

    Automatically finds the correct NVIDIA adapter subkey (not the Intel
    iGPU). Requires admin privileges. If not admin, returns False and logs
    a helpful message.

    Returns:
        True if all keys were set successfully.
    """
    if sys.platform != "win32":
        return False

    nvidia_key = _find_nvidia_registry_key()
    if not nvidia_key:
        logger.warning("Could not find NVIDIA adapter in registry")
        return False

    keys_to_set = {
        "PerfLevelSrc": 0x33222220,
        "PowerMizerEnable": 0x00000000,
        "PowerMizerLevel": 0x00000001,
        "PowerMizerLevelDC": 0x00000001,
    }

    try:
        import winreg
    except ImportError:
        return False

    success = True
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, nvidia_key, 0, winreg.KEY_SET_VALUE) as key:
            for name, value in keys_to_set.items():
                try:
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
                    logger.info(f"Set NVIDIA registry: {name} = 0x{value:08X}")
                except PermissionError:
                    logger.warning(f"Permission denied setting {name} (need admin)")
                    success = False
                except Exception as e:
                    logger.debug(f"Failed to set {name}: {e}")
                    success = False
    except PermissionError:
        logger.warning(
            "Cannot open NVIDIA registry key (need admin). "
            "Run setup_gpu_admin.bat as Administrator for permanent GPU fix."
        )
        return False
    except FileNotFoundError:
        logger.debug("NVIDIA registry key not found")
        return False

    if success:
        logger.info("NVIDIA registry keys set - restart may be needed for full effect")
    return success


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
    2. Applies Windows-side fixes (PCI Express power management, processor state)
    3. Forces CUDA initialization
    4. If on battery AND nvidia-smi is available, locks GPU clocks to max
    5. Logs current vs max clock speeds for diagnostics

    Args:
        lock_clocks: If True (default), attempt to lock GPU clocks when on battery.

    Returns:
        Dict with diagnostic info:
        - on_battery: bool
        - gpu_available: bool
        - clocks_locked: bool
        - windows_unthrottle_applied: bool
        - current_graphics_mhz: int (if available)
        - max_graphics_mhz: int (if available)
    """
    result = {
        "on_battery": False,
        "gpu_available": False,
        "clocks_locked": False,
        "windows_unthrottle_applied": False,
        "current_graphics_mhz": None,
        "max_graphics_mhz": None,
    }

    # Check power state
    on_battery = is_on_battery()
    result["on_battery"] = on_battery
    if on_battery:
        logger.warning("System is on BATTERY power. Applying GPU unthrottle measures.")
    else:
        logger.info("System is on AC power (or power state unknown).")

    # Apply Windows-side fixes to prevent GPU throttling
    # This does NOT require admin for current user scheme
    windows_ok = apply_windows_gpu_unthrottle()
    result["windows_unthrottle_applied"] = windows_ok
    if windows_ok:
        logger.info("Windows power management unthrottle applied (PCI Express + processor state)")
    else:
        logger.warning("Some Windows power management fixes failed (may need admin)")

    # Try to set NVIDIA registry keys (requires admin, fails gracefully)
    try_set_nvidia_registry_keys()

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
                "Could not lock GPU clocks via nvidia-smi. "
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


# ═══════════════════════════════════════════════════════════════════════════
# GPU Keepalive — prevent Optimus from powering off the dGPU on battery
# ═══════════════════════════════════════════════════════════════════════════

_keepalive_thread = None
_keepalive_running = False


def start_gpu_keepalive(interval_ms: int = 100):
    """Start a background thread that keeps the NVIDIA dGPU awake.

    On laptops with NVIDIA Optimus, the dGPU powers off after ~2 seconds
    of inactivity to save battery. When the next CUDA call arrives, it
    takes 1-2 seconds for the dGPU to wake up, causing massive latency
    spikes (0 FPS for several frames).

    This thread runs a tiny CUDA operation every `interval_ms` milliseconds,
    preventing the dGPU from entering sleep state. The overhead is negligible
    (~0.05 ms per tick).

    Args:
        interval_ms: How often to ping the GPU (default 100ms).
    """
    global _keepalive_thread, _keepalive_running

    if _keepalive_running:
        return  # Already running

    try:
        import torch
        if not torch.cuda.is_available():
            return
    except ImportError:
        return

    _keepalive_running = True

    def _keepalive_loop():
        global _keepalive_running
        try:
            # Allocate a small tensor on GPU once
            dummy = torch.randn(16, 16, device="cuda")
            while _keepalive_running:
                # Tiny operation to keep dGPU awake
                dummy = dummy * 1.0001
                torch.cuda.synchronize()
                time_module.sleep(interval_ms / 1000.0)
        except Exception:
            pass  # Silent failure — keepalive is best-effort

    import threading
    import time as time_module
    _keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
    _keepalive_thread.start()
    logger.info(f"GPU keepalive started (interval={interval_ms}ms) — prevents dGPU sleep on battery")


def stop_gpu_keepalive():
    """Stop the GPU keepalive thread."""
    global _keepalive_running
    _keepalive_running = False
