"""Quick diagnostic to check GPU setup and battery throttling.

Run with: python tests/diagnose_gpu.py
"""
import sys
import os
import subprocess
import ctypes
from ctypes import wintypes

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Vision", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Vision", "inc"))


def run_cmd(cmd, timeout=10):
    """Run a shell command and return (rc, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=True
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", "command not found"
    except Exception as e:
        return -2, "", str(e)


def check_battery():
    """Check battery state via Windows API."""
    print("=" * 60)
    print("BATTERY STATE")
    print("=" * 60)
    if sys.platform != "win32":
        print("Not Windows, skipping.")
        return False

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
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        print("  Could not read power status.")
        return False

    states = {0: "BATTERY", 1: "AC (plugged in)", 255: "UNKNOWN"}
    state = states.get(status.ACLineStatus, f"code={status.ACLineStatus}")
    print(f"  Power source: {state}")
    if status.ACLineStatus == 0:
        print(f"  Battery: {status.BatteryLifePercent}%")
    return status.ACLineStatus == 0


def check_nvidia_smi():
    """Check nvidia-smi availability and GPU info."""
    print("\n" + "=" * 60)
    print("NVIDIA-SMI")
    print("=" * 60)
    rc, stdout, stderr = run_cmd("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    if rc != 0:
        print(f"  nvidia-smi not available: {stderr.strip()}")
        return False
    print(f"  GPU: {stdout.strip()}")

    # Get current vs max clocks
    rc, stdout, stderr = run_cmd(
        "nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,clocks.current.sm,clocks.max.sm "
        "--format=csv,noheader,nounits"
    )
    if rc == 0:
        for line in stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                cur_g, max_g, cur_sm, max_sm = parts
                pct_g = 100.0 * int(cur_g) / int(max_g) if int(max_g) > 0 else 0
                pct_sm = 100.0 * int(cur_sm) / int(max_sm) if int(max_sm) > 0 else 0
                print(f"  Graphics clock: {cur_g}/{max_g} MHz ({pct_g:.0f}%)")
                print(f"  SM clock:       {cur_sm}/{max_sm} MHz ({pct_sm:.0f}%)")
                if pct_g < 50:
                    print("  WARNING: GPU is heavily throttled!")

    # Check power state
    rc, stdout, stderr = run_cmd("nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits")
    if rc == 0:
        for line in stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                print(f"  Power draw: {parts[0]}W / {parts[1]}W limit")

    # Check persistence mode
    rc, stdout, stderr = run_cmd("nvidia-smi --query-gpu=persistence_mode --format=csv,noheader")
    if rc == 0:
        print(f"  Persistence mode: {stdout.strip()}")
    return True


def check_torch():
    """Check PyTorch CUDA detection."""
    print("\n" + "=" * 60)
    print("PYTORCH CUDA")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("  PyTorch not installed.")
        return False

    print(f"  torch version: {torch.__version__}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        # Try CUDA init
        try:
            torch.cuda.init()
            print("  torch.cuda.init() OK")
        except Exception as e:
            print(f"  torch.cuda.init() failed: {e}")

    # Check CUDA_VISIBLE_DEVICES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    print(f"  CUDA_VISIBLE_DEVICES: {cvd}")
    return True


def check_power_plan():
    """Check Windows power plan."""
    print("\n" + "=" * 60)
    print("WINDOWS POWER PLAN")
    print("=" * 60)
    rc, stdout, stderr = run_cmd("powercfg /getactivescheme")
    if rc == 0:
        print(f"  Active plan: {stdout.strip()}")
    else:
        print(f"  Could not read: {stderr.strip()}")

    rc, stdout, stderr = run_cmd("powercfg /list")
    if rc == 0:
        for line in stdout.split("\n"):
            if "Active" in line or "*" in line:
                print(f"  {line.strip()}")


def check_admin():
    """Check if running as admin."""
    print("\n" + "=" * 60)
    print("ADMIN STATUS")
    print("=" * 60)
    if sys.platform != "win32":
        print("  Not Windows.")
        return
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        print(f"  Running as admin: {is_admin}")
        if not is_admin:
            print("  WARNING: GPU clock locking requires admin privileges.")
            print("  Re-run PowerShell as Administrator for full GPU performance.")
    except Exception as e:
        print(f"  Could not check: {e}")


def main():
    print("\n" + "#" * 60)
    print("# GPU DIAGNOSTIC TOOL")
    print("#" * 60)

    check_battery()
    check_admin()
    check_power_plan()
    check_nvidia_smi()
    check_torch()

    print("\n" + "#" * 60)
    print("# RECOMMENDATIONS")
    print("#" * 60)
    print("If GPU clock is < 50% of max AND you're on battery:")
    print("  1. Run PowerShell as Administrator")
    print("  2. python main.py")
    print("  3. Check that clock lock succeeds in the logs")
    print()
    print("If torch.cuda.is_available() is False:")
    print("  - Check that pytorch with CUDA is installed:")
    print("    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print()
    print("If nvidia-smi is not found:")
    print("  - Install NVIDIA drivers from nvidia.com")


if __name__ == "__main__":
    main()
