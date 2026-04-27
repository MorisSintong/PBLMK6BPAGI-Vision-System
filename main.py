"""
main.py — Entry point for the Depth Obstacle Detector application.

Responsibilities:
  - Register all submodule paths into sys.path
  - Bootstrap the QApplication
  - Instantiate and show MainWindow
  - Start the Qt event loop
"""

import os
import sys

# ── sys.path registration ─────────────────────────────────────────────────────
# Allows all submodules to use flat imports (e.g. `from camera_thread import ...`)
# without needing package __init__.py files.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_PATHS = [
    os.path.join(BASE_DIR, "gui", "src"),
    os.path.join(BASE_DIR, "gui", "inc"),
    os.path.join(BASE_DIR, "vision", "src"),
    os.path.join(BASE_DIR, "vision", "inc"),
]

for path in SEARCH_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Qt bootstrap ──────────────────────────────────────────────────────────────
# from PyQt6.QtGui     import QIcon   #tidak digunakan, nanti kalau perlu tinggal di ilangin simbol komennya
from main_window import MainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from ui_config import APP_NAME, WINDOW_MIN_H, WINDOW_MIN_W


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    # High-DPI scaling (automatic in PyQt6, but explicit for clarity)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("depth-obstacle-detector")

    # Apply global stylesheet
    try:
        from styles import GLOBAL_STYLESHEET  # type: ignore

        app.setStyleSheet(GLOBAL_STYLESHEET)
    except ImportError:
        pass  # styles.py is optional

    window = MainWindow()
    window.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
