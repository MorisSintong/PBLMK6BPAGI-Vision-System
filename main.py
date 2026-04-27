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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_PATHS = [
    os.path.join(BASE_DIR, "GUI",    "src"),
    os.path.join(BASE_DIR, "GUI",    "inc"),
    os.path.join(BASE_DIR, "Vision", "src"),
    os.path.join(BASE_DIR, "Vision", "inc"),
]

for path in SEARCH_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Qt bootstrap ──────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from main_window     import MainWindow
from ui_config       import APP_NAME, WINDOW_MIN_W, WINDOW_MIN_H

# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("depth-obstacle-detector")

    try:
        from styles import GLOBAL_STYLESHEET  # type: ignore

        app.setStyleSheet(GLOBAL_STYLESHEET)
    except ImportError:
        pass

    window = MainWindow()
    window.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
