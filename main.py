"""
main.py — Entry point for the Depth Obstacle Detector application.

Responsibilities:
  - Bootstrap the QApplication
  - Apply the global stylesheet
  - Instantiate and show MainWindow
  - Start the Qt event loop
"""

import os
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt, QSize
from GUI.src.main_window import MainWindow
from GUI.inc.ui_config import APP_NAME, WINDOW_MIN_W, WINDOW_MIN_H

# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("depth-obstacle-detector")

    try:
        from GUI.inc.styles import GLOBAL_STYLESHEET  # type: ignore

        app.setStyleSheet(GLOBAL_STYLESHEET)
    except ImportError:
        pass

    window = MainWindow()
    window.setMinimumSize(QSize(WINDOW_MIN_W, WINDOW_MIN_H))
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
