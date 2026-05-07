GLOBAL_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QLabel {
    color: #cdd6f4;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 8px;
    padding: 8px;
    background-color: #1e1e2e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #89b4fa;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: #45475a;
}

QPushButton:disabled {
    color: #585b70;
}

QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px;
}
"""
