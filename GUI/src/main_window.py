import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from depth_view import DepthView
from radar_view import RadarView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Pengaturan Jendela Utama
        self.setWindowTitle("Vision System PBL GUI")
        self.resize(1024, 768)

        # Membuat Widget Pusat
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout Utama: Kiri dan Kanan
        main_layout = QHBoxLayout(central_widget)

        # --- Bagian Kiri ---
        kiri_layout = QVBoxLayout()
        self.area_kamera = DepthView() 
        kiri_layout.addWidget(self.area_kamera)

        # --- Bagian Kanan ---
        kanan_layout = QVBoxLayout()
        label_kanan = QLabel("Area untuk Controls & Alert Panel nanti")
        label_kanan.setStyleSheet("background-color: lightblue; border: 1px solid black;")
        kanan_layout.addWidget(label_kanan)
        self.radar = RadarView()
        kanan_layout.addWidget(self.radar, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)

        # Gabungkan ke layout utama
        main_layout.addLayout(kiri_layout, 70) 
        main_layout.addLayout(kanan_layout, 30)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())