import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy
from PyQt6.QtCore import Qt

# Import komponen modular kamu
from depth_view import DepthView
from radar_view import RadarView
# Nantinya kamu bisa import AlertPanel dan ControlsPanel di sini

class MainWindow(QMainWindow):
    PAD = 20  # Jarak antar widget (Padding)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Robot Surveillance & Monitoring HUD")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 720)

        # 1. DASAR: Depth View (Video Kamera) sebagai Background Utama
        # Kita buat DepthView memenuhi seluruh jendela
        self.area_kamera = DepthView()
        self.setCentralWidget(self.area_kamera)

        # 2. OVERLAY: Tempat menaruh elemen HUD (Radar, dll)
        # Widget ini transparan dan menempel di atas kamera
        self.overlay = QWidget(self.area_kamera)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 3. KONTEN HUD
        # Kita panggil Radar yang sudah kita buat tadi
        self.radar = RadarView(self.overlay)
        
        # --- Kamu bisa tambah AlertPanel atau InfoPanel di sini nanti ---
        # self.alert_panel = AlertPanel(self.overlay)

    def resizeEvent(self, event):
        """Fungsi ini otomatis jalan saat jendela dibesar/kecilkan"""
        super().resizeEvent(event)
        W = self.width()
        H = self.height()

        # Pastikan overlay selalu menutupi seluruh area kamera
        self.overlay.setGeometry(0, 0, W, H)

        # POSISI RADAR: Pojok Kanan Bawah
        rw, rh = self.radar.width(), self.radar.height()
        self.radar.move(W - rw - self.PAD, H - rh - self.PAD)

        # POSISI LAIN (Contoh untuk nanti):
        # self.alert_panel.move(self.PAD, self.PAD) # Pojok Kiri Atas

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Memberikan gaya "Fusion" agar lebih modern
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())