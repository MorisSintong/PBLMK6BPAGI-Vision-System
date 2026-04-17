from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class DepthView(QWidget):
    def __init__(self):
        super().__init__()
        
        # Layout utama untuk area kamera
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Menghilangkan jarak tepi (margin) agar layar penuh

        # Membuat Label yang akan berfungsi sebagai "Layar TV" atau monitor
        self.layar_kamera = QLabel("KAMERA OFFLINE\nMenunggu koneksi dari vision...")
        self.layar_kamera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Memberikan warna latar hitam seperti layar yang mati, dan teks putih
        self.layar_kamera.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #aaaaaa; 
            font-size: 20px;
            font-weight: bold;
            border: 2px dashed #555555;
        """)

        layout.addWidget(self.layar_kamera)

    # --- FUNGSI PENTING UNTUK NANTI ---
    # Fungsi ini belum kita pakai sekarang, tapi ini akan dipanggil oleh 
    # camera_thread.py dari tim vision kalian untuk mengirim gambar video terus-menerus
    def update_frame(self, pixmap):
        self.layar_kamera.setPixmap(pixmap)
        self.layar_kamera.setScaledContents(True) # Agar video pas dengan ukuran jendela