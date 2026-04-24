# GUI/src/Alert_panel.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class AlertPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Default threshold (meter)
        self.threshold_warning = 1.0   
        self.threshold_danger  = 0.5   
        
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        # Layout utama panel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Dorong konten ke bawah agar berada di pojok kanan bawah seperti mockup
        layout.addStretch(1)

        # ── Container Info (Lebih Kecil & Modern) ──
        self.info_box = QFrame()
        self.info_box.setObjectName("infoBox")
        self.info_box.setFixedHeight(160) # Ukuran dikecilkan agar tidak terlalu dominan
        
        box_layout = QVBoxLayout(self.info_box)
        box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_layout.setSpacing(2) # Jarak antar teks lebih rapat
        
        # ── Label Nama Object ──
        self.lbl_object_name = QLabel("Menunggu...")
        self.lbl_object_name.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.lbl_object_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_object_name.setStyleSheet("color: #cdd6f4; background: transparent;")
        
        # ── Label Jarak ──
        self.lbl_distance = QLabel("-- m")
        self.lbl_distance.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        self.lbl_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_distance.setStyleSheet("color: white; background: transparent;")
        
        box_layout.addWidget(self.lbl_object_name)
        box_layout.addWidget(self.lbl_distance)
        
        layout.addWidget(self.info_box)

    def _apply_style(self):
        # Background disesuaikan dengan tema utama (#1e1e2e)
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
            #infoBox {
                background-color: #313244; /* Warna permukaan yang senada dengan tombol GUI */
                border: 2px solid #45475a;  /* Border tipis agar terlihat rapi */
                border-radius: 15px;
            }
        """)

    # ------------------------------------------------------------------ #
    #  Fungsi Komunikasi (Panggil ini untuk update dinamis)              #
    # ------------------------------------------------------------------ #
    
    def set_thresholds(self, warning_m: float, danger_m: float):
        self.threshold_warning = warning_m
        self.threshold_danger  = danger_m

    def update_info(self, object_name: str, distance_m: float):
        """
        DIPANGGIL OLEH VISION:
        object_name: misal 'Person', 'Helmet', dll.
        distance_m: jarak dalam meter hasil kalkulasi kamera RealSense.
        """
        self.lbl_object_name.setText(object_name.upper())
        
        if distance_m is None:
            self.lbl_distance.setText("-- m")
            self.info_box.setStyleSheet("background-color: #313244; border: 2px solid #45475a; border-radius: 15px;")
            return

        self.lbl_distance.setText(f"{distance_m:.1f} m")

        # Perubahan warna box dinamis berdasarkan threshold
        if distance_m <= self.threshold_danger:
            bg_color = "#f38ba8" # Danger (Red)
            text_color = "#1e1e2e" # Teks gelap agar kontras
        elif distance_m <= self.threshold_warning:
            bg_color = "#f9e2af" # Warning (Yellow)
            text_color = "#1e1e2e"
        else:
            bg_color = "#313244" # Safe (Default Dark)
            text_color = "#cdd6f4"
            
        self.info_box.setStyleSheet(f"""
            background-color: {bg_color}; 
            border-radius: 15px;
        """)
        self.lbl_distance.setStyleSheet(f"color: {text_color}; background: transparent;")
        self.lbl_object_name.setStyleSheet(f"color: {text_color}; background: transparent;")