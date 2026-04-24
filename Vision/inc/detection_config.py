class DetectionConfig:
    def __init__(self):
        # Jarak threshold dalam meter
        self.min_distance = 0.3   # jarak minimum (terlalu dekat, diabaikan)
        self.max_distance = 5.0   # jarak maksimum yang dipantau
        self.danger_distance = 1.5  # jarak bahaya = obstacle terdeteksi

    def set_danger_distance(self, distance: float):
        """Ubah jarak bahaya, dipanggil dari GUI"""
        if self.min_distance <= distance <= self.max_distance:
            self.danger_distance = distance
            print(f"Danger distance diubah ke: {distance} meter")
        else:
            print(f"Error: Jarak harus antara {self.min_distance} - {self.max_distance} meter")

    def get_danger_distance(self):
        return self.danger_distance