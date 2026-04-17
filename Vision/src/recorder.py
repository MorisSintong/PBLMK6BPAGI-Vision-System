import cv2
import os
import datetime

class FrameRecorder:
    def __init__(self, save_path="data/recordings"):
        # 0 adalah ID default untuk webcam bawaan laptop
        self.cap = cv2.VideoCapture(0)
        
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def start(self):
        if not self.cap.isOpened():
            print("Error: Kamera laptop tidak terdeteksi!")
            return False
        print("Kamera laptop aktif. Tekan 'q' untuk berhenti.")
        return True

    def get_frames(self):
        # Ambil frame dari webcam
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame

    def stop(self):
        self.cap.release()
        print("Kamera dimatikan.")

if __name__ == "__main__":
    recorder = FrameRecorder()
    if recorder.start():
        try:
            while True:
                frame = recorder.get_frames()
                if frame is not None:
                    # Tampilkan hasil kamera di layar
                    cv2.imshow('Laptop Webcam - Security Robot System', frame)
                
                # Tekan 'q' pada keyboard untuk stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            recorder.stop()
            cv2.destroyAllWindows()