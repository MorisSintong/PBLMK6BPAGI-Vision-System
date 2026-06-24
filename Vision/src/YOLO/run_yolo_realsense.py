import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# =============================================
# KONFIGURASI
# =============================================
BRIGHTNESS_THRESHOLD = 40   # Di bawah ini = gelap → pakai model depth
CONF_THRESHOLD = 0.6        # Minimum confidence YOLO
IOU_THRESHOLD = 0.3         # IOU untuk NMS
MAX_DETECTIONS = 10         # Maksimum objek per frame

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. LOAD 2 MODEL
    rgb_model_path = os.path.join(script_dir, "ModelRGB_V4_Beta.pt")
    depth_model_path = os.path.join(script_dir, "ModelDepth.pt")

    if not os.path.exists(rgb_model_path):
        print(f"[!] Model RGB tidak ditemukan: {rgb_model_path}")
        return
    if not os.path.exists(depth_model_path):
        print(f"[!] Model Depth tidak ditemukan: {depth_model_path}")
        return

    print(f"[*] Loading Model RGB  : {os.path.basename(rgb_model_path)}")
    model_rgb = YOLO(rgb_model_path)
    
    print(f"[*] Loading Model Depth: {os.path.basename(depth_model_path)}")
    model_depth = YOLO(depth_model_path)

    # Model yang sedang aktif (mulai dengan RGB)
    active_mode = "RGB"
    active_model = model_rgb
    print(f"[*] Mode awal: {active_mode}")

    # 2. INISIALISASI REALSENSE
    print("[*] Menghubungkan ke kamera Intel RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)

    try:
        pipeline.start(config)
        print("[*] Kamera siap!")
        print("[*] Kontrol: 'q' = Keluar | 'm' = Paksa swap manual")
        print(f"[*] Auto-Swap: Brightness < {BRIGHTNESS_THRESHOLD} → DEPTH | >= {BRIGHTNESS_THRESHOLD} → RGB")
    except Exception as e:
        print(f"[!] Gagal memulai RealSense: {e}")
        return

    # 3. LOOP DETEKSI
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # --- AUTO-SWAP: Cek kecerahan RGB ---
            brightness = color_image.mean()

            if brightness < BRIGHTNESS_THRESHOLD and active_mode != "DEPTH":
                active_mode = "DEPTH"
                active_model = model_depth
                print(f"[SWAP] Gelap terdeteksi (brightness={brightness:.1f}) → Beralih ke MODEL DEPTH")
            elif brightness >= BRIGHTNESS_THRESHOLD and active_mode != "RGB":
                active_mode = "RGB"
                active_model = model_rgb
                print(f"[SWAP] Terang terdeteksi (brightness={brightness:.1f}) → Beralih ke MODEL RGB")

            # --- DETEKSI YOLO (dengan tracker agar stabil) ---
            rgb_display = color_image.copy()
            colormap_display = depth_colormap.copy()

            if active_mode == "RGB":
                # Deteksi pada frame RGB
                results = active_model.track(
                    color_image, persist=True, verbose=False,
                    conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS
                )
            else:
                # Deteksi ada frame Colormap (DEPTH mandiri!)
                results = active_model.track(
                    depth_colormap, persist=True, verbose=False,
                    conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS
                )

            # --- GAMBAR BOUNDING BOX ---
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label_name = active_model.names[cls_id]

                # Track ID
                track_id = ""
                if box.id is not None:
                    track_id = f" #{int(box.id[0])}"

                # Hitung jarak
                cx = max(0, min((x1 + x2) // 2, 639))
                cy = max(0, min((y1 + y2) // 2, 479))
                distance_m = depth_frame.get_distance(cx, cy)

                text_label = f"{label_name}{track_id} {conf:.2f} | {distance_m:.2f}m"

                # Warna per kelas (BGR): 0=mobil(hijau), 1=motor(biru), 2=person(merah)
                CLASS_COLORS = {
                    0: (0, 255, 0),     # mobil  → Hijau
                    1: (255, 150, 0),   # motor  → Biru
                    2: (0, 0, 255),     # person → Merah
                }
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Gambar di RGB
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb_display, text_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(rgb_display, (cx, cy), 5, (0, 0, 255), -1)

                # Gambar di Colormap (warna sama agar konsisten)
                cv2.rectangle(colormap_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(colormap_display, text_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(colormap_display, (cx, cy), 5, color, -1)

            # --- STATUS BAR (pojok kiri atas) ---
            mode_color = (0, 255, 0) if active_mode == "RGB" else (0, 200, 255)
            status_text = f"Mode: {active_mode} | Brightness: {brightness:.0f}"
            cv2.putText(rgb_display, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            cv2.putText(colormap_display, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Gabungkan & Tampilkan
            combined = np.hstack((rgb_display, colormap_display))
            cv2.imshow('Dual YOLO: RGB + Depth (Auto-Swap) | q=Keluar | m=Manual Swap', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Manual swap untuk testing
                if active_mode == "RGB":
                    active_mode = "DEPTH"
                    active_model = model_depth
                else:
                    active_mode = "RGB"
                    active_model = model_rgb
                print(f"[MANUAL] Dipaksa beralih ke: {active_mode}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[*] Selesai.")

if __name__ == "__main__":
    main()
