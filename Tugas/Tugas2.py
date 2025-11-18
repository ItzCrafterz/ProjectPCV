import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mode warna yang akan dideteksi
current_color_mode = "Hijau"

# Definisi rentang warna dalam HSV untuk berbagai warna
color_ranges = {
    "Hijau": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255]),
        "color_bgr": (0, 255, 0)
    },
    "Biru": {
        "lower": np.array([90, 50, 50]),
        "upper": np.array([130, 255, 255]),
        "color_bgr": (255, 0, 0)
    },
    "Merah": {
        "lower1": np.array([0, 50, 50]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 50, 50]),
        "upper2": np.array([180, 255, 255]),
        "color_bgr": (0, 0, 255)
    },
    "Kuning": {
        "lower": np.array([20, 50, 50]),
        "upper": np.array([35, 255, 255]),
        "color_bgr": (0, 255, 255)
    }
}

# Background yang tersedia
backgrounds = {
    "Gradient": None,
    "Space": None,
    "Sunset": None,
    "Matrix": None
}

current_background = "Gradient"
background_enabled = False

def create_gradient_background(h, w):
    """Membuat background gradien"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        r = int(100 + (i / h) * 100)
        g = int(50 + (i / h) * 150)
        b = int(150 - (i / h) * 100)
        bg[i, :] = [b, g, r]
    return bg

def create_space_background(h, w):
    """Membuat background luar angkasa dengan bintang"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = [20, 10, 5]  # Dark blue space
    
    # Tambahkan bintang random
    num_stars = 200
    for _ in range(num_stars):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        brightness = np.random.randint(150, 255)
        size = np.random.choice([1, 1, 1, 2, 2, 3])
        cv2.circle(bg, (x, y), size, (brightness, brightness, brightness), -1)
    
    return bg

def create_sunset_background(h, w):
    """Membuat background sunset"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        if i < h // 3:
            # Sky - orange to purple
            r = int(200 - (i / (h // 3)) * 100)
            g = int(100 - (i / (h // 3)) * 50)
            b = int(50 + (i / (h // 3)) * 100)
        else:
            # Ground - darker
            r = int(100 - ((i - h // 3) / (h * 2 // 3)) * 50)
            g = int(50 - ((i - h // 3) / (h * 2 // 3)) * 30)
            b = int(150 - ((i - h // 3) / (h * 2 // 3)) * 50)
        bg[i, :] = [b, g, r]
    return bg

def create_matrix_background(h, w):
    """Membuat background Matrix style"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = [10, 20, 5]  # Dark green
    
    # Tambahkan vertical lines
    num_lines = 30
    for _ in range(num_lines):
        x = np.random.randint(0, w)
        start_y = np.random.randint(0, h // 2)
        length = np.random.randint(50, 200)
        for i in range(length):
            y = start_y + i
            if y < h:
                brightness = int(255 - (i / length) * 200)
                bg[y, x] = [0, brightness, 0]
    
    return bg

def detect_color_object(frame, color_mode):
    """
    Deteksi objek berdasarkan warna menggunakan HSV color space
    Returns: mask yang sudah dibersihkan dan kontur terbesar
    """
    # Konversi BGR ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dapatkan range warna
    color_info = color_ranges[color_mode]
    
    # Thresholding warna
    if color_mode == "Merah":
        # Merah memiliki 2 range karena wraparound di HSV
        mask1 = cv2.inRange(hsv, color_info["lower1"], color_info["upper1"])
        mask2 = cv2.inRange(hsv, color_info["lower2"], color_info["upper2"])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, color_info["lower"], color_info["upper"])
    
    # Operasi morfologi untuk membersihkan noise
    # Opening: menghapus noise kecil (false positives)
    kernel_open = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing: menutup lubang kecil (false negatives)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Temukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    max_area = 0
    
    # Cari kontur terbesar
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > 500:  # Threshold minimal area
            max_area = area
            largest_contour = contour
    
    return mask, largest_contour, max_area

def apply_background_replacement(frame, mask_inverse, background):
    """
    Ganti background menggunakan mask
    mask_inverse: area yang BUKAN objek (area yang akan diganti)
    """
    h, w = frame.shape[:2]
    
    # Pastikan background memiliki ukuran yang sama
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h))
    
    # Buat mask 3 channel
    mask_3ch = cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2BGR)
    
    # Smooth edges dengan blur
    mask_3ch = cv2.GaussianBlur(mask_3ch, (5, 5), 0)
    
    # Normalisasi mask ke range 0-1
    mask_float = mask_3ch.astype(float) / 255.0
    
    # Terapkan background: frame * (1 - mask) + background * mask
    result = (frame.astype(float) * (1 - mask_float) + 
              background.astype(float) * mask_float)
    
    return result.astype(np.uint8)

def draw_ui(frame, color_mode, bg_mode, bg_enabled, detected, area, h, w):
    """Menggambar UI informasi"""
    # Panel utama
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (650, 280), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (10, 10), (650, 280), (0, 255, 255), 3)
    
    # Title
    cv2.putText(frame, "TUGAS 2: DETEKSI WARNA & GANTI BACKGROUND", (25, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Status deteksi
    if detected:
        status_text = f"TERDETEKSI: Objek {color_mode}"
        status_color = (0, 255, 0)
        cv2.putText(frame, f"Area: {int(area)} px", (25, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        status_text = f"Mencari objek {color_mode}..."
        status_color = (0, 165, 255)
    
    cv2.putText(frame, status_text, (25, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Background status
    bg_status = f"Background: {bg_mode} {'[AKTIF]' if bg_enabled else '[NONAKTIF]'}"
    bg_color = (0, 255, 0) if bg_enabled else (128, 128, 128)
    cv2.putText(frame, bg_status, (25, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg_color, 2)
    
    # Separator
    cv2.line(frame, (25, 150), (635, 150), (0, 255, 255), 1)
    
    # Kontrol
    cv2.putText(frame, "KONTROL WARNA:", (25, 175),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    controls_color = [
        ("1", "Hijau"), ("2", "Biru"), ("3", "Merah"), ("4", "Kuning")
    ]
    x_off = 25
    for key, label in controls_color:
        highlight = (color_mode == label)
        color = (0, 255, 255) if highlight else (150, 150, 150)
        cv2.putText(frame, f"[{key}]{label}", (x_off, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1 if not highlight else 2)
        x_off += 100
    
    cv2.putText(frame, "KONTROL BACKGROUND:", (25, 225),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "[SPACE] Toggle BG  [B] Ganti BG  [Q] Keluar", (25, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def draw_hsv_info(frame, h, w):
    """Menggambar informasi tentang HSV"""
    x_start = w - 350
    y_start = 10
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (w - 10, 180), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x_start, y_start), (w - 10, 180), (255, 200, 0), 2)
    
    cv2.putText(frame, "RUANG WARNA HSV", (x_start + 15, y_start + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    
    info_texts = [
        "H: Hue (Warna 0-179)",
        "S: Saturation (0-255)",
        "V: Value (0-255)",
        "",
        "Keunggulan HSV:",
        "> Tahan perubahan cahaya",
        "> Deteksi warna akurat"
    ]
    
    y_off = y_start + 55
    for text in info_texts:
        size = 0.4 if text.startswith(">") else 0.45
        color = (200, 200, 200) if text.startswith(">") else (255, 255, 255)
        cv2.putText(frame, text, (x_start + 15, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
        y_off += 20 if text else 10

print("=" * 70)
print("TUGAS 2: INTERAKSI BERBASIS DETEKSI WARNA HSV")
print("=" * 70)
print("\nKonsep yang Diimplementasikan:")
print("[✓] Konversi BGR ke HSV (cv2.cvtColor)")
print("[✓] Thresholding warna (cv2.inRange)")
print("[✓] Operasi morfologi Opening & Closing")
print("[✓] Deteksi kontur (cv2.findContours)")
print("[✓] Background replacement real-time")
print("\n" + "=" * 70)
print("KONTROL:")
print("  1-4 : Pilih warna deteksi (Hijau/Biru/Merah/Kuning)")
print("  SPACE : Toggle background aktif/nonaktif")
print("  B : Ganti background")
print("  Q : Keluar")
print("=" * 70)

# Generate backgrounds
print("\n[*] Generating backgrounds...")
h_bg, w_bg = 720, 1280
backgrounds["Gradient"] = create_gradient_background(h_bg, w_bg)
backgrounds["Space"] = create_space_background(h_bg, w_bg)
backgrounds["Sunset"] = create_sunset_background(h_bg, w_bg)
backgrounds["Matrix"] = create_matrix_background(h_bg, w_bg)
print("[✓] Backgrounds ready!")

background_list = list(backgrounds.keys())
bg_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Deteksi objek warna
    mask, largest_contour, area = detect_color_object(frame, current_color_mode)
    
    detected = largest_contour is not None
    
    # Aplikasikan background replacement jika aktif
    display_frame = frame.copy()
    
    if background_enabled and detected:
        # Inverse mask: area yang BUKAN objek terdeteksi
        mask_inverse = cv2.bitwise_not(mask)
        
        # Ganti background
        current_bg = backgrounds[current_background]
        display_frame = apply_background_replacement(frame, mask_inverse, current_bg)
    
    # Gambar kontur objek yang terdeteksi
    if detected:
        color_bgr = color_ranges[current_color_mode]["color_bgr"]
        cv2.drawContours(display_frame, [largest_contour], -1, color_bgr, 3)
        
        # Bounding box
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), color_bgr, 2)
        cv2.putText(display_frame, f"{current_color_mode}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    
    # Gambar UI
    draw_ui(display_frame, current_color_mode, current_background, 
            background_enabled, detected, area, h, w)
    draw_hsv_info(display_frame, h, w)
    
    # Tampilkan mask untuk debugging
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_display, "Mask Hasil Thresholding & Morfologi", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Tampilkan windows
    cv2.imshow('Tugas 2: Deteksi Warna & Background', display_frame)
    cv2.imshow('Mask (HSV + Morfologi)', mask_display)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n[✓] Keluar dari program...")
        break
    elif key == ord('1'):
        current_color_mode = "Hijau"
        print(f"[→] Deteksi warna: {current_color_mode}")
    elif key == ord('2'):
        current_color_mode = "Biru"
        print(f"[→] Deteksi warna: {current_color_mode}")
    elif key == ord('3'):
        current_color_mode = "Merah"
        print(f"[→] Deteksi warna: {current_color_mode}")
    elif key == ord('4'):
        current_color_mode = "Kuning"
        print(f"[→] Deteksi warna: {current_color_mode}")
    elif key == ord(' '):  # Spacebar
        background_enabled = not background_enabled
        status = "AKTIF" if background_enabled else "NONAKTIF"
        print(f"[→] Background replacement: {status}")
    elif key == ord('b') or key == ord('B'):
        bg_index = (bg_index + 1) % len(background_list)
        current_background = background_list[bg_index]
        print(f"[→] Background: {current_background}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("RINGKASAN IMPLEMENTASI:")
print("=" * 70)
print("✓ Konversi Ruang Warna:")
print("  - BGR → HSV menggunakan cv2.cvtColor()")
print("  - HSV lebih robust terhadap perubahan cahaya")
print("\n✓ Thresholding Warna:")
print("  - cv2.inRange() untuk deteksi range warna HSV")
print("  - Range disesuaikan untuk setiap warna")
print("\n✓ Operasi Morfologi:")
print("  - Opening (MORPH_OPEN): Hapus noise kecil")
print("  - Closing (MORPH_CLOSE): Tutup lubang kecil")
print("\n✓ Deteksi Kontur:")
print("  - cv2.findContours() untuk menemukan objek")
print("  - Filter berdasarkan area minimum")
print("\n✓ Background Replacement:")
print("  - Inverse mask untuk area background")
print("  - Blend frame dan background menggunakan mask")
print("=" * 70)
print("\n[✓] Program selesai! Terima kasih!")