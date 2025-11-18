import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mode filter
current_mode = "Normal"

def apply_average_blur(frame, kernel_size):
    """Terapkan Average Blurring dengan kernel size tertentu"""
    return cv2.blur(frame, (kernel_size, kernel_size))

def apply_gaussian_blur_custom(frame, kernel_size=9, sigma=0):
    """
    Terapkan Gaussian Blurring menggunakan kernel custom
    Menggunakan cv2.getGaussianKernel() dan cv2.filter2D() seperti yang diwajibkan
    """
    # Buat kernel Gaussian 1D untuk sumbu X dan Y
    kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
    
    # Buat kernel 2D dengan perkalian outer product
    kernel_2d = kernel_x * kernel_y.T
    
    # Terapkan konvolusi menggunakan cv2.filter2D()
    result = cv2.filter2D(frame, -1, kernel_2d)
    
    return result

def apply_sharpening(frame):
    """
    Terapkan filter Sharpening menggunakan kernel custom
    """
    # Kernel sharpening sesuai spesifikasi tugas
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Terapkan konvolusi
    sharpened = cv2.filter2D(frame, -1, kernel_sharpen)
    
    return sharpened

def draw_info_panel(frame, mode, h, w):
    """Menggambar panel informasi di layar"""
    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (600, 220), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Border panel
    cv2.rectangle(frame, (10, 10), (600, 220), (100, 255, 100), 3)
    
    # Title
    cv2.putText(frame, "TUGAS 1: SMOOTHING & BLURRING", (25, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
    
    # Current mode dengan highlight
    mode_color = (0, 255, 255) if mode != "Normal" else (255, 255, 255)
    cv2.putText(frame, f"Mode Aktif: {mode}", (25, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Separator line
    cv2.line(frame, (25, 100), (585, 100), (100, 255, 100), 1)
    
    # Kontrol keyboard
    controls = [
        ("0", "Normal (Tanpa Filter)"),
        ("1", "Average Blur 5x5"),
        ("2", "Average Blur 9x9"),
        ("3", "Gaussian Blur 9x9"),
        ("4", "Sharpening"),
        ("Q", "Keluar")
    ]
    
    y_offset = 130
    for key, desc in controls:
        # Background untuk key
        cv2.rectangle(frame, (25, y_offset - 18), (65, y_offset + 2), (50, 200, 50), -1)
        cv2.putText(frame, key, (35, y_offset - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Description
        cv2.putText(frame, f": {desc}", (75, y_offset - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        y_offset += 25 if key != "2" else 30

def draw_kernel_visualization(frame, mode, h, w):
    """Menggambar visualisasi kernel yang sedang digunakan"""
    if mode == "Normal":
        return
    
    # Position untuk visualisasi kernel
    x_start = w - 280
    y_start = 10
    cell_size = 40
    
    # Background untuk visualisasi
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 10, y_start), (w - 10, y_start + 200), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (x_start - 10, y_start), (w - 10, y_start + 200), (255, 200, 0), 2)
    
    # Title
    cv2.putText(frame, "Kernel:", (x_start, y_start + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    
    y_start += 40
    
    # Definisi kernel untuk visualisasi
    if mode == "Average Blur 5x5":
        kernel_size = 5
        kernel_text = "1/25"
        cv2.putText(frame, "5x5 Average", (x_start, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25
        for i in range(3):  # Tampilkan 3x3 representasi
            for j in range(3):
                x = x_start + j * 35
                y = y_start + i * 25
                cv2.putText(frame, kernel_text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    elif mode == "Average Blur 9x9":
        kernel_text = "1/81"
        cv2.putText(frame, "9x9 Average", (x_start, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25
        for i in range(3):
            for j in range(3):
                x = x_start + j * 35
                y = y_start + i * 25
                cv2.putText(frame, kernel_text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    elif mode == "Gaussian Blur 9x9":
        cv2.putText(frame, "9x9 Gaussian", (x_start, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25
        # Representasi bobot Gaussian (lebih besar di tengah)
        gaussian_repr = [
            ["0.03", "0.11", "0.03"],
            ["0.11", "0.44", "0.11"],
            ["0.03", "0.11", "0.03"]
        ]
        for i in range(3):
            for j in range(3):
                x = x_start + j * 35
                y = y_start + i * 25
                cv2.putText(frame, gaussian_repr[i][j], (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    elif mode == "Sharpening":
        cv2.putText(frame, "3x3 Sharpen", (x_start, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25
        sharpen_kernel = [
            ["0", "-1", "0"],
            ["-1", "5", "-1"],
            ["0", "-1", "0"]
        ]
        for i in range(3):
            for j in range(3):
                x = x_start + j * 35
                y = y_start + i * 25
                cv2.putText(frame, sharpen_kernel[i][j], (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

print("=" * 70)
print("TUGAS 1: IMPLEMENTASI SMOOTHING DAN BLURRING REAL-TIME")
print("=" * 70)
print("\nKonsep yang Diimplementasikan:")
print("[✓] Average Blurring (5x5 dan 9x9)")
print("[✓] Gaussian Blurring dengan kernel custom (cv2.getGaussianKernel)")
print("[✓] Konvolusi menggunakan cv2.filter2D()")
print("[✓] Sharpening dengan kernel custom")
print("[✓] Kontrol real-time via keyboard")
print("\n" + "=" * 70)
print("KONTROL:")
print("  0 - Normal (Tanpa Filter)")
print("  1 - Average Blur 5x5")
print("  2 - Average Blur 9x9")
print("  3 - Gaussian Blur 9x9 (Custom Kernel)")
print("  4 - Sharpening")
print("  Q - Keluar")
print("=" * 70)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Terapkan filter sesuai mode
    processed_frame = frame.copy()
    
    if current_mode == "Average Blur 5x5":
        processed_frame = apply_average_blur(frame, 5)
    elif current_mode == "Average Blur 9x9":
        processed_frame = apply_average_blur(frame, 9)
    elif current_mode == "Gaussian Blur 9x9":
        # Menggunakan implementasi custom dengan cv2.getGaussianKernel dan filter2D
        processed_frame = apply_gaussian_blur_custom(frame, kernel_size=9, sigma=0)
    elif current_mode == "Sharpening":
        processed_frame = apply_sharpening(frame)
    
    # Gambar UI
    draw_info_panel(processed_frame, current_mode, h, w)
    draw_kernel_visualization(processed_frame, current_mode, h, w)
    
    # Tampilkan frame
    cv2.imshow('Tugas 1: Smoothing & Blurring', processed_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n[✓] Keluar dari program...")
        break
    elif key == ord('0'):
        current_mode = "Normal"
        print(f"[→] Mode: {current_mode}")
    elif key == ord('1'):
        current_mode = "Average Blur 5x5"
        print(f"[→] Mode: {current_mode}")
    elif key == ord('2'):
        current_mode = "Average Blur 9x9"
        print(f"[→] Mode: {current_mode}")
    elif key == ord('3'):
        current_mode = "Gaussian Blur 9x9"
        print(f"[→] Mode: {current_mode} (Custom Kernel)")
    elif key == ord('4'):
        current_mode = "Sharpening"
        print(f"[→] Mode: {current_mode}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("RINGKASAN IMPLEMENTASI:")
print("=" * 70)
print("✓ Average Blurring:")
print("  - Menggunakan cv2.blur() dengan kernel 5x5 dan 9x9")
print("  - Semua elemen kernel bernilai sama (1/k²)")
print("\n✓ Gaussian Blurring:")
print("  - Kernel dibuat dengan cv2.getGaussianKernel()")
print("  - Konvolusi diterapkan dengan cv2.filter2D()")
print("  - Bobot lebih tinggi di pusat kernel")
print("\n✓ Sharpening:")
print("  - Kernel custom: [[0,-1,0], [-1,5,-1], [0,-1,0]]")
print("  - Meningkatkan detail dan edge pada citra")
print("=" * 70)
print("\n[✓] Program selesai! Terima kasih!")