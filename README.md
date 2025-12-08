# VTuber Motion Capture System (MediaPipe to VSeeFace via OSC)

**Pengembang:** Gilang Gallan Indrana  
**NRP:** 5024231030  
**Departemen:** Teknik Komputer - ITS  
**Mata Kuliah:** Pengolahan Citra dan Video

---

# 📌 Daftar Isi
1. [Deskripsi Proyek](#-deskripsi-proyek)
2. [Fitur Unggulan: Dual-Mode Tracking](#-fitur-unggulan-dual-mode-tracking)
3. [Arsitektur & Alur Kerja](#-arsitektur--alur-kerja)
4. [Detail Teknis Implementasi](#-detail-teknis-implementasi)
5. [Prasyarat & Instalasi](#-prasyarat--instalasi)
6. [Konfigurasi VSeeFace](#-konfigurasi-vseeface)
7. [Cara Menjalankan Program](#-cara-menjalankan-program)
8. [Demo Video](#-demo-video)
9. [Kesimpulan](#-kesimpulan)

---

# 📘 Deskripsi Proyek

Proyek ini adalah sistem **Markerless Motion Capture** yang dirancang untuk menggerakkan avatar VTuber 3D secara real-time hanya dengan menggunakan satu webcam standar. Sistem ini menghilangkan kebutuhan akan peralatan *motion capture* berbasis sensor fisik (seperti jas mocap atau tracker VR) yang mahal.

Program ini membaca input video, mendeteksi titik-titik kunci tubuh manusia (landmark) menggunakan **MediaPipe Holistic**, memproses data kinematika, dan mengirimkan data rotasi tulang (Bone Rotation) ke aplikasi **VSeeFace** melalui protokol **VMC (Virtual Motion Capture)** berbasis OSC.

---

# ⚔️ Fitur Unggulan: Dual-Mode Tracking

Salah satu fitur utama dari sistem ini adalah kemampuan untuk beralih antara dua mode tracking secara dinamis sesuai dengan kebutuhan pengguna (Streaming duduk vs. Berdiri).

### 1. Mode Setengah Badan (Half Body Tracking) - *Default*
Mode ini diaktifkan dengan menekan tombol **`H`**.
* **Fokus:** Mengutamakan deteksi Kepala, Torso (Badan Atas), Lengan, dan Jari Tangan.
* **Logika:** Sistem secara sengaja mengabaikan data landmark kaki meskipun terdeteksi oleh kamera.
* **Kegunaan:** Sangat optimal untuk skenario *Live Streaming* atau *Video Conference* di mana pengguna duduk di depan meja.
* **Keunggulan:** Mengurangi noise gerakan yang tidak perlu pada kaki avatar saat pengguna sedang duduk diam, sehingga avatar terlihat lebih tenang dan profesional.

### 2. Mode Seluruh Badan (Full Body Tracking)
Mode ini diaktifkan dengan menekan tombol **`F`**.
* **Fokus:** Melacak seluruh 33 titik tubuh termasuk Pinggul, Lutut, dan Pergelangan Kaki.
* **Logika:** Sistem mengaktifkan kalkulasi rotasi untuk tulang:
    * `LeftUpperLeg`, `LeftLowerLeg`, `LeftFoot`
    * `RightUpperLeg`, `RightLowerLeg`, `RightFoot`
* **Kegunaan:** Cocok untuk presentasi berdiri, menari, atau demonstrasi gerakan penuh.
* **Syarat:** Pengguna harus berdiri cukup jauh dari kamera agar seluruh tubuh (dari kepala hingga kaki) masuk ke dalam frame kamera.

---

# 🔄 Arsitektur & Alur Kerja

Pipeline pemrosesan data berjalan secara real-time (target 30 FPS):

1.  **Image Acquisition:** Frame diambil dari Webcam menggunakan OpenCV.
2.  **Holistic Detection:** MediaPipe mendeteksi 468 titik wajah, 33 titik pose, dan 21 titik tangan per sisi.
3.  **Kinematic Calculation:**
    * **Head 6DOF:** Menggunakan algoritma *SolvePnP* untuk menghitung rotasi 3D kepala.
    * **Eye & Mouth:** Analisis geometri wajah untuk kedipan mata (*Eye Aspect Ratio*) dan bukaan mulut.
    * **Limb Rotation:** Menghitung vektor arah tulang (Vector Math) dibandingkan dengan vektor referensi T-Pose.
4.  **Signal Processing (Stabilization):**
    * Raw data dari kamera seringkali memiliki *jitter* (getaran).
    * Setiap sendi diproses melalui **Kalman Filter** untuk menghaluskan gerakan tanpa menambah latensi yang signifikan.
5.  **Data Transmission:**
    * Data rotasi dikonversi menjadi **Quaternion** `(x, y, z, w)`.
    * Dikirim via UDP ke port `39539` (VMC Protocol).

---

# 🔬 Detail Teknis Implementasi

### A. Stabilisasi Gerakan (Kalman Filter)
Untuk mengatasi noise deteksi webcam, digunakan kelas `Stabilizer` kustom.
```python
# Implementasi Kalman Filter untuk 1 variabel (misal: sudut siku)
self.filter = cv2.KalmanFilter(2, 1, 0) # 2 State (Posisi, Kecepatan), 1 Measurement
self.filter.processNoiseCov = 1e-5      # Kovarian noise proses (responsivitas)
self.filter.measurementNoiseCov = 1e-1  # Kovarian noise pengukuran (kehalusan)
```
---

### B. Konversi Euler ke Quaternion
VSeeFace (Unity Engine) menggunakan Quaternion untuk rotasi guna menghindari *Gimbal Lock*.
```python
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    # ... (perhitungan qy, qz, qw)
    return [qx, qy, qz, qw]
```
### C. Logika Spine & Chest
Agar gerakan tubuh terlihat alami, rotasi bahu pengguna tidak hanya memutar satu tulang, tetapi didistribusikan:
* **Spine (Tulang Belakang):** Menerima 60% dari total rotasi tubuh.
* **Chest (Dada):** Menerima 40% sisanya.
Hal ini membuat avatar membungkuk secara luwes (organic bending).

---

# 🛠️ Prasyarat & Instalasi

### Hardware
* Webcam (Min. 720p disarankan)
* PC/Laptop (Windows/Linux/Mac)

### Software Requirement
1.  **Python 3.8** atau lebih baru.
2.  **VSeeFace** (Aplikasi Receiver VTuber).

### Instalasi Library
Buka terminal/command prompt dan jalankan:
```bash
pip install opencv-python mediapipe numpy python-osc
```
---

# ⚙️ Konfigurasi VSeeFace

Agar avatar dapat bergerak, VSeeFace harus diatur sebagai *Receiver*:

1.  Buka **VSeeFace** dan muat avatar `.vrm` Anda.
2.  Klik **Settings** > **General Settings**.
3.  Scroll ke bawah hingga menemukan bagian **OSC / VMC Protocol Receiver**.
4.  Centang **Enable**.
5.  Pastikan **Receiver Port** diisi `39539`.
6.  Tutup menu settings. Avatar sekarang siap menerima data.

---

# ▶️ Cara Menjalankan Program

1.  Pastikan webcam terhubung.
2.  Jalankan VSeeFace terlebih dahulu.
3.  Jalankan script Python:
    ```bash
    python Proyek.py
    ```
4.  Jendela kamera akan muncul dengan overlay deteksi.

### 🎮 Kontrol Keyboard
Saat jendela kamera aktif (fokus), gunakan tombol berikut:

| Tombol | Fungsi | Deskripsi |
| :---: | :--- | :--- |
| **`H`** | **Half Body Mode** | Mode Setengah Badan (Duduk). Fokus Kepala & Tangan. |
| **`F`** | **Full Body Mode** | Mode Seluruh Badan (Berdiri). Mengaktifkan Kaki. |
| **`Q`** | **Quit** | Menghentikan program dan menutup kamera. |

---

# 🎥 Demo Video

Berikut adalah demonstrasi hasil tracking:
**[-]**

---

# 📝 Kesimpulan

Proyek ini berhasil mengimplementasikan sistem *Computer Vision* yang kompleks menjadi aplikasi praktis yang menghibur. Beberapa poin pencapaian utama dalam tugas ini meliputi:

1.  **Penerapan Teori:** Berhasil menerapkan algoritma **Perspective-n-Point (PnP)** untuk estimasi pose 3D kepala dari citra 2D.
2.  **Matematika Lanjut:** Penggunaan **Quaternion** dan **Matriks Rotasi** terbukti efektif menghindari *Gimbal Lock* yang sering terjadi pada animasi Euler angles biasa.
3.  **Optimasi Noise:** Implementasi **Kalman Filter** memberikan dampak signifikan pada kehalusan gerakan, menjadikan avatar terlihat hidup dan tidak "gemetar" (jittery).
4.  **Interaktivitas:** Fitur *Dual-Mode* (Setengah Badan vs. Seluruh Badan) memberikan fleksibilitas nyata bagi pengguna dalam berbagai skenario penggunaan.

Sistem ini membuktikan bahwa webcam standar dapat dimanfaatkan sebagai alat *motion capture* yang andal dengan pemrosesan citra yang tepat.

---
*Dokumen ini disusun oleh Gilang Gallan Indrana (5024231030)*
