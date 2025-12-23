# VTuber Motion Capture System (MediaPipe to VSeeFace via OSC)

**Pengembang:** Gilang Gallan Indrana  
**NRP:** 5024231030  
**Departemen:** Teknik Komputer - ITS  
**Mata Kuliah:** Pengolahan Citra dan Video

---

# 🎥 Demo Video

Berikut adalah demonstrasi hasil tracking:
**[https://drive.google.com/file/d/12_pPIChOhFAYVkCDjadbW_gulUcg3nK6/view?usp=sharing]** 
---

# 📌 Daftar Isi
1. [Deskripsi Proyek](#-deskripsi-proyek)
2. [Evolusi Proyek: Beta hingga Final](#-evolusi-proyek-beta-hingga-final)
3. [Fitur Unggulan: Dual-Mode Tracking](#-fitur-unggulan-dual-mode-tracking)
4. [Arsitektur & Alur Kerja](#-arsitektur--alur-kerja)
5. [Detail Teknis Implementasi](#-detail-teknis-implementasi)
6. [Prasyarat & Instalasi](#-prasyarat--instalasi)
7. [Konfigurasi VSeeFace](#-konfigurasi-vseeface)
8. [Cara Menjalankan Program](#-cara-menjalankan-program)
9. [Kesimpulan](#-kesimpulan)

---

# 📘 Deskripsi Proyek

Proyek ini adalah sistem **Markerless Motion Capture** yang dirancang untuk menggerakkan avatar VTuber 3D secara real-time hanya dengan menggunakan satu webcam standar. Sistem ini menghilangkan kebutuhan akan peralatan *motion capture* berbasis sensor fisik (seperti jas mocap atau tracker VR) yang mahal.

Program ini membaca input video, mendeteksi titik-titik kunci tubuh manusia (landmark) menggunakan **MediaPipe Holistic**, memproses data kinematika, dan mengirimkan data rotasi tulang (Bone Rotation) ke aplikasi **VSeeFace** melalui protokol **VMC (Virtual Motion Capture)** berbasis OSC.

---

# 🔄 Evolusi Proyek: Beta hingga Final

Proyek ini dikembangkan melalui dua tahap utama yang menunjukkan progres dari visualisasi sederhana hingga integrasi aplikasi VTuber profesional.

## 📁 ProyekBeta.py - Versi Awal (Stickman Visualization)

### Overview
File **ProyekBeta.py** adalah implementasi awal sistem yang fokus pada **deteksi dan visualisasi** gerakan manusia menggunakan representasi stickman. Versi ini dibuat sebagai proof-of-concept untuk memvalidasi kemampuan deteksi MediaPipe sebelum integrasi dengan sistem VTuber.

### Fitur Utama ProyekBeta.py:
1. **Full Body Detection**
   - Deteksi pose 33 titik tubuh lengkap (kepala hingga kaki)
   - Tracking untuk multiple persons (multi-orang dalam satu frame)
   
2. **Animated Stickman Rendering**
   - Visualisasi real-time gerakan dalam bentuk stickman sederhana
   - Animasi wajah dengan detail:
     * **Kedipan mata** - Eye Aspect Ratio (EAR) detection
     * **Gerakan mulut** - Deteksi buka/tutup mulut secara dinamis
     * **Ekspresi alis** - Tracking gerakan eyebrow
   
3. **Hand & Finger Tracking**
   - Deteksi 21 landmark per tangan
   - Visualisasi jari-jari dengan struktur skeletal lengkap
   - Support untuk kedua tangan secara simultan
   
4. **Dual Window Display**
   - **Window 1:** Output kamera dengan overlay landmark MediaPipe
   - **Window 2:** Pure stickman visualization dengan background hitam

### Keunggulan Versi Beta:
- ✅ **Tidak memerlukan software eksternal** - Standalone application
- ✅ **Lightweight** - Cocok untuk PC dengan spesifikasi rendah
- ✅ **Educational** - Visualisasi jelas untuk memahami deteksi pose
- ✅ **Multi-person capable** - Dapat mendeteksi beberapa orang sekaligus

### Teknologi yang Digunakan:
```python
- MediaPipe Pose (33 landmarks)
- MediaPipe Face Mesh (468 landmarks)
- MediaPipe Hands (21 landmarks per tangan)
- OpenCV untuk rendering dan display
- NumPy untuk kalkulasi geometris
```

### Screenshot Fitur Beta:
- **Deteksi Pose**: Garis hijau menghubungkan sendi tubuh
- **Face Animation**: Mata dan mulut bergerak sesuai ekspresi real
- **Hand Skeleton**: Struktur jari tergambar detail dengan warna berbeda

---

## 📁 Proyek.py - Versi Final (VSeeFace Integration)

### Overview
File **Proyek.py** adalah evolusi dari versi beta yang mengintegrasikan deteksi gerakan dengan aplikasi **VSeeFace** melalui protokol OSC/VMC. Versi ini mentransformasi data deteksi menjadi data animasi avatar 3D profesional.

### Peningkatan dari Beta ke Final:
| Aspek | ProyekBeta.py | Proyek.py |
|-------|---------------|-----------|
| **Output** | Stickman 2D | Avatar VTuber 3D |
| **Protokol** | Tidak ada | OSC/VMC Protocol |
| **Stabilisasi** | Tidak ada | Kalman Filter |
| **Target FPS** | Unlimited | 30 FPS (optimized) |
| **Bone Rotation** | Tidak ada | Quaternion calculation |
| **Mode Tracking** | Full body only | Dual-mode (Half/Full) |
| **Eye Tracking** | Basic blink | 6DOF + Iris tracking |
| **Finger Control** | Visualization only | Full articulation |

### Fitur Tambahan Versi Final:
1. **Advanced Head Tracking (6DOF)**
   - Estimasi pose 3D menggunakan SolvePnP
   - Rotasi kepala dan leher terpisah untuk gerakan natural
   
2. **Signal Smoothing**
   - Kalman Filter untuk semua joint
   - Eliminasi jitter dan noise webcam
   
3. **Body Decomposition**
   - Rotasi Spine (60%) dan Chest (40%) terdistribusi
   - Gerakan membungkuk yang lebih organik
   
4. **Professional Integration**
   - Real-time data streaming ke VSeeFace
   - Compatible dengan avatar VRM standar
   - Live streaming ready

### Kapan Menggunakan Versi Mana?

**Gunakan ProyekBeta.py jika:**
- 🎓 Belajar computer vision dan pose detection
- 🖥️ PC tidak kuat menjalankan VSeeFace
- 🔬 Melakukan eksperimen dengan deteksi multi-person
- 📊 Membutuhkan visualisasi debug yang jelas

**Gunakan Proyek.py jika:**
- 🎥 Live streaming sebagai VTuber
- 🎮 Presentasi dengan avatar 3D
- 💼 Aplikasi profesional (virtual meeting, dll)
- 🎭 Membutuhkan kualitas animasi tinggi

---

# ⚔️ Fitur Unggulan: Dual-Mode Tracking

Salah satu fitur utama dari sistem final (Proyek.py) adalah kemampuan untuk beralih antara dua mode tracking secara dinamis sesuai dengan kebutuhan pengguna (Streaming duduk vs. Berdiri).

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
5.  **Data Transmission (Proyek.py only):**
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

### D. Eye Aspect Ratio (EAR) untuk Deteksi Kedipan
Digunakan pada kedua versi untuk mendeteksi kapan mata tertutup:
```python
def calculate_ear(face_landmarks, indices, img_w, img_h):
    # Menghitung rasio vertikal/horizontal mata
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)
```
Threshold: EAR < 0.15 = Mata Tertutup

---

# 🛠️ Prasyarat & Instalasi

### Hardware
* Webcam (Min. 720p disarankan)
* PC/Laptop (Windows/Linux/Mac)

### Software Requirement
1.  **Python 3.8** atau lebih baru.
2.  **VSeeFace** (Hanya untuk Proyek.py - Aplikasi Receiver VTuber).

### Instalasi Library

#### Untuk ProyekBeta.py:
```bash
pip install opencv-python mediapipe numpy
```

#### Untuk Proyek.py (Tambahan):
```bash
pip install opencv-python mediapipe numpy python-osc
```

---

# ⚙️ Konfigurasi VSeeFace

> **⚠️ Catatan:** Konfigurasi ini hanya diperlukan untuk **Proyek.py**. ProyekBeta.py tidak memerlukan VSeeFace.

Agar avatar dapat bergerak, VSeeFace harus diatur sebagai *Receiver*:

1.  Buka **VSeeFace** dan muat avatar `.vrm` Anda.
2.  Klik **Settings** > **General Settings**.
3.  Scroll ke bawah hingga menemukan bagian **OSC / VMC Protocol Receiver**.
4.  Centang **Enable**.
5.  Pastikan **Receiver Port** diisi `39539`.
6.  Tutup menu settings. Avatar sekarang siap menerima data.

---

# ▶️ Cara Menjalankan Program

## Menjalankan ProyekBeta.py (Stickman Visualization)

1.  Pastikan webcam terhubung.
2.  Jalankan script Python:
    ```bash
    python ProyekBeta.py
    ```
3.  Dua jendela akan muncul:
    - **Detection Output (Original)**: Kamera dengan overlay MediaPipe
    - **Stickman Filter (Animated)**: Visualisasi stickman murni

### 🎮 Kontrol ProyekBeta.py
| Tombol | Fungsi |
| :---: | :--- |
| **`Q`** | Keluar dari program |

---

## Menjalankan Proyek.py (VTuber Tracking)

1.  Pastikan webcam terhubung.
2.  Jalankan VSeeFace terlebih dahulu.
3.  Jalankan script Python:
    ```bash
    python Proyek.py
    ```
4.  Jendela kamera akan muncul dengan overlay deteksi warna-warni:
    - **Cyan**: Face landmarks
    - **Magenta**: Pose/Body
    - **Kuning**: Hand landmarks

### 🎮 Kontrol Proyek.py
| Tombol | Fungsi | Deskripsi |
| :---: | :--- | :--- |
| **`H`** | **Half Body Mode** | Mode Setengah Badan (Duduk). Fokus Kepala & Tangan. |
| **`F`** | **Full Body Mode** | Mode Seluruh Badan (Berdiri). Mengaktifkan Kaki. |
| **`Q`** | **Quit** | Menghentikan program dan menutup kamera. |

---

# 📝 Kesimpulan

Proyek ini berhasil mengimplementasikan sistem *Computer Vision* yang kompleks menjadi aplikasi praktis yang menghibur. Beberapa poin pencapaian utama dalam tugas ini meliputi:

1.  **Penerapan Teori:** Berhasil menerapkan algoritma **Perspective-n-Point (PnP)** untuk estimasi pose 3D kepala dari citra 2D.
2.  **Matematika Lanjut:** Penggunaan **Quaternion** dan **Matriks Rotasi** terbukti efektif menghindari *Gimbal Lock* yang sering terjadi pada animasi Euler angles biasa.
3.  **Optimasi Noise:** Implementasi **Kalman Filter** memberikan dampak signifikan pada kehalusan gerakan, menjadikan avatar terlihat hidup dan tidak "gemetar" (jittery).
4.  **Interaktivitas:** Fitur *Dual-Mode* (Setengah Badan vs. Seluruh Badan) memberikan fleksibilitas nyata bagi pengguna dalam berbagai skenario penggunaan.
5.  **Iterasi Pengembangan:** Evolusi dari ProyekBeta.py (visualisasi) ke Proyek.py (aplikasi profesional) menunjukkan pemahaman mendalam tentang pipeline pengembangan sistem real-time.

Sistem ini membuktikan bahwa webcam standar dapat dimanfaatkan sebagai alat *motion capture* yang andal dengan pemrosesan citra yang tepat.

---

## 📊 Perbandingan Akhir

| Fitur | ProyekBeta.py | Proyek.py |
|-------|---------------|-----------|
| **Tujuan** | Proof-of-concept & Learning | Production-ready VTuber |
| **Kompleksitas** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Output** | Stickman 2D | Avatar 3D |
| **Eksternal Software** | ❌ Tidak perlu | ✅ VSeeFace |
| **Stabilisasi** | ❌ Raw data | ✅ Kalman Filter |
| **Use Case** | Edukasi, Debug | Streaming, Profesional |

---

*Dokumen ini disusun oleh Gilang Gallan Indrana - 5024231030*  
*Departemen Teknik Komputer - Institut Teknologi Sepuluh Nopember*
