# PROYEK AKHIR PENGOLAHAN CITRA & VIDEO
## VTuber Motion Capture System (MediaPipe to VSeeFace via OSC)

### Nama: Gilang Gallan Indrana  
### NRP: 5024231030
### Mata Kuliah: Pengolahan Citra dan Video
### Departemen: Teknik Komputer - ITS

---

# 📌 DAFTAR ISI
1. Deskripsi Proyek
2. Alur Kerja Sistem (Pipeline)
3. Fitur Utama
   - Face Tracking & Gaze
   - Body & Limb Tracking (IK Solver)
   - Hand & Finger Tracking
   - Stabilization (Kalman Filter)
4. Persyaratan Sistem & Instalasi
5. Konfigurasi VSeeFace
6. Cara Menjalankan
7. Kontrol Keyboard
8. Demo Video

---

# 📘 DESKRIPSI PROYEK

Proyek ini adalah sistem **Markerless Motion Capture** real-time yang mengubah input webcam standar menjadi data gerakan animasi 3D. Sistem ini dirancang untuk menggerakkan avatar VTuber di aplikasi **VSeeFace** tanpa memerlukan peralatan VR mahal.

Program ini dibangun menggunakan **Python** dan **OpenCV**, memanfaatkan **MediaPipe Holistic** untuk ekstraksi fitur tubuh, dan mengirimkan data rotasi tulang (Bone Rotation) menggunakan protokol **VMC (Virtual Motion Capture)** berbasis **OSC (Open Sound Control)**.

---

# 🔄 ALUR KERJA SISTEM (PIPELINE)

Sistem bekerja dengan alur pemrosesan data sebagai berikut:

1.  **Input Akuisisi**
    - Mengambil frame video dari Webcam menggunakan OpenCV.
2.  **Deteksi Fitur (Computer Vision)**
    - MediaPipe Holistic mendeteksi 468 titik wajah (Face Mesh), 33 titik pose tubuh, dan 21 titik per tangan.
3.  **Kalkulasi Geometri & Kinematika**
    - **Head Pose:** Menggunakan algoritma *Perspective-n-Point* (PnP) untuk menghitung rotasi kepala (Pitch, Yaw, Roll) dari titik wajah 2D ke model 3D.
    - **Eye Tracking:** Menghitung *Eye Aspect Ratio* (EAR) untuk deteksi kedipan dan vektor posisi iris untuk arah pandangan mata.
    - **Limb Rotation:** Menghitung rotasi sendi (bahu, siku, lutut) dengan membandingkan vektor tulang saat ini terhadap vektor referensi (T-Pose).
    - **Finger Curl:** Menghitung rasio jarak ujung jari ke pergelangan tangan untuk menentukan seberapa menekuk jari tersebut.
4.  **Smoothing (Stabilisasi)**
    - Menerapkan **Kalman Filter** pada setiap sendi dan parameter wajah untuk mengurangi *jitter* (getaran noise) agar gerakan avatar terlihat halus dan natural.
5.  **Transmisi Data**
    - Mengonversi sudut Euler ke **Quaternion** untuk menghindari *Gimbal Lock*.
    - Mengirim data tulang (Bone) via protokol OSC ke `localhost` port `39539`.
6.  **Rendering**
    - VSeeFace menerima data OSC dan menggerakkan model 3D (format .VRM).

---

# 🌟 FITUR UTAMA

## 1. Advanced Face Tracking
- **Head Rotation 6DOF:** Rotasi kepala presisi.
- **Eye Gaze & Blink:** Deteksi lirikan mata dan kedipan halus (*smooth transition*) menggunakan interpolasi.
- **Mouth Movement:** Deteksi buka/tutup mulut berdasarkan jarak vertikal bibir.

## 2. Body Tracking (Half & Full Body)
- **Mode Setengah Badan (Default):** Fokus pada pergerakan kepala, badan bagian atas, dan tangan. Mode ini optimal untuk VTuber yang melakukan streaming sambil duduk.
- **Mode Seluruh Badan:** Mengaktifkan tracking kaki. Mode ini cocok jika pengguna berdiri cukup jauh dari kamera agar seluruh badan terlihat.
- **Spine & Chest Logic:** Logika pergerakan tulang belakang dan dada yang dinamis mengikuti rotasi bahu.

## 3. Hand & Finger Tracking
- Melacak setiap ruas jari secara individual.
- Mendukung gestur tangan kompleks (menggenggam, menunjuk, peace sign, dll).

## 4. Jitter Reduction (Kalman Filter)
- Menggunakan implementasi kelas `Stabilizer` berbasis Kalman Filter untuk meminimalisir noise deteksi kamera.
- Menghasilkan pergerakan avatar yang stabil meskipun pencahayaan kurang optimal.

---

# 🛠️ PERSYARATAN SISTEM & INSTALASI

### Prasyarat Software
1.  **Python 3.8+**
2.  **VSeeFace** (Aplikasi VTuber Receiver) - [Download VSeeFace](https://www.vseeface.icu/)

### Instalasi Library Python
Jalankan perintah berikut di terminal/command prompt:

```bash
pip install opencv-python mediapipe numpy python-osc
