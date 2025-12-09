import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def calculate_distance(point1, point2):
    """Hitung jarak antara dua titik"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_hand_stickman(canvas, hand_landmarks, img_width, img_height, color=(0, 255, 0)):
    """Menggambar jari-jari tangan pada stickman"""
    thickness = 2
    
    def get_hand_point(landmark_id):
        landmark = hand_landmarks.landmark[landmark_id]
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        return (x, y)
    
    # Koneksi antar jari
    finger_connections = [
        # Jempol
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Telunjuk
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Jari tengah
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Jari manis
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Kelingking
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Telapak tangan
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Gambar koneksi jari
    for connection in finger_connections:
        start_point = get_hand_point(connection[0])
        end_point = get_hand_point(connection[1])
        cv2.line(canvas, start_point, end_point, color, thickness)
    
    # Gambar titik sendi jari
    for i in range(21):
        point = get_hand_point(i)
        if i == 0:  # Pergelangan tangan lebih besar
            cv2.circle(canvas, point, 6, color, -1)
        elif i in [4, 8, 12, 16, 20]:  # Ujung jari lebih besar
            cv2.circle(canvas, point, 5, color, -1)
        else:  # Sendi biasa
            cv2.circle(canvas, point, 3, color, -1)

def draw_advanced_stickman(canvas, pose_landmarks, face_landmarks, img_width, img_height):
    """
    Menggambar stickman dengan animasi wajah dan full body tracking
    """
    color = (0, 255, 0)
    thickness = 3
    
    # Fungsi helper untuk konversi landmark
    def get_pose_point(landmark_id):
        landmark = pose_landmarks.landmark[landmark_id]
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        visibility = landmark.visibility
        return (x, y, visibility)
    
    def get_face_point(landmark_id):
        landmark = face_landmarks.landmark[landmark_id]
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        return (x, y)
    
    # ===== BAGIAN TUBUH (POSE) =====
    # Dapatkan koordinat pose
    nose = get_pose_point(0)
    left_eye = get_pose_point(2)
    right_eye = get_pose_point(5)
    left_shoulder = get_pose_point(11)
    right_shoulder = get_pose_point(12)
    left_elbow = get_pose_point(13)
    right_elbow = get_pose_point(14)
    left_wrist = get_pose_point(15)
    right_wrist = get_pose_point(16)
    left_hip = get_pose_point(23)
    right_hip = get_pose_point(24)
    left_knee = get_pose_point(25)
    right_knee = get_pose_point(26)
    left_ankle = get_pose_point(27)
    right_ankle = get_pose_point(28)
    
    # Hitung radius kepala dari pose
    head_radius = int(calculate_distance(left_eye[:2], right_eye[:2]) * 1.5)
    if head_radius < 10:
        head_radius = 30
    
    # Gambar kepala (lingkaran)
    if nose[2] > 0.5:
        cv2.circle(canvas, nose[:2], head_radius, color, thickness)
    
    # ===== ANIMASI WAJAH DETAIL =====
    if face_landmarks:
        # MATA KIRI - dengan kedipan
        left_eye_top = get_face_point(159)
        left_eye_bottom = get_face_point(145)
        left_eye_left = get_face_point(33)
        left_eye_right = get_face_point(133)
        left_eye_center = ((left_eye_left[0] + left_eye_right[0]) // 2,
                          (left_eye_top[1] + left_eye_bottom[1]) // 2)
        
        # Deteksi kedipan mata kiri
        left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)
        left_eye_width = calculate_distance(left_eye_left, left_eye_right)
        left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        
        if left_eye_ratio > 0.15:  # Mata terbuka
            cv2.ellipse(canvas, left_eye_center, 
                       (int(left_eye_width * 0.3), int(left_eye_height * 0.8)),
                       0, 0, 360, color, -1)
        else:  # Mata tertutup/kedip
            cv2.line(canvas, left_eye_left, left_eye_right, color, thickness)
        
        # MATA KANAN - dengan kedipan
        right_eye_top = get_face_point(386)
        right_eye_bottom = get_face_point(374)
        right_eye_left = get_face_point(362)
        right_eye_right = get_face_point(263)
        right_eye_center = ((right_eye_left[0] + right_eye_right[0]) // 2,
                           (right_eye_top[1] + right_eye_bottom[1]) // 2)
        
        # Deteksi kedipan mata kanan
        right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)
        right_eye_width = calculate_distance(right_eye_left, right_eye_right)
        right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        if right_eye_ratio > 0.15:  # Mata terbuka
            cv2.ellipse(canvas, right_eye_center,
                       (int(right_eye_width * 0.3), int(right_eye_height * 0.8)),
                       0, 0, 360, color, -1)
        else:  # Mata tertutup/kedip
            cv2.line(canvas, right_eye_left, right_eye_right, color, thickness)
        
        # MULUT - dengan animasi buka tutup
        mouth_top = get_face_point(13)
        mouth_bottom = get_face_point(14)
        mouth_left = get_face_point(61)
        mouth_right = get_face_point(291)
        mouth_center = ((mouth_left[0] + mouth_right[0]) // 2,
                       (mouth_top[1] + mouth_bottom[1]) // 2)
        
        # Deteksi mulut terbuka
        mouth_height = calculate_distance(mouth_top, mouth_bottom)
        mouth_width = calculate_distance(mouth_left, mouth_right)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        if mouth_ratio > 0.25:  # Mulut terbuka
            cv2.ellipse(canvas, mouth_center,
                       (int(mouth_width * 0.4), int(mouth_height * 1.2)),
                       0, 0, 360, color, thickness)
        else:  # Mulut tertutup (senyum)
            cv2.ellipse(canvas, mouth_center,
                       (int(mouth_width * 0.4), int(mouth_width * 0.2)),
                       0, 0, 180, color, thickness)
        
        # ALIS
        left_eyebrow_inner = get_face_point(70)
        left_eyebrow_outer = get_face_point(46)
        cv2.line(canvas, left_eyebrow_inner, left_eyebrow_outer, color, 2)
        
        right_eyebrow_inner = get_face_point(300)
        right_eyebrow_outer = get_face_point(276)
        cv2.line(canvas, right_eyebrow_inner, right_eyebrow_outer, color, 2)
    
    # ===== TUBUH DAN ANGGOTA BADAN =====
    # Bahu ke bahu (garis horizontal)
    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
        cv2.line(canvas, left_shoulder[:2], right_shoulder[:2], color, thickness)
    
    # Tulang belakang (tengah bahu ke tengah pinggul)
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                       (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_center = ((left_hip[0] + right_hip[0]) // 2,
                  (left_hip[1] + right_hip[1]) // 2)
    cv2.line(canvas, shoulder_center, hip_center, color, thickness)
    
    # Pinggul ke pinggul (garis horizontal)
    if left_hip[2] > 0.5 and right_hip[2] > 0.5:
        cv2.line(canvas, left_hip[:2], right_hip[:2], color, thickness)
    
    # Tangan kiri
    if left_shoulder[2] > 0.5 and left_elbow[2] > 0.5:
        cv2.line(canvas, left_shoulder[:2], left_elbow[:2], color, thickness)
    if left_elbow[2] > 0.5 and left_wrist[2] > 0.5:
        cv2.line(canvas, left_elbow[:2], left_wrist[:2], color, thickness)
    
    # Tangan kanan
    if right_shoulder[2] > 0.5 and right_elbow[2] > 0.5:
        cv2.line(canvas, right_shoulder[:2], right_elbow[:2], color, thickness)
    if right_elbow[2] > 0.5 and right_wrist[2] > 0.5:
        cv2.line(canvas, right_elbow[:2], right_wrist[:2], color, thickness)
    
    # Kaki kiri
    if left_hip[2] > 0.5 and left_knee[2] > 0.5:
        cv2.line(canvas, left_hip[:2], left_knee[:2], color, thickness)
    if left_knee[2] > 0.5 and left_ankle[2] > 0.5:
        cv2.line(canvas, left_knee[:2], left_ankle[:2], color, thickness)
        cv2.circle(canvas, left_ankle[:2], 8, color, -1)  # Kaki
    
    # Kaki kanan
    if right_hip[2] > 0.5 and right_knee[2] > 0.5:
        cv2.line(canvas, right_hip[:2], right_knee[:2], color, thickness)
    if right_knee[2] > 0.5 and right_ankle[2] > 0.5:
        cv2.line(canvas, right_knee[:2], right_ankle[:2], color, thickness)
        cv2.circle(canvas, right_ankle[:2], 8, color, -1)  # Kaki
    
    # Sendi (lingkaran kecil)
    joints = [left_shoulder, right_shoulder, left_elbow, right_elbow,
              left_hip, right_hip, left_knee, right_knee]
    for joint in joints:
        if joint[2] > 0.5:
            cv2.circle(canvas, joint[:2], 6, color, -1)
    
    return left_wrist, right_wrist

print("=" * 60)
print("ADVANCED STICKMAN FILTER WITH HAND TRACKING")
print("=" * 60)
print("Fitur:")
print("[+] Full body tracking (seluruh tubuh)")
print("[+] Hand & finger detection (deteksi tangan & jari)")
print("[+] Animasi mata (kedipan real-time)")
print("[+] Animasi mulut (buka-tutup sesuai gerakan)")
print("[+] Multi-person detection")
print("\nTekan 'q' untuk keluar")
print("=" * 60)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1) as pose, \
     mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses deteksi
        pose_results = pose.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        
        h, w, _ = frame.shape
        stickman_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        detected_people = 0
        left_wrist_pos = None
        right_wrist_pos = None
        
        # Deteksi pose
        if pose_results.pose_landmarks:
            detected_people += 1
            
            # Gambar pose pada frame asli
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Cari face landmarks yang sesuai dengan pose
            face_to_use = None
            if face_results.multi_face_landmarks:
                face_to_use = face_results.multi_face_landmarks[0]
            
            # Gambar stickman dan dapatkan posisi pergelangan tangan
            left_wrist_pos, right_wrist_pos = draw_advanced_stickman(
                stickman_canvas, pose_results.pose_landmarks,
                face_to_use, w, h)
        
        # Gambar face mesh pada frame asli untuk semua wajah
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
        
        # Gambar tangan dan jari
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Gambar pada frame asli
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Gambar pada stickman canvas
                draw_hand_stickman(stickman_canvas, hand_landmarks, w, h)
        
        # Status text - hanya status deteksi
        if detected_people > 0:
            status_text = f"TERDETEKSI: {detected_people} ORANG"
            status_color = (0, 255, 0)
        else:
            status_text = "TIDAK ADA DETEKSI"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        cv2.putText(stickman_canvas, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Hanya petunjuk keluar
        cv2.putText(frame, "Tekan 'q' untuk keluar", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(stickman_canvas, "Tekan 'q' untuk keluar", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Detection Output (Original)', frame)
        cv2.imshow('Stickman Filter (Animated)', stickman_canvas)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n[+] Program selesai! Terima kasih!")