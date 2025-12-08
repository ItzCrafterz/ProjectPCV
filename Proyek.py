import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time

# --- KONFIGURASI UTAMA ---
OSC_IP = "127.0.0.1"
OSC_PORT = 39539
WEBCAM_ID = 0
TARGET_FPS = 30

# ==========================================
# === DATA KALIBRASI FINAL ===
# ==========================================
INVERT_X = 1.0  
INVERT_Y = 1.0 
INVERT_Z = 1.0 

ARM_GAIN_XY = 1.2
ARM_GAIN_Z  = 0.5 

# JARI
FINGER_AXIS_L, FINGER_AXIS_R = 2, 2
FINGER_SIGN_L, FINGER_SIGN_R = 1.0, -1.0
THUMB_AXIS_L, THUMB_AXIS_R = 1, 1
THUMB_SIGN_L, THUMB_SIGN_R = -1.0, -1.0
FINGER_SENSITIVITY = 1.3

# SPINE & BODY
INVERT_BODY_PITCH = -1.0 
BODY_LEAN_SENSITIVITY  = 1.5
BODY_TWIST_SENSITIVITY = 1.2
BODY_BEND_SENSITIVITY  = 1.0
SPINE_RATIO = 0.6 
CHEST_RATIO = 0.4

# --- HELPER FUNCTIONS ---
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

def get_finger_quat(angle, axis_idx):
    s, c = math.sin(angle/2), math.cos(angle/2)
    if axis_idx == 0: return [s, 0, 0, c]
    elif axis_idx == 1: return [0, s, 0, c]
    return [0, 0, s, c]

def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1]
    v_curr = v_curr / norm
    v_rest = np.array(rest_vector)
    v_rest = v_rest / np.linalg.norm(v_rest)
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot)) 
    angle = math.acos(dot)
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    axis = axis / axis_len
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def get_relative_iris(face_landmarks, iris_idx, inner_idx, outer_idx, img_w, img_h):
    iris = np.array([face_landmarks.landmark[iris_idx].x * img_w, face_landmarks.landmark[iris_idx].y * img_h])
    inner = np.array([face_landmarks.landmark[inner_idx].x * img_w, face_landmarks.landmark[inner_idx].y * img_h])
    outer = np.array([face_landmarks.landmark[outer_idx].x * img_w, face_landmarks.landmark[outer_idx].y * img_h])
    eye_width = np.linalg.norm(outer - inner)
    eye_vec = outer - inner
    eye_vec_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-6)
    iris_vec = iris - inner
    proj_x = np.dot(iris_vec, eye_vec_norm)
    norm_x = (proj_x / eye_width) * 2.0 - 1.0
    cross_prod = (eye_vec[0] * (iris[1] - inner[1])) - (eye_vec[1] * (iris[0] - inner[0]))
    dist_y = cross_prod / eye_width
    norm_y = dist_y / (eye_width * 0.3)
    return norm_x, norm_y

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)
    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    curl = (ratio - 1.9) / (0.8 - 1.9)
    return max(0.0, min(1.0, curl)) * FINGER_SENSITIVITY

# --- CLASS STABILIZER ---
class Stabilizer:
    def __init__(self, state_num=2, measure_num=1, cov_process=0.0001, cov_measure=0.1):
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process
        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure
    def update(self, measurement):
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(measurement)]]))
        self.state = self.filter.statePost
        return self.state[0][0]

# --- SMOOTH BLINK CLASS ---
class SmoothBlink:
    def __init__(self, speed=0.15):
        self.current_value = 0.0
        self.target_value = 0.0
        self.speed = speed
        self.last_ear = 0.0
        
    def update(self, ear_value, thresh_close=0.15, thresh_open=0.25, hysteresis=0.02):
        if abs(ear_value - self.last_ear) < hysteresis:
            ear_value = self.last_ear
        self.last_ear = ear_value
        
        if ear_value < thresh_close:
            self.target_value = 1.0
        elif ear_value > thresh_open:
            self.target_value = 0.0
        
        diff = self.target_value - self.current_value
        self.current_value += diff * self.speed
        self.current_value = max(0.0, min(1.0, self.current_value))
        return self.current_value

# --- CUSTOM DRAWING SPECS ---
# Warna Cyan untuk Face
face_landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
face_connection_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1)

# Warna Magenta untuk Pose
pose_landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=4)
pose_connection_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2)

# Warna Kuning untuk Hand
hand_landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=4)
hand_connection_style = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2)

# --- INIT ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True, model_complexity=1)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Tuning Wajah
EYE_Y_OFFSET = 0.02
GAZE_SENSITIVITY = 2.0   
PITCH_CORRECTION_FACTOR = 0.015
DEADZONE = 0.3      
NECK_RATIO = 0.5
EAR_THRESH_CLOSE, EAR_THRESH_OPEN = 0.15, 0.25
MOUTH_OPEN_MIN, MOUTH_OPEN_MAX = 5.0, 40.0 

# Stabilizers
stab_pitch = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_yaw   = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_roll  = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_eye_x = Stabilizer(cov_process=0.005, cov_measure=0.1)
stab_eye_y = Stabilizer(cov_process=0.005, cov_measure=0.1)
stab_spine_roll  = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_spine_yaw   = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_spine_pitch = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_fingers_L = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]
stab_fingers_R = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]

# Smooth Blink
smooth_blink_L = SmoothBlink(speed=0.15)
smooth_blink_R = SmoothBlink(speed=0.15)

model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
L_IRIS_C, L_IN, L_OUT = 468, 133, 33  
R_IRIS_C, R_IN, R_OUT = 473, 362, 263 
last_raw_pitch, last_raw_yaw, last_raw_roll = 0, 0, 0
prev_time = 0

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)] 
BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]

# --- MODE TRACKING ---
TRACKING_MODE = "SETENGAH"  # "SETENGAH" atau "SELURUH"

cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

print("=" * 60)
print("=== VTuber Tracking - Setengah & Seluruh Badan ===")
print("=" * 60)
print("Kontrol:")
print("  [H] - Mode Setengah Badan (Half Body)")
print("  [F] - Mode Seluruh Badan (Full Body)")
print("  [Q] - Keluar")
print("=" * 60)
print(f"Mode Awal: {TRACKING_MODE} BADAN")
print("=" * 60)

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    
    image.flags.writeable = False
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True

    # ========================================
    # 1. FACE TRACKING (WARNA CYAN)
    # ========================================
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 face_landmark_style, face_connection_style)
        fl = results.face_landmarks
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        curr_pitch, curr_yaw, curr_roll = angles[0], angles[1], math.degrees(math.atan2(dY, dX))

        if abs(curr_pitch - last_raw_pitch) < DEADZONE: curr_pitch = last_raw_pitch
        else: last_raw_pitch = curr_pitch
        if abs(curr_yaw - last_raw_yaw) < DEADZONE: curr_yaw = last_raw_yaw
        else: last_raw_yaw = curr_yaw
        if abs(curr_roll - last_raw_roll) < DEADZONE: curr_roll = last_raw_roll
        else: last_raw_roll = curr_roll

        s_pitch = stab_pitch.update(curr_pitch)
        s_yaw, s_roll = stab_yaw.update(curr_yaw), stab_roll.update(curr_roll)

        neck_pitch, neck_yaw, neck_roll = s_pitch * NECK_RATIO, s_yaw * NECK_RATIO, s_roll * NECK_RATIO
        head_pitch, head_yaw, head_roll = s_pitch - neck_pitch, s_yaw - neck_yaw, s_roll - neck_roll
        
        raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
        raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)
        
        blink_l_state = smooth_blink_L.update(raw_ear_l, EAR_THRESH_CLOSE, EAR_THRESH_OPEN)
        blink_r_state = smooth_blink_R.update(raw_ear_r, EAR_THRESH_CLOSE, EAR_THRESH_OPEN)
        
        if s_yaw > 20.0: blink_r_state = blink_l_state 
        elif s_yaw < -20.0: blink_l_state = blink_r_state

        lx, ly = get_relative_iris(fl, L_IRIS_C, L_IN, L_OUT, img_w, img_h)
        rx, ry = get_relative_iris(fl, R_IRIS_C, R_IN, R_OUT, img_w, img_h)
        avg_x, avg_y = (lx + rx)/2.0, ((ly + ry)/2.0) - (s_pitch * PITCH_CORRECTION_FACTOR) + EYE_Y_OFFSET
        if not (blink_l_state > 0.5 or blink_r_state > 0.5):
            smooth_eye_x, smooth_eye_y = stab_eye_x.update(avg_x), stab_eye_y.update(avg_y)
        else:
            smooth_eye_x, smooth_eye_y = stab_eye_x.state[0][0], stab_eye_y.state[0][0]
        
        mouth_dist = np.linalg.norm(np.array([fl.landmark[13].x*img_w, fl.landmark[13].y*img_h]) - np.array([fl.landmark[14].x*img_w, fl.landmark[14].y*img_h]))
        mouth_open = max(0.0, min(1.0, (mouth_dist - 5.0) * (1.0/(35.0))))

        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(neck_pitch), math.radians(neck_yaw), math.radians(neck_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nqx), float(nqy), float(nqz), float(nqw)])
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(head_pitch), math.radians(head_yaw), math.radians(head_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hqx), float(hqy), float(hqz), float(hqw)])
        client.send_message("/VMC/Ext/Root/Pos", ["Root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])
        eqx, eqy, eqz, eqw = euler_to_quaternion(math.radians(smooth_eye_y*70.0), math.radians(smooth_eye_x*70.0), 0)
        client.send_message("/VMC/Ext/Bone/Pos", ["LeftEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["RightEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])

    # ========================================
    # 2. BODY & ARM TRACKING (WARNA MAGENTA)
    # ========================================
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 pose_landmark_style, pose_connection_style)
        lm = results.pose_landmarks.landmark

        def get_vec(idx): return [lm[idx].x, lm[idx].y, lm[idx].z]
        def to_unity_vec(mp_vec): 
            return np.array([ mp_vec[0] * INVERT_X * ARM_GAIN_XY, mp_vec[1] * INVERT_Y * ARM_GAIN_XY, mp_vec[2] * INVERT_Z * ARM_GAIN_Z ])

        # --- SPINE & CHEST ---
        l_sh, r_sh = get_vec(11), get_vec(12)
        l_hip, r_hip = get_vec(23), get_vec(24)

        shoulder_roll = (l_sh[1] - r_sh[1]) * 80.0 * BODY_LEAN_SENSITIVITY
        shoulder_yaw = (l_sh[2] - r_sh[2]) * 80.0 * BODY_TWIST_SENSITIVITY
        mid_sh_z = (l_sh[2] + r_sh[2]) / 2.0
        mid_hip_z = (l_hip[2] + r_hip[2]) / 2.0
        body_pitch = (mid_sh_z - mid_hip_z) * 100.0 * BODY_BEND_SENSITIVITY * INVERT_BODY_PITCH

        b_roll = stab_spine_roll.update(shoulder_roll)
        b_yaw  = stab_spine_yaw.update(shoulder_yaw)
        b_pitch = stab_spine_pitch.update(body_pitch)

        spine_q = euler_to_quaternion(math.radians(b_pitch * SPINE_RATIO), math.radians(b_yaw * SPINE_RATIO), math.radians(b_roll * SPINE_RATIO))
        client.send_message("/VMC/Ext/Bone/Pos", ["Spine", 0.0, 0.0, 0.0, float(spine_q[0]), float(spine_q[1]), float(spine_q[2]), float(spine_q[3])])
        chest_q = euler_to_quaternion(math.radians(b_pitch * CHEST_RATIO), math.radians(b_yaw * CHEST_RATIO), math.radians(b_roll * CHEST_RATIO))
        client.send_message("/VMC/Ext/Bone/Pos", ["Chest", 0.0, 0.0, 0.0, float(chest_q[0]), float(chest_q[1]), float(chest_q[2]), float(chest_q[3])])

        # --- ARMS ---
        if lm[11].visibility > 0.5 and lm[13].visibility > 0.5:
            start, end = to_unity_vec(get_vec(11)), to_unity_vec(get_vec(13))
            q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
            if lm[15].visibility > 0.5:
                start, end = to_unity_vec(get_vec(13)), to_unity_vec(get_vec(15))
                q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])

        if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
            start, end = to_unity_vec(get_vec(12)), to_unity_vec(get_vec(14))
            q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
            if lm[16].visibility > 0.5:
                start, end = to_unity_vec(get_vec(14)), to_unity_vec(get_vec(16))
                q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])

        # --- LEGS (HANYA JIKA MODE SELURUH BADAN) ---
        if TRACKING_MODE == "SELURUH":
            # LEFT LEG
            if lm[23].visibility > 0.5 and lm[25].visibility > 0.5:
                start, end = to_unity_vec(get_vec(23)), to_unity_vec(get_vec(25))
                q_lu = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperLeg", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
                if lm[27].visibility > 0.5:
                    start, end = to_unity_vec(get_vec(25)), to_unity_vec(get_vec(27))
                    q_ll = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                    client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerLeg", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])
                    if lm[31].visibility > 0.5:
                        start, end = to_unity_vec(get_vec(27)), to_unity_vec(get_vec(31))
                        q_f = get_limb_rotation(start, end, [0.0, 0.0, 1.0])
                        client.send_message("/VMC/Ext/Bone/Pos", ["LeftFoot", 0.0, 0.0, 0.0, float(q_f[0]), float(q_f[1]), float(q_f[2]), float(q_f[3])])

            # RIGHT LEG
            if lm[24].visibility > 0.5 and lm[26].visibility > 0.5:
                start, end = to_unity_vec(get_vec(24)), to_unity_vec(get_vec(26))
                q_ru = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperLeg", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
                if lm[28].visibility > 0.5:
                    start, end = to_unity_vec(get_vec(26)), to_unity_vec(get_vec(28))
                    q_rl = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                    client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerLeg", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])
                    if lm[32].visibility > 0.5:
                        start, end = to_unity_vec(get_vec(28)), to_unity_vec(get_vec(32))
                        q_f = get_limb_rotation(start, end, [0.0, 0.0, 1.0])
                        client.send_message("/VMC/Ext/Bone/Pos", ["RightFoot", 0.0, 0.0, 0.0, float(q_f[0]), float(q_f[1]), float(q_f[2]), float(q_f[3])])

    # ========================================
    # 3. HAND TRACKING (WARNA KUNING)
    # ========================================
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 hand_landmark_style, hand_connection_style)
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0)
            curl = stab_fingers_L[i].update(raw_curl)
            if name == "Thumb":
                angle = curl * (math.pi / 2.0) * THUMB_SIGN_L
                fqx, fqy, fqz, fqw = get_finger_quat(angle, THUMB_AXIS_L)
            else:
                angle = curl * (math.pi / 1.5) * FINGER_SIGN_L
                fqx, fqy, fqz, fqw = get_finger_quat(angle, FINGER_AXIS_L)
            for suffix in BONE_SUFFIXES: 
                client.send_message("/VMC/Ext/Bone/Pos", [f"Left{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 hand_landmark_style, hand_connection_style)
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0)
            curl = stab_fingers_R[i].update(raw_curl)
            if name == "Thumb":
                angle = curl * (math.pi / 2.0) * THUMB_SIGN_R
                fqx, fqy, fqz, fqw = get_finger_quat(angle, THUMB_AXIS_R)
            else:
                angle = curl * (math.pi / 1.5) * FINGER_SIGN_R
                fqx, fqy, fqz, fqw = get_finger_quat(angle, FINGER_AXIS_R)
            for suffix in BONE_SUFFIXES: 
                client.send_message("/VMC/Ext/Bone/Pos", [f"Right{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    # ========================================
    # 4. UI DISPLAY & MODE INDICATOR
    # ========================================
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    mode_color = (0, 255, 255) if TRACKING_MODE == "SETENGAH" else (255, 100, 255)
    mode_text = f"Mode: {TRACKING_MODE} BADAN"
    cv2.putText(image, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    cv2.putText(image, "[H]Setengah [F]Seluruh [Q]Keluar", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    status_x = 10
    status_y = img_h - 30
    if results.face_landmarks:
        cv2.putText(image, "[WAJAH]", (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        status_x += 90
    if results.pose_landmarks:
        cv2.putText(image, "[BADAN]", (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        status_x += 90
    if results.left_hand_landmarks:
        cv2.putText(image, "[TANGAN-L]", (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        status_x += 110
    if results.right_hand_landmarks:
        cv2.putText(image, "[TANGAN-R]", (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    if TRACKING_MODE == "SELURUH":
        leg_status_y = img_h - 60
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if lm[23].visibility > 0.5 or lm[24].visibility > 0.5:
                cv2.putText(image, "[KAKI AKTIF]", (10, leg_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    cv2.imshow('VTuber Tracking - Setengah & Seluruh Badan', image)
    
    # ========================================
    # 5. KEYBOARD INPUT
    # ========================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("\n[INFO] Keluar dari program...")
        break
    elif key == ord('h') or key == ord('H'):
        if TRACKING_MODE != "SETENGAH":
            TRACKING_MODE = "SETENGAH"
            print("\n[MODE] Beralih ke: SETENGAH BADAN")
            print("       (Kepala, Lengan, Tangan)")
    elif key == ord('f') or key == ord('F'):
        if TRACKING_MODE != "SELURUH":
            TRACKING_MODE = "SELURUH"
            print("\n[MODE] Beralih ke: SELURUH BADAN")
            print("       (Kepala, Lengan, Tangan, Kaki)")

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 60)
print("=== Tracking Dihentikan ===")
print("Terima kasih telah menggunakan VTuber Tracker!")
print("=" * 60)
print("\nDibuat oleh:")
print("Nama  : Gilang Gallan Indrana")
print("NRP   : 5024231030")
print("=" * 60)