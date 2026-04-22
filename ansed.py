import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
import cv2
import numpy as np
import pickle

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from flask import Flask, Response
from ultralytics import YOLO

# =============================================
# تحميل الموديلات
# =============================================
print("[1/4] تحميل موديل CNN...")
crime_model = tf.keras.models.load_model(r"D:\app\binarycnn200.h5", compile=False)

print("[2/4] تحميل موديل YOLO...")
yolo_custom = YOLO(r"D:\app\my_model.pt")

print("[3/4] تحميل MediaPipe — 3 instances مستقلة...")
import mediapipe as mp
mp_pose = mp.solutions.pose

# ✅ FIX 1: 3 instances مستقلة، واحدة لكل فيديو
pose_instances = [
    mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    for _ in range(3)
]

print("[4/4] تحميل PKL classifier...")
sys.path.insert(0, r'D:\app')
with open(r"D:\app\action_classifier.pkl", 'rb') as f:
    action_clf_obj = pickle.load(f)

action_model = action_clf_obj.classifiers[6]
print("تم تحميل جميع الموديلات بنجاح!")

# =============================================
# إعدادات
# =============================================
ACTION_NAMES = {
    0: "Standing", 1: "Walking",  2: "Running",
    3: "Sitting",  4: "Falling",  5: "Fighting",
    6: "Punching",  7: "Kicking",  8: "Unknown"
}
VIOLENT_ACTIONS = {5, 6, 7}

IMG_SIZE      = 200
DISPLAY_WIDTH  = 400
DISPLAY_HEIGHT = 300

# ✅ FIX 2: Skip Frames — يحلل كل N إطار ويحتفظ بآخر نتيجة
ANALYZE_EVERY_N_FRAMES = 5

video_files = [
    r"D:\app\knives1.mp4",
    r"D:\app\V_19.mp4",
    r"D:\app\guns1.mp4"
]

# =============================================
# دالة استخراج الـ 50 feature من MediaPipe
# الآن تأخذ pose_instance كـ parameter
# =============================================
def extract_pose_features(frame, pose_instance):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_instance.process(rgb)      # ✅ كل فيديو له instance خاص به
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    features = []
    for i in range(25):
        features.append(landmarks[i].x)
        features.append(landmarks[i].y)
    return np.array(features).reshape(1, -1)

# =============================================
# Flask App
# =============================================
app = Flask(__name__)

def generate_multi_frames():
    caps  = [cv2.VideoCapture(v) for v in video_files]
    names = [os.path.basename(v) for v in video_files]

    # ✅ FIX 2: كاش لآخر نتيجة تحليل لكل فيديو
    last_results = [
        {
            "crime_status": "SAFE", "score": 0.0,
            "weapon_detected": False, "weapon_label": "No Weapon",
            "action_label": "No Pose", "action_color": (128, 128, 128),
            "action_is_violent": False, "is_alert": False
        }
        for _ in range(3)
    ]

    frame_counters = [0, 0, 0]   # عداد الإطارات لكل فيديو

    while True:
        combined_frames_list = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, f"No video: {names[i]}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                combined_frames_list.append(blank)
                continue

            frame_counters[i] += 1

            # ✅ FIX 2: تحليل فقط كل N إطار
            if frame_counters[i] % ANALYZE_EVERY_N_FRAMES == 0:

                # --- 1: YOLO ---
                weapon_detected = False
                weapon_label    = "No Weapon"
                try:
                    yolo_results = yolo_custom.predict(frame, conf=0.25, verbose=False)
                    for box in yolo_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label  = yolo_results[0].names[cls_id]
                        conf   = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"{label} {conf*100:.0f}%",
                                    (x1, max(y1-10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        weapon_detected = True
                        weapon_label    = f"ALERT: {label}"
                except:
                    weapon_label = "YOLO Error"

                # --- 2: CNN ---
                crime_status = "SAFE"
                score        = 0.0
                try:
                    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized    = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
                    input_data = np.expand_dims(resized.astype('float32') / 255.0, axis=0)
                    prediction = crime_model.predict(input_data, verbose=0)
                    score       = float(prediction[0][0])
                    crime_status = "CRIME" if score > 0.5 else "SAFE"
                except:
                    crime_status = "CNN Error"

                # --- 3: MediaPipe + PKL ---
                action_label      = "No Pose"
                action_color      = (128, 128, 128)
                action_is_violent = False
                try:
                    # ✅ FIX 1: بنمرر الـ instance الخاصة بالفيديو ده
                    pose_features = extract_pose_features(frame, pose_instances[i])
                    if pose_features is not None:
                        action_id         = int(action_model.predict(pose_features)[0])
                        action_label      = ACTION_NAMES.get(action_id, f"Action_{action_id}")
                        action_is_violent = action_id in VIOLENT_ACTIONS
                        action_color      = (0, 0, 255) if action_is_violent else (0, 255, 0)
                except:
                    action_label = "Pose Error"

                # ✅ حفظ النتيجة في الكاش
                last_results[i] = {
                    "crime_status":     crime_status,
                    "score":            score,
                    "weapon_detected":  weapon_detected,
                    "weapon_label":     weapon_label,
                    "action_label":     action_label,
                    "action_color":     action_color,
                    "action_is_violent": action_is_violent,
                    "is_alert":         (score > 0.5) or weapon_detected or action_is_violent
                }

            # ✅ استخدام آخر نتيجة محفوظة (سواء حللنا الإطار ده أو لأ)
            r = last_results[i]

            frame_disp   = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            status_color = (0, 0, 255) if r["is_alert"] else (0, 255, 0)

            cv2.rectangle(frame_disp, (0, 0), (DISPLAY_WIDTH, 90), (0, 0, 0), -1)
            cv2.putText(frame_disp, f"Vid: {names[i]}", (10, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame_disp, f"Crime: {r['crime_status']} ({r['score']:.2f})", (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            cv2.putText(frame_disp, f"Action: {r['action_label']}", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, r["action_color"], 2)
            cv2.putText(frame_disp, r["weapon_label"], (10, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.rectangle(frame_disp, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), status_color, 4)
            combined_frames_list.append(frame_disp)

        final_combined = np.hstack(combined_frames_list)
        ret2, buffer   = cv2.imencode('.jpg', final_combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Security Dashboard</title>
        <style>
            body { background: #1a1a1a; color: white; text-align: center;
                   font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1   { color: #ff4444; font-size: 22px; margin-bottom: 5px; }
            p    { color: #aaa; font-size: 13px; margin: 5px 0 15px; }
            img  { border: 3px solid #333; width: 95%; border-radius: 6px; }
            .badge { display: inline-block; background: #333; border-radius: 4px;
                     padding: 3px 10px; margin: 0 4px; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>AI Security Monitoring System</h1>
        <p>
            <span class="badge">CNN — Crime Detection</span>
            <span class="badge">YOLO — Weapon Detection</span>
            <span class="badge">PKL — Action Analysis</span>
        </p>
        <img src="/video_feed">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_multi_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    print("\nStarting server → http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)