import cv2
import numpy as np
import tensorflow as tf
import os
from flask import Flask, Response                                                                                                                                                                                                                                                       
from flask_socketio import SocketIO
from ultralytics import YOLO

# --- 1. إعدادات Flask و SocketIO ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  

# --- 2. تحميل الموديلات ---
# موديل الجريمة (CNN)
crime_model = tf.keras.models.load_model(r"D:\app\binarycnn200.h5", compile=False)
# موديل YOLO الخاص بك للأسلحة
yolo_custom = YOLO(r"D:\app\my_model.pt")

# --- 3. إعدادات الفيديوهات ---
# يمكنك إضافة مسارات الفيديوهات الثلاثة هنا
video_files = [
    r"D:\app\knives1.mp4",
    r"D:\app\123g.mp4",
    r"D:\app\guns1.mp4"
]

IMG_SIZE = 200
DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 300

def generate_multi_frames():
    # فتح قنوات الاتصال للفيديوهات
    caps = [cv2.VideoCapture(v) for v in video_files]
    names = [os.path.basename(v) for v in video_files]

    while True:
        combined_frames_list = []
        
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            # --- أ: كشف الأسلحة بموديل YOLO الخاص بك ---
            results = yolo_custom.predict(frame, conf=0.25, verbose=False)
            weapon_detected = False
            weapon_label = "No Weapon"

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]
                conf = float(box.conf[0])
                
                # رسم مربع السلاح (أحمر)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"{label} {conf*100:.0f}%", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                weapon_detected = True
                weapon_label = f"ALERT: {label}"

            # --- ب: تحليل الجريمة بموديل الـ CNN ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_for_model = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
            input_data = np.expand_dims(resized_for_model.astype('float32') / 255.0, axis=0)
            
            prediction = crime_model.predict(input_data, verbose=0)
            score = prediction[0][0]
            crime_status = "CRIME" if score > 0.5 else "SAFE"
            
            # تحديد اللون (أحمر لو جريمة أو سلاح)
            status_color = (0, 0, 255) if (score > 0.5 or weapon_detected) else (0, 255, 0)

            # --- ج: تجهيز الفريم للعرض المجمع ---
            frame_disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # شريط المعلومات العلوي الأسود
            cv2.rectangle(frame_disp, (0, 0), (DISPLAY_WIDTH, 70), (0, 0, 0), -1)
            cv2.putText(frame_disp, f"Vid: {names[i]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame_disp, f"Status: {crime_status}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame_disp, weapon_label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # برواز الفيديو
            cv2.rectangle(frame_disp, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), status_color, 4)
            
            combined_frames_list.append(frame_disp)

        # دمج الفيديوهات الثلاثة أفقياً
        final_combined = np.hstack(combined_frames_list)

        # تحويل النتيجة لـ Bytes لـ Flask
        ret, buffer = cv2.imencode('.jpg', final_combined)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 4. طرق (Routes) Flask ---

@app.route('/')
def index():
    return """
    <html>
        <head><title>AI Security Dashboard</title></head>
        <body style="background-color: #1a1a1a; color: white; text-align: center;">
            <h1>AI Crime & Weapon Detection System</h1>
            <hr>
            <img src="/video_feed" style="border: 5px solid #444; width: 90%;">
            <p>Status: Monitoring 3 Streams via YOLO & CNN Models</p>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_multi_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Security Server on http://127.0.0.1:5000")
    # تم تعطيل debug=True لتجنب مشاكل تشغيل الموديلات مرتين
    app.run(host='0.0.0.0', port=5000, debug=False)