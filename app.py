import os
import pickle
import requests
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)

MODEL_PATH = "model.p"
DRIVE_FILE_ID = "1A2hpFwgtkTlC6OY_AxkT837UoeiFVXey"  # your Google Drive file ID
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# -----------------------------
# DOWNLOAD MODEL IF NOT PRESENT
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    r = requests.get(DOWNLOAD_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model download complete!")

# -----------------------------
# LOAD MODEL
# -----------------------------
model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

# -----------------------------
# HAND TRACKING SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: '', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N',
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U',
    22: 'V', 23: 'X', 24: 'Y', 25: 'Z'
}

# -----------------------------
# CAMERA LOGIC (GENERATOR)
# -----------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Normalize/pad input
                if len(data_aux) < 84:
                    data_aux += [0] * (84 - len(data_aux))
                elif len(data_aux) > 84:
                    data_aux = data_aux[:84]

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = str(prediction[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                except Exception as e:
                    print("Prediction error:", e)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    # Get the port from the environment variable Render sets
    port = int(os.environ.get("PORT", 10000))
    # Run the app, listening on all available network interfaces
    app.run(host='0.0.0.0', port=port)
