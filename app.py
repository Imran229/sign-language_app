from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import os
import gdown

app = Flask(__name__)

# Download model from Google Drive
def download_model_from_gdrive():
    model_path = 'model.p'
    if not os.path.exists(model_path):
        file_id = "1A2hpFwgtkTlC6OY_AxkT837UoeiFVXey"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

model = None

def load_model():
    global model
    try:
        model_path = download_model_from_gdrive()
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        print('✅ Model loaded successfully')
    except Exception as e:
        print('❌ Model load error:', e)

load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'prediction': None})

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) < 84:
                    data_aux += [0] * (84 - len(data_aux))

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    return jsonify({'prediction': str(prediction[0])})
                except Exception as e:
                    print('Prediction error:', e)
                    return jsonify({'prediction': None})

        return jsonify({'prediction': None})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
