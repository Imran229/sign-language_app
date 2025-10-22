from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import os
import gdown

app = Flask(__name__)

# Google Drive model download function
def download_model_from_gdrive():
    """Download model from Google Drive if not present"""
    model_path = 'model.p'
    
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        
        # Google Drive File ID
        file_id = "1A2hpFwgtkTlC6OY_AxkT837UoeiFVXey"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            # Use gdown with fuzzy match for large files
            gdown.download(url, model_path, quiet=False, fuzzy=True)
            print("✅ Model downloaded successfully!")
            
            # Verify the downloaded file
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"📦 Model size: {file_size / (1024*1024):.2f} MB")
                
                # Check if it's a valid pickle file
                try:
                    with open(model_path, 'rb') as f:
                        first_bytes = f.read(10)
                        if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                            print("❌ Downloaded HTML instead of model file!")
                            os.remove(model_path)
                            return None
                except:
                    pass
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return None
    else:
        print("✅ Model already exists!")
    
    return model_path

# Load trained model at startup
model = None

def load_model_startup():
    """Load model from Google Drive"""
    global model
    try:
        model_path = download_model_from_gdrive()
        if model_path and os.path.exists(model_path):
            model_dict = pickle.load(open(model_path, 'rb'))
            model = model_dict['model']
            print("✅ Model loaded successfully!")
        else:
            model = None
            print("❌ Model not found!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Load model when module is imported
print("🚀 Initializing application...")
load_model_startup()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
is_running = False

def initialize_camera():
    """Initialize camera"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ Cannot open camera")
            return False
    return True

def detect_sign_language():
    """Detect sign language from camera feed"""
    global output_frame, camera, is_running
    
    if not initialize_camera():
        return
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    is_running = True
    
    while is_running:
        success, frame = camera.read()
        if not success:
            print("⚠️ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)
        
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
                
                if model is not None:
                    try:
                        if len(data_aux) < 84:
                            data_aux += [0] * (84 - len(data_aux))
                        elif len(data_aux) > 84:
                            data_aux = data_aux[:84]
                        
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = str(prediction[0])
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(
                            frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,
                            cv2.LINE_AA
                        )
                    except Exception as e:
                        print(f"Prediction error: {e}")
        
        cv2.putText(
            frame, "LIVE", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        with lock:
            output_frame = frame.copy()
    
    hands.close()

def generate():
    """Generate frames for streaming"""
    global output_frame
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    """Start detection in background thread"""
    global is_running
    from flask import jsonify
    if not is_running:
        thread = threading.Thread(target=detect_sign_language, daemon=True)
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop():
    """Stop detection"""
    global is_running, camera
    from flask import jsonify
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("🚀 Starting Sign Language Detection Server...")
    print("📹 Open http://127.0.0.1:5500 in your browser")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5500)), threaded=True)
