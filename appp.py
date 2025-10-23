from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import gdown
import base64

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
            gdown.download(url, model_path, quiet=False, fuzzy=True)
            print("✅ Model downloaded successfully!")
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"📦 Model size: {file_size / (1024*1024):.2f} MB")
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

def process_frame(image_data):
    """Process frame and return prediction"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, None
        
        data_aux = []
        x_ = []
        y_ = []
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3
        ) as hands:
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract coordinates
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                    
                    # Bounding box
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    
                    if model is not None:
                        try:
                            # Pad or trim to 84 features
                            if len(data_aux) < 84:
                                data_aux += [0] * (84 - len(data_aux))
                            elif len(data_aux) > 84:
                                data_aux = data_aux[:84]
                            
                            # Predict
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = str(prediction[0])
                            
                            # Draw rectangle and text
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.putText(
                                frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,
                                cv2.LINE_AA
                            )
                            
                            # Encode back to base64
                            _, buffer = cv2.imencode('.jpg', frame)
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            return f"data:image/jpeg;base64,{frame_base64}", predicted_character
                        except Exception as e:
                            print(f"Prediction error: {e}")
            
            # No hand detected - return original frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_base64}", None
            
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None, None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process():
    """Process a single frame"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        processed_image, prediction = process_frame(image_data)
        
        if processed_image:
            return jsonify({
                'success': True,
                'image': processed_image,
                'prediction': prediction
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Sign Language Detection Server...")
    port = int(os.environ.get('PORT', 5500))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
