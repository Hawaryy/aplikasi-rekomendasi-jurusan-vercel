from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import traceback

print("Starting Flask App for Vercel", flush=True)

app = Flask(__name__)
CORS(app)

# ==== Lazy Loading Models ====
model = None
scaler = None
label_encoder = None

def load_models():
    """Load models only when needed"""
    global model, scaler, label_encoder
    
    if model is not None:
        return
    
    try:
        print("Loading models...", flush=True)
        
        # Get the directory where this file is located (api/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to root directory
        base_dir = os.path.dirname(current_dir)
        
        # Build full paths to model files
        model_path = os.path.join(base_dir, 'model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        label_path = os.path.join(base_dir, 'label_encoder.pkl')
        
        print(f"Base directory: {base_dir}", flush=True)
        print(f"Loading model from: {model_path}", flush=True)
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.pkl not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"scaler.pkl not found at {scaler_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"label_encoder.pkl not found at {label_path}")
        
        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_path)
        
        print("Models loaded successfully!", flush=True)
    except Exception as e:
        print(f"Error loading models: {e}", flush=True)
        traceback.print_exc()
        raise

# ==== Kolom Fitur ====
FEATURE_COLUMNS = [
    "Matematika", "Fisika", "Kimia", "Biologi", "Ekonomi", 
    "Sosiologi", "Agama Islam", "PPKN", "Sejarah", 
    "Seni Budaya", "Penjas", "B_Indonesia", "B_Inggris"
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API Rekomendasi Jurusan',
        'status': 'running',
        'platform': 'Vercel Serverless',
        'endpoints': {
            '/': 'Home',
            '/health': 'Health check',
            '/predict': 'POST - Predict jurusan'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Load models saat request pertama
        load_models()
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        missing = [f for f in FEATURE_COLUMNS if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        input_values = [data[f] for f in FEATURE_COLUMNS]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        hasil_encoded = model.predict(input_scaled)
        hasil_jurusan = label_encoder.inverse_transform(hasil_encoded)[0]

        print(f"Prediction: {hasil_jurusan}", flush=True)

        return jsonify({
            'rekomendasi_jurusan': hasil_jurusan,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error in predict: {e}", flush=True)
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'failed'}), 500