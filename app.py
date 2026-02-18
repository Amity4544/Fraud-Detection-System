"""
Flask API for Fraud Detection Real-time prediction system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
encoder = None
scaler = None


def load_model_and_transformers():
    """Load trained model and transformers"""
    global model, encoder, scaler
    
    try:
        model = joblib.load('models/best_model.pkl')
        encoder = joblib.load('models/encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logger.info("✅ Model and transformers loaded")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


def preprocess_transaction(transaction):
    # Parse dates
    trans_time = pd.to_datetime(transaction['trans_date_trans_time'])
    dob = pd.to_datetime(transaction['dob'])
    
    # Create time features
    hour = trans_time.hour
    day_of_week = trans_time.dayofweek
    is_night = 1 if hour <= 4 or hour >= 22 else 0
    age = (trans_time - dob).days // 365
    
    # Calculate distance using Haversine formula
    R = 6371.0
    lat1_rad = np.radians(transaction['lat'])
    lon1_rad = np.radians(transaction['long'])
    lat2_rad = np.radians(transaction['merch_lat'])
    lon2_rad = np.radians(transaction['merch_long'])
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_to_merchant = R * c
    
    # Create feature dataframe (same order as training!)
    features = pd.DataFrame({
        'merchant': [transaction['merchant']],
        'category': [transaction['category']],
        'amt': [transaction['amt']],
        'gender': [transaction['gender']],
        'city_pop': [transaction['city_pop']],
        'job': [transaction['job']],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_night': [is_night],
        'age': [age],
        'dist_to_merchant': [dist_to_merchant]
    })
    
    # Encode categorical variables
    cat_cols = ['merchant', 'category', 'gender', 'job']
    features[cat_cols] = encoder.transform(features[cat_cols])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    return features_scaled


@app.route('/', methods=['GET'])
def home():
    """Serve the main HTML page"""
    from flask import render_template
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_info():
    """API information"""
    return jsonify({
        'service': 'Fraud Detection API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Fraud prediction'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get transaction data
        transaction = request.get_json()
        
        # Validate required fields
        required_fields = [
            'trans_date_trans_time', 'merchant', 'category', 'amt',
            'gender', 'city_pop', 'job', 'dob',
            'lat', 'long', 'merch_lat', 'merch_long'
        ]
        
        missing = [f for f in required_fields if f not in transaction]
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing
            }), 400
        
        # Preprocess
        features = preprocess_transaction(transaction)
        
        # Predict
        fraud_probability = float(model.predict_proba(features)[0, 1])
        is_fraud = bool(model.predict(features)[0])
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.6:
            risk_level = "MEDIUM"
        elif fraud_probability < 0.85:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Log prediction
        logger.info(f"Prediction: {'FRAUD' if is_fraud else 'SAFE'} (prob: {fraud_probability:.4f})")
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': {
                'is_fraud': is_fraud,
                'fraud_probability': fraud_probability,
                'risk_level': risk_level,
                'amount': transaction['amt'],
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  FRAUD DETECTION API")
    print("="*60)
    print("\nLoading model and transformers...")
    
    if load_model_and_transformers():
        print("\n✅ Model loaded successfully!")
        print("\n" + "="*60)
        print("  SERVER RUNNING")
        print("="*60)
        print("  URL: http://127.0.0.1:5000")
        print("\n  Endpoints:")
        print("    GET  /         - API info")
        print("    GET  /health   - Health check")
        print("    POST /predict  - Fraud prediction")
        print("="*60)
        print("\n🚀 Server starting...\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model!")
        print("   Please run 'python train_models.py' first")
