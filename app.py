"""
Flask Web Application for Diabetes Prediction
Version: 2.0.0 - Adapted for diabetes risk assessment
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import sys

app = Flask(__name__)

# Load both models
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Basic health model paths
BASIC_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'diabetes_model.pkl')
BASIC_SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
BASIC_FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_names.pkl')
BASIC_METADATA_PATH = os.path.join(BASE_DIR, 'model', 'metadata.pkl')
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, 'model', 'label_encoders.pkl')

# Lifestyle model paths
LIFESTYLE_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'lifestyle_diabetes_model.pkl')
LIFESTYLE_SCALER_PATH = os.path.join(BASE_DIR, 'model', 'lifestyle_scaler.pkl')
LIFESTYLE_FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'lifestyle_feature_names.pkl')
LIFESTYLE_METADATA_PATH = os.path.join(BASE_DIR, 'model', 'lifestyle_metadata.pkl')

# Load basic health model
try:
    print(f"🔍 Loading basic health model from: {BASE_DIR}")
    basic_model = joblib.load(BASIC_MODEL_PATH)
    basic_scaler = joblib.load(BASIC_SCALER_PATH)
    basic_feature_names = joblib.load(BASIC_FEATURES_PATH)
    basic_metadata = joblib.load(BASIC_METADATA_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print(f"✅ Basic model loaded successfully!")
    print(f"   Model: {basic_metadata['model_name']}")
    print(f"   Accuracy: {basic_metadata['accuracy']:.4f}")
except Exception as e:
    print(f"❌ Error loading basic model: {e}")
    basic_model = None
    basic_scaler = None
    basic_feature_names = None
    basic_metadata = {'model_name': 'Basic Model Failed', 'accuracy': 0.0}
    label_encoders = None

# Load lifestyle model
try:
    print(f"🔍 Loading lifestyle model from: {BASE_DIR}")
    lifestyle_model = joblib.load(LIFESTYLE_MODEL_PATH)
    lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_PATH)
    lifestyle_feature_names = joblib.load(LIFESTYLE_FEATURES_PATH)
    lifestyle_metadata = joblib.load(LIFESTYLE_METADATA_PATH)
    print(f"✅ Lifestyle model loaded successfully!")
    print(f"   Model: {lifestyle_metadata['model_name']}")
    print(f"   Accuracy: {lifestyle_metadata['accuracy']:.4f}")
except Exception as e:
    print(f"❌ Error loading lifestyle model: {e}")
    lifestyle_model = None
    lifestyle_scaler = None
    lifestyle_feature_names = None
    lifestyle_metadata = {'model_name': 'Lifestyle Model Failed', 'accuracy': 0.0}

@app.route('/health')
def health():
    """
    Health check endpoint to verify model loading
    """
    return jsonify({
        'status': 'healthy' if basic_model is not None and lifestyle_model is not None else 'partial',
        'basic_model_loaded': basic_model is not None,
        'lifestyle_model_loaded': lifestyle_model is not None,
        'label_encoders_loaded': label_encoders is not None,
        'basic_metadata': basic_metadata,
        'lifestyle_metadata': lifestyle_metadata,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    })

@app.route('/')
def landing():
    """
    Landing page with model selection
    """
    return render_template('landing.html')

@app.route('/basic')
def basic_assessment():
    """
    Basic health assessment form
    """
    return render_template('index.html',
                         features=basic_feature_names,
                         accuracy=basic_metadata.get('accuracy', 0.0),
                         auc_score=basic_metadata.get('auc_score', 0.0),
                         model_name=basic_metadata['model_name'])

@app.route('/lifestyle')
def lifestyle_assessment():
    """
    Lifestyle assessment form
    """
    return render_template('lifestyle.html')

@app.route('/predict', methods=['POST'])
def predict_basic():
    """
    Make basic diabetes risk prediction
    """
    try:
        if basic_model is None or basic_scaler is None or label_encoders is None:
            return jsonify({
                'error': 'Basic model not loaded. Please train the model first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Prepare features with proper encoding
        features = []
        
        # Gender encoding
        gender = data.get('gender', 'Female')
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        features.append(gender_encoded)
        
        # Age
        age = float(data.get('age', 50))
        features.append(age)
        
        # Hypertension (0 or 1)
        hypertension = int(data.get('hypertension', 0))
        features.append(hypertension)
        
        # Heart disease (0 or 1)
        heart_disease = int(data.get('heart_disease', 0))
        features.append(heart_disease)
        
        # Smoking history encoding
        smoking_history = data.get('smoking_history', 'never')
        smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]
        features.append(smoking_encoded)
        
        # BMI
        bmi = float(data.get('bmi', 25))
        features.append(bmi)
        
        # HbA1c level
        hba1c = float(data.get('HbA1c_level', 5.5))
        features.append(hba1c)
        
        # Blood glucose level
        glucose = float(data.get('blood_glucose_level', 100))
        features.append(glucose)
        
        # Create DataFrame with feature names
        features_df = pd.DataFrame([features], columns=basic_feature_names)
        
        # Scale features
        features_scaled = basic_scaler.transform(features_df)
        
        # Predict
        predicted_class = basic_model.predict(features_scaled)[0]
        predicted_proba = basic_model.predict_proba(features_scaled)[0]
        
        # Get prediction probability for diabetes (class 1)
        diabetes_probability = predicted_proba[1]
        no_diabetes_probability = predicted_proba[0]
        
        # Risk assessment
        if diabetes_probability < 0.2:
            risk_level = "LOW"
            risk_color = "green"
            risk_message = "Very low diabetes risk. Continue healthy lifestyle."
        elif diabetes_probability < 0.4:
            risk_level = "MODERATE"
            risk_color = "orange"
            risk_message = "Moderate risk. Consider lifestyle improvements."
        elif diabetes_probability < 0.7:
            risk_level = "HIGH"
            risk_color = "red"
            risk_message = "High risk. Consult healthcare provider."
        else:
            risk_level = "VERY HIGH"
            risk_color = "darkred"
            risk_message = "Very high risk. Immediate medical consultation recommended."
        
        # Health assessments
        assessments = []
        
        # BMI assessment
        if bmi < 18.5:
            assessments.append({"metric": "BMI", "status": "Underweight", "color": "orange"})
        elif bmi <= 24.9:
            assessments.append({"metric": "BMI", "status": "Normal", "color": "green"})
        elif bmi <= 29.9:
            assessments.append({"metric": "BMI", "status": "Overweight", "color": "orange"})
        else:
            assessments.append({"metric": "BMI", "status": "Obese", "color": "red"})
        
        # HbA1c assessment
        if hba1c < 5.7:
            assessments.append({"metric": "HbA1c", "status": "Normal", "color": "green"})
        elif hba1c <= 6.4:
            assessments.append({"metric": "HbA1c", "status": "Pre-diabetes", "color": "orange"})
        else:
            assessments.append({"metric": "HbA1c", "status": "Diabetes range", "color": "red"})
        
        # Blood glucose assessment  
        if glucose < 100:
            assessments.append({"metric": "Glucose", "status": "Normal", "color": "green"})
        elif glucose <= 125:
            assessments.append({"metric": "Glucose", "status": "Pre-diabetes", "color": "orange"})
        else:
            assessments.append({"metric": "Glucose", "status": "Diabetes range", "color": "red"})
        
        # Risk factors
        risk_factors = []
        if age >= 45:
            risk_factors.append("Age ≥45 years")
        if hypertension:
            risk_factors.append("Hypertension")
        if heart_disease:
            risk_factors.append("Heart disease")
        if smoking_history in ['current', 'former', 'ever']:
            risk_factors.append("Smoking history")
        if bmi >= 25:
            risk_factors.append("Overweight/Obese")
        if hba1c >= 5.7:
            risk_factors.append("Elevated HbA1c")
        if glucose >= 100:
            risk_factors.append("Elevated glucose")
        
        return jsonify({
            'predicted_class': int(predicted_class),
            'diabetes_probability': float(round(diabetes_probability, 4)),
            'no_diabetes_probability': float(round(no_diabetes_probability, 4)),
            'risk_level': str(risk_level),
            'risk_color': str(risk_color),
            'risk_message': str(risk_message),
            'assessments': assessments,
            'risk_factors': risk_factors,
            'input_summary': {
                'gender': str(gender),
                'age': float(age),
                'bmi': float(round(bmi, 1)),
                'hba1c': float(round(hba1c, 1)),
                'glucose': float(glucose),
                'hypertension': bool(hypertension),
                'heart_disease': bool(heart_disease),
                'smoking_history': str(smoking_history)
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in prediction: {error_details}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/predict_lifestyle', methods=['POST'])
def predict_lifestyle():
    """
    Make lifestyle-based diabetes risk prediction
    """
    try:
        if lifestyle_model is None or lifestyle_scaler is None:
            return jsonify({
                'error': 'Lifestyle model not loaded. Please train the model first.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Prepare features in the correct order
        features = []
        for feature_name in lifestyle_feature_names:
            value = float(data.get(feature_name, 0))
            features.append(value)
        
        # Create DataFrame with feature names
        features_df = pd.DataFrame([features], columns=lifestyle_feature_names)
        
        # Scale features
        features_scaled = lifestyle_scaler.transform(features_df)
        
        # Predict
        predicted_class = lifestyle_model.predict(features_scaled)[0]
        predicted_proba = lifestyle_model.predict_proba(features_scaled)
        
        # Get class probabilities
        class_probabilities = predicted_proba[0].tolist()
        
        # Risk messages based on predicted class
        risk_messages = {
            0: "Low diabetes risk. Continue maintaining healthy lifestyle habits.",
            1: "Pre-diabetes detected. Lifestyle changes can prevent progression to diabetes.",
            2: "High diabetes risk. Consult healthcare provider for comprehensive evaluation."
        }
        
        return jsonify({
            'predicted_class': int(predicted_class),
            'class_probabilities': class_probabilities,
            'risk_message': risk_messages.get(int(predicted_class), 'Assessment complete.'),
            'class_names': ['No Diabetes', 'Pre-diabetes', 'Diabetes']
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in lifestyle prediction: {error_details}")
        return jsonify({
            'error': f'Lifestyle prediction error: {str(e)}'
        }), 500

@app.route('/risk_factors', methods=['GET'])
def risk_factors():
    """
    Get information about diabetes risk factors
    """
    return jsonify({
        'risk_factors': [
            {
                'name': 'Age',
                'description': 'Risk increases with age, especially after 45',
                'controllable': False
            },
            {
                'name': 'BMI',
                'description': 'Body Mass Index above 25 increases risk',
                'controllable': True
            },
            {
                'name': 'Physical Activity',
                'description': 'Regular exercise reduces diabetes risk',
                'controllable': True
            },
            {
                'name': 'Family History',
                'description': 'Genetics play a role in diabetes development',
                'controllable': False
            },
            {
                'name': 'Blood Pressure',
                'description': 'High blood pressure is linked to diabetes',
                'controllable': True
            },
            {
                'name': 'Smoking',
                'description': 'Smoking increases diabetes risk significantly',
                'controllable': True
            }
        ],
        'recommendations': [
            'Maintain a healthy weight',
            'Exercise regularly (150+ minutes per week)',
            'Eat a balanced diet low in processed foods',
            'Monitor blood sugar levels regularly',
            'Quit smoking if you smoke',
            'Manage stress levels',
            'Get regular health checkups'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

