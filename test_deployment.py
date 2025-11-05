#!/usr/bin/env python3
"""
Test script to verify deployment readiness
Run this to check if all models load correctly
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np

def test_model_loading():
    """Test if all required model files load correctly"""
    print("🔍 Testing Model Loading...")
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Test basic model files
    basic_files = [
        'model/diabetes_model.pkl',
        'model/scaler.pkl', 
        'model/feature_names.pkl',
        'model/metadata.pkl',
        'model/label_encoders.pkl'
    ]
    
    lifestyle_files = [
        'model/lifestyle_diabetes_model.pkl',
        'model/lifestyle_scaler.pkl',
        'model/lifestyle_feature_names.pkl', 
        'model/lifestyle_metadata.pkl'
    ]
    
    all_files = basic_files + lifestyle_files
    
    print(f"\n📂 Checking {len(all_files)} model files...")
    
    missing_files = []
    for file_path in all_files:
        full_path = os.path.join(BASE_DIR, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files!")
        return False
    
    print(f"\n✅ All model files present!")
    return True

def test_model_functionality():
    """Test if models can actually make predictions"""
    print("\n🧪 Testing Model Functionality...")
    
    try:
        # Load basic model
        basic_model = joblib.load('model/diabetes_model.pkl')
        basic_scaler = joblib.load('model/scaler.pkl')
        basic_feature_names = joblib.load('model/feature_names.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        basic_metadata = joblib.load('model/metadata.pkl')
        
        print(f"   ✅ Basic model loaded: {basic_metadata['model_name']}")
        print(f"      Accuracy: {basic_metadata['accuracy']:.4f}")
        
        # Test basic prediction
        test_features = [1, 45, 0, 0, 2, 25.5, 6.2, 140]  # Sample test data
        features_df = pd.DataFrame([test_features], columns=basic_feature_names)
        features_scaled = basic_scaler.transform(features_df)
        prediction = basic_model.predict(features_scaled)
        probability = basic_model.predict_proba(features_scaled)
        
        print(f"      Test prediction: {prediction[0]} (probability: {probability[0][1]:.4f})")
        
    except Exception as e:
        print(f"   ❌ Basic model error: {e}")
        return False
    
    try:
        # Load lifestyle model
        lifestyle_model = joblib.load('model/lifestyle_diabetes_model.pkl')
        lifestyle_scaler = joblib.load('model/lifestyle_scaler.pkl')
        lifestyle_feature_names = joblib.load('model/lifestyle_feature_names.pkl')
        lifestyle_metadata = joblib.load('model/lifestyle_metadata.pkl')
        
        print(f"   ✅ Lifestyle model loaded: {lifestyle_metadata['model_name']}")
        print(f"      Accuracy: {lifestyle_metadata['accuracy']:.4f}")
        
        # Test lifestyle prediction
        test_features = [1, 1, 1, 25.0, 0, 0, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0, 1, 8]  # Sample test data
        features_df = pd.DataFrame([test_features], columns=lifestyle_feature_names)
        features_scaled = lifestyle_scaler.transform(features_df)
        prediction = lifestyle_model.predict(features_scaled)
        probability = lifestyle_model.predict_proba(features_scaled)
        
        print(f"      Test prediction: {prediction[0]} (probabilities: {probability[0]})")
        
    except Exception as e:
        print(f"   ❌ Lifestyle model error: {e}")
        return False
    
    return True

def test_data_files():
    """Test if data files are accessible"""
    print("\n📊 Testing Data Files...")
    
    data_files = [
        'data/diabetes_prediction_dataset.csv',
        'data/processed_data (2).csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file_path} - {len(df)} rows")
            except Exception as e:
                print(f"   ❌ {file_path} - Error reading: {e}")
                return False
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 70)
    print("🚀 DEPLOYMENT READINESS TEST")
    print("=" * 70)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("Model Files", test_model_loading),
        ("Model Functionality", test_model_functionality),  
        ("Data Files", test_data_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("✅ Your application should work correctly on Render")
    else:
        print("❌ SOME TESTS FAILED - DEPLOYMENT ISSUES LIKELY")
        print("⚠️  Fix the failed tests before deploying")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())