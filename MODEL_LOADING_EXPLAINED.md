# Model Loading Sequence Explained

## 1. Basic Health Model Loading
```python
# What happens in Render:
BASE_DIR = '/opt/render/project/src'  # Render's app directory

# File paths resolved to:
BASIC_MODEL_PATH = '/opt/render/project/src/model/diabetes_model.pkl'
BASIC_SCALER_PATH = '/opt/render/project/src/model/scaler.pkl'
# ... etc

# Models loaded into memory:
basic_model = joblib.load(BASIC_MODEL_PATH)      # ~15MB Gradient Boosting model
basic_scaler = joblib.load(BASIC_SCALER_PATH)    # ~2KB StandardScaler
feature_names = joblib.load(BASIC_FEATURES_PATH) # ~1KB feature list
metadata = joblib.load(BASIC_METADATA_PATH)      # ~2KB model info
label_encoders = joblib.load(LABEL_ENCODERS_PATH) # ~5KB encoders
```

## 2. Lifestyle Model Loading  
```python
# Similarly for lifestyle model:
lifestyle_model = joblib.load(LIFESTYLE_MODEL_PATH)      # ~25MB Logistic Regression
lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_PATH)    # ~3KB StandardScaler  
lifestyle_features = joblib.load(LIFESTYLE_FEATURES_PATH) # ~2KB feature list
lifestyle_metadata = joblib.load(LIFESTYLE_METADATA_PATH) # ~2KB model info
```

## 3. Memory Usage
- Total model files: ~42MB loaded into RAM
- Models stay in memory for entire application lifetime
- No re-loading needed for each prediction (fast responses)

## 4. Error Handling
If any model fails to load:
- Application continues running
- Health endpoint reports the issue  
- Failed models show as None
- User gets clear error message