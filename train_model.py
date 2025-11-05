"""
Train Diabetes Prediction Models
Compare multiple algorithms and select the best
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
import os

print("=" * 70)
print("DIABETES PREDICTION - MODEL TRAINING")
print("=" * 70)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('data/diabetes_prediction_dataset.csv')
print(f"✅ Loaded {len(df)} samples")

# Preprocess categorical features
print("\n🔄 Preprocessing categorical features...")
label_encoders = {}

# Encode gender
le_gender = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['gender'])
label_encoders['gender'] = le_gender

# Encode smoking history
le_smoking = LabelEncoder()
df['smoking_history_encoded'] = le_smoking.fit_transform(df['smoking_history'])
label_encoders['smoking_history'] = le_smoking

# Prepare features and target
feature_columns = ['gender_encoded', 'age', 'hypertension', 'heart_disease', 
                   'smoking_history_encoded', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X = df[feature_columns]
y = df['diabetes']

print(f"\n📊 Features: {feature_columns}")
print(f"   Target: diabetes (0=No, 1=Yes)")
print(f"   Class distribution: {dict(y.value_counts())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔀 Train/Test Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Scale features
print("\n🔄 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models directory
os.makedirs('model', exist_ok=True)

# Train models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=None,  # No random state to avoid BitGenerator issues
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=None  # No random state to avoid BitGenerator issues
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
    
    # Metrics
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'model': model,
        'acc_train': acc_train,
        'acc_test': acc_test,
        'auc_test': auc_test,
        'cv_mean': cv_mean
    }
    
    print(f"   Accuracy (Train): {acc_train:.4f}")
    print(f"   Accuracy (Test):  {acc_test:.4f}")
    print(f"   AUC Score: {auc_test:.4f}")
    print(f"   CV Accuracy (5-fold): {cv_mean:.4f}")
    
    # Detailed classification report for best model preview
    if name == 'Random Forest':
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['No Diabetes', 'Diabetes']))

# Select best model
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (Test)': [r['acc_test'] for r in results.values()],
    'AUC Score': [r['auc_test'] for r in results.values()],
    'CV Accuracy': [r['cv_mean'] for r in results.values()]
})

print(f"\n{comparison.to_string(index=False)}")

best_model_name = max(results.keys(), key=lambda k: results[k]['auc_test'])
best_model = results[best_model_name]['model']
best_acc = results[best_model_name]['acc_test']
best_auc = results[best_model_name]['auc_test']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"   AUC Score: {best_auc:.4f}")

if best_auc >= 0.85:
    print(f"   ✅ TARGET ACHIEVED (AUC ≥ 0.85)!")
else:
    print(f"   ⚠️  Below target, but still good performance")

# Save best model - model was trained without random_state to avoid BitGenerator issues
print(f"\n💾 Saving model files...")
import pickle

# Save with standard pickle (no joblib) to avoid any serialization issues
with open('model/diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=3)

# Save other files normally
joblib.dump(scaler, 'model/scaler.pkl', compress=3, protocol=3)
joblib.dump(feature_columns, 'model/feature_names.pkl', protocol=3)
joblib.dump(label_encoders, 'model/label_encoders.pkl', protocol=3)

metadata = {
    'model_name': best_model_name,
    'accuracy': best_acc,
    'auc_score': best_auc,
    'samples': len(df),
    'features': feature_columns,
    'target_classes': ['No Diabetes', 'Diabetes'],
    'class_distribution': dict(y.value_counts())
}
joblib.dump(metadata, 'model/metadata.pkl', protocol=3)

print(f"   ✅ model/diabetes_model.pkl")
print(f"   ✅ model/scaler.pkl")
print(f"   ✅ model/feature_names.pkl")
print(f"   ✅ model/label_encoders.pkl")
print(f"   ✅ model/metadata.pkl")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Feature Importance:")
    feature_names_display = ['Gender', 'Age', 'Hypertension', 'Heart Disease', 
                            'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose Level']
    feature_importance = pd.DataFrame({
        'Feature': feature_names_display,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']:.<25} {row['Importance']:.4f}")

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 Ready to predict diabetes risk with {best_acc*100:.1f}% accuracy!")
