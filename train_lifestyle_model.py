"""
Train Lifestyle-Based Diabetes Prediction Model
Features: Physical activity, diet, cholesterol, blood pressure, lifestyle factors
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os

print("=" * 70)
print("LIFESTYLE DIABETES PREDICTION - MODEL TRAINING")
print("=" * 70)

# Load data
print("\n📂 Loading lifestyle dataset...")
df = pd.read_csv('data/processed_data (2).csv')
print(f"✅ Loaded {len(df)} samples")

# Prepare features and target (removed Education, Income, NoDocbcCost)
feature_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                   'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth', 
                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

X = df[feature_columns]
y = df['Diabetes_012']

print(f"\n📊 Features: {len(feature_columns)} lifestyle and health factors")
print(f"   Target: Diabetes_012 (0=No Diabetes, 1=Pre-diabetes, 2=Diabetes)")
print(f"   Class distribution: {dict(y.value_counts().sort_index())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print("TRAINING LIFESTYLE MODELS")
print("=" * 70)

# Choose fast training mode for large datasets
FAST_MODE = True  # Set to False if you want full training

if FAST_MODE:
    print("🚀 Fast training mode enabled!")
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,    # Increased for better performance
            max_depth=25,        # Deeper trees for complex patterns
            min_samples_split=5, # More sensitive splitting
            min_samples_leaf=2,  # More detailed leaves
            max_features='sqrt', # Feature selection optimization
            bootstrap=True,
            oob_score=True,      # Out-of-bag validation
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,     # Extra Trees use all data
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=80,     # Further reduced for speed
            max_depth=4,         # Even shallower trees
            learning_rate=0.2,   # Higher learning rate for faster convergence
            subsample=0.6,       # Less data per tree for speed
            max_features='sqrt',
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            solver='lbfgs',
            C=1.0,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=80,     # Fast AdaBoost alternative
            learning_rate=0.8,
            random_state=42
        )
    }
else:
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=500,    # Much more trees for full training
            max_depth=30,        # Deeper trees
            min_samples_split=3, # More aggressive splitting
            min_samples_leaf=1,  # Maximum detail
            max_features='sqrt', # Feature selection
            bootstrap=True,
            oob_score=True,      # Validation scoring
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=35,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,    # Reduced from 300 for faster training
            max_depth=6,         # Reduced from 10 for speed
            learning_rate=0.1,   # Increased from 0.05 for faster convergence
            subsample=0.75,      # Slightly reduced for speed
            max_features='sqrt',
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            C=0.1,  # L2 regularization for better generalization
            random_state=42
        )
    }

results = {}
import time

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Track training time
    start_time = time.time()
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # For multiclass AUC, we need to handle it differently
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba_test = model.predict_proba(X_test_scaled)
            # Use macro average for multiclass AUC
            auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='macro')
        else:
            auc_test = 0.0
    except:
        auc_test = 0.0
    
    # Metrics
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'model': model,
        'acc_train': acc_train,
        'acc_test': acc_test,
        'auc_test': auc_test,
        'cv_mean': cv_mean,
        'training_time': training_time
    }
    
    print(f"   Accuracy (Train): {acc_train:.4f}")
    print(f"   Accuracy (Test):  {acc_test:.4f}")
    print(f"   AUC Score (Macro): {auc_test:.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   CV Accuracy (5-fold): {cv_mean:.4f}")
    
    # Detailed classification report for best model preview
    if name == 'Random Forest':
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                  target_names=['No Diabetes', 'Pre-diabetes', 'Diabetes']))

# Select best model
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (Test)': [r['acc_test'] for r in results.values()],
    'AUC Score': [r['auc_test'] for r in results.values()],
    'CV Accuracy': [r['cv_mean'] for r in results.values()],
    'Training Time (s)': [r['training_time'] for r in results.values()]
})

print(f"\n{comparison.to_string(index=False)}")

# Select best model using weighted score (accuracy + AUC)
best_model_name = max(results.keys(), key=lambda k: 0.6 * results[k]['acc_test'] + 0.4 * results[k]['auc_test'])
best_model = results[best_model_name]['model']
best_acc = results[best_model_name]['acc_test']
best_auc = results[best_model_name]['auc_test']
best_time = results[best_model_name]['training_time']

# Also find fastest model
fastest_model_name = min(results.keys(), key=lambda k: results[k]['training_time'])
fastest_time = results[fastest_model_name]['training_time']
fastest_acc = results[fastest_model_name]['acc_test']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"   AUC Score: {best_auc:.4f}")
print(f"   Training Time: {best_time:.2f}s")

print(f"\n⚡ FASTEST MODEL: {fastest_model_name}")
print(f"   Training Time: {fastest_time:.2f}s")
print(f"   Accuracy: {fastest_acc:.4f} ({fastest_acc*100:.2f}%)")

if best_acc >= 0.80:
    print(f"   ✅ EXCELLENT PERFORMANCE ACHIEVED!")
elif best_acc >= 0.75:
    print(f"   ✅ GOOD PERFORMANCE ACHIEVED!")
elif best_acc >= 0.70:
    print(f"   ⚠️  DECENT PERFORMANCE - Could be improved")
else:
    print(f"   ❌ LOW PERFORMANCE - Needs improvement")

# Save best model
print(f"\n💾 Saving lifestyle model files...")
import pickle

# Save with standard pickle
with open('model/lifestyle_diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=3)

# Save other files
joblib.dump(scaler, 'model/lifestyle_scaler.pkl', compress=3, protocol=3)
joblib.dump(feature_columns, 'model/lifestyle_feature_names.pkl', protocol=3)

lifestyle_metadata = {
    'model_name': best_model_name,
    'accuracy': best_acc,
    'auc_score': best_auc,
    'samples': len(df),
    'features': feature_columns,
    'target_classes': ['No Diabetes', 'Pre-diabetes', 'Diabetes'],
    'class_distribution': dict(y.value_counts().sort_index()),
    'model_type': 'lifestyle'
}
joblib.dump(lifestyle_metadata, 'model/lifestyle_metadata.pkl', protocol=3)

print(f"   ✅ model/lifestyle_diabetes_model.pkl")
print(f"   ✅ model/lifestyle_scaler.pkl")
print(f"   ✅ model/lifestyle_feature_names.pkl")
print(f"   ✅ model/lifestyle_metadata.pkl")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Lifestyle Feature Importance:")
    feature_names_display = [
        'High Blood Pressure', 'High Cholesterol', 'Cholesterol Check', 'BMI', 'Smoker',
        'Stroke History', 'Heart Disease', 'Physical Activity', 'Fruits Consumption',
        'Vegetables Consumption', 'Heavy Alcohol', 'Healthcare Access',
        'General Health', 'Mental Health Days', 'Physical Health Days', 'Difficulty Walking',
        'Sex', 'Age'
    ]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names_display,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['Feature']:.<30} {row['Importance']:.4f}")

print("\n" + "=" * 70)
print("✅ LIFESTYLE MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 Ready to predict diabetes risk from lifestyle factors with {best_acc*100:.1f}% accuracy!")