# DiabeticsIntel - AI-Powered Diabetes Risk Assessment

## 🎯 Project Overview

An intelligent web application that provides **dual-mode diabetes risk assessment** using advanced machine learning models. Help individuals understand their diabetes risk through comprehensive health and lifestyle analysis.

🌟 **Live Demo:** [Deploy on Render](https://render.com) - Production ready!

## ✨ Key Features

- 🤖 **Dual ML Models** - Basic health assessment & comprehensive lifestyle analysis
- 📊 **Real-time Risk Scoring** - Instant diabetes risk probability calculations
- 🎨 **Interactive Dashboard** - User-friendly interface with visual risk indicators
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- 🔒 **Privacy-First** - No data stored, all processing done locally
- 📈 **Detailed Analytics** - BMI, HbA1c, glucose level assessments
- 💡 **Personalized Recommendations** - Actionable health advice based on results

## 🏗️ Technology Stack

- **Backend:** Python 3.12, Flask
- **ML Framework:** scikit-learn (Random Forest, Gradient Boosting)
- **Data Processing:** pandas, numpy, joblib
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
- **Deployment:** Render-optimized with gunicorn

## 🎯 Assessment Modes

### 1. Basic Health Assessment
- **Input Features:** Age, Gender, BMI, HbA1c, Blood Glucose, Hypertension, Heart Disease, Smoking History
- **Output:** Risk probability, health metrics evaluation, personalized recommendations
- **Use Case:** Quick screening for healthcare providers and individuals

### 2. Lifestyle Assessment  
- **Input Features:** Comprehensive lifestyle factors and behavioral patterns
- **Output:** Multi-class prediction (No Diabetes, Pre-diabetes, Diabetes)
- **Use Case:** Detailed lifestyle-based risk evaluation

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/kris07hna/diabetics-intel-predictor.git
cd diabetics-intel-predictor

# Install dependencies
pip install -r requirements.txt

# Train models (if needed)
python train_model.py
python train_lifestyle_model.py

# Run the application
python app.py
```

Visit: `http://localhost:5000`

### 🌐 Render Deployment

This application is **production-ready** for Render deployment:

1. **Fork/Clone** this repository
2. **Connect** your GitHub repository to Render
3. **Deploy** with these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3.12
4. **Access** your live application!

## 📊 Model Performance

### Basic Health Model
- **Algorithm:** Random Forest Classifier
- **Features:** 8 key health indicators
- **Performance:** High accuracy with clinical validation
- **Validation:** Cross-validated on diabetes prediction dataset

### Lifestyle Model  
- **Algorithm:** Multi-class classification
- **Features:** Comprehensive lifestyle factors
- **Output Classes:** No Diabetes, Pre-diabetes, Diabetes
- **Performance:** Robust lifestyle-based risk assessment

## 🎨 User Interface

### Landing Page
- Clean, professional design
- Model selection interface
- Health risk education content

### Assessment Forms
- **Basic Assessment:** Clinical health parameters
- **Lifestyle Assessment:** Behavioral and lifestyle factors
- **Real-time Validation:** Input validation and guidance

### Results Dashboard
- **Risk Level Indicators:** Visual risk scoring (Low/Moderate/High/Very High)
- **Health Metrics:** BMI, HbA1c, Glucose assessments
- **Risk Factors Analysis:** Identification of modifiable risk factors
- **Recommendations:** Personalized health improvement suggestions

## 🔍 How It Works

1. **User Selection:** Choose between Basic or Lifestyle assessment
2. **Data Input:** Enter health parameters through intuitive forms
3. **ML Processing:** Models analyze input using trained algorithms
4. **Risk Calculation:** Generate probability scores and risk classifications
5. **Results Display:** Present comprehensive results with actionable insights

## 🏥 Clinical Applications

- **Healthcare Screening:** Quick diabetes risk assessment for patients
- **Preventive Care:** Early identification of high-risk individuals
- **Health Education:** Risk factor awareness and education
- **Population Health:** Large-scale diabetes risk screening programs

## 🔒 Privacy & Security

- **No Data Storage:** All processing happens in real-time
- **Local Processing:** No personal data transmitted to external services
- **HIPAA Consideration:** Designed with healthcare privacy in mind
- **Secure Deployment:** Production-ready security configurations

## 📱 API Endpoints

- `GET /` - Landing page
- `GET /basic` - Basic health assessment form
- `GET /lifestyle` - Lifestyle assessment form  
- `POST /predict` - Basic model prediction
- `POST /predict_lifestyle` - Lifestyle model prediction
- `GET /health` - Application health check
- `GET /risk_factors` - Risk factors information

## 🚀 Render Deployment Guide

### Environment Variables (Optional)
```
FLASK_ENV=production
FLASK_DEBUG=False
```

### render.yaml Configuration
```yaml
services:
  - type: web
    name: diabetics-intel
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    plan: free
```

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation Steps
```bash
# Create virtual environment
python -m venv diabetes_env
source diabetes_env/bin/activate  # Linux/Mac
# diabetes_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify model files exist
python check_pickle.py

# Run development server
python app.py
```

## 📈 Future Enhancements

- [ ] Integration with wearable devices
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Mobile application
- [ ] API for third-party integration
- [ ] Enhanced lifestyle questionnaire
- [ ] Longitudinal risk tracking

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - Open source and free for educational and commercial use.

## 🙋‍♂️ Support

- **Documentation:** Check the `/docs` folder for detailed guides
- **Issues:** Report bugs via GitHub Issues
- **Healthcare Use:** Consult with medical professionals for clinical decisions

## 🎯 Disclaimer

This application is for **educational and screening purposes only**. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions. The predictions are based on statistical models and should not replace professional medical advice.

---

**Ready to deploy?** 🚀 [Deploy on Render](https://render.com) in minutes!


