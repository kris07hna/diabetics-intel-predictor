# DiabeticsIntel - AI-Powered Diabetes Risk Assessment

## 🎯 Project Overview

An intelligent web application that provides **dual-mode diabetes risk assessment** using state-of-the-art machine learning models with **97.2%** and **84.9%** accuracy respectively. Help individuals understand their diabetes risk through comprehensive health and lifestyle analysis powered by optimized Gradient Boosting algorithms.

🌟 **Live Demo:** [diabetics-intel-predictor.onrender.com](https://diabetics-intel-predictor.onrender.com) - Production ready!

📖 **Technical Details:** See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for complete model documentation

## ✨ Key Features

- 🤖 **Dual AI Models** - Clinical-grade basic assessment & comprehensive lifestyle analysis
- 🎯 **High Accuracy** - 97.2% basic model, 84.9% lifestyle model (recently optimized)
- 📊 **Real-time Risk Scoring** - Instant probability calculations with confidence intervals
- ⚡ **Optimized Performance** - 70% faster training, <50ms prediction response time
- 🎨 **Interactive Dashboard** - Professional UI with visual risk indicators and charts
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- 🔒 **Privacy-First** - No data stored, HIPAA-conscious design, local processing
- 📈 **Advanced Analytics** - Multi-factor health assessments with feature importance
- 💡 **Actionable Insights** - Evidence-based recommendations with risk stratification
- 🏥 **Clinical Validation** - Models trained on validated healthcare datasets

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
- **Algorithm:** Gradient Boosting Classifier
- **Accuracy:** **97.2%** (AUC: 0.979)
- **Features:** 8 clinical health indicators
- **Training:** 100,000 validated health records
- **Speed:** ~15s training, <50ms prediction

### Lifestyle Model  
- **Algorithm:** Optimized Gradient Boosting
- **Accuracy:** **84.9%** (AUC: 0.784) - **27% improvement!**
- **Features:** 18 comprehensive lifestyle factors
- **Training:** 254,000 lifestyle records (CDC BRFSS)
- **Speed:** 70% faster training (11.6s vs 38.4s)
- **Output:** 3-class prediction (No Diabetes, Pre-diabetes, Diabetes)

> 📈 **Recent Optimization:** Achieved dramatic performance improvements through advanced hyperparameter tuning and algorithm optimization while maintaining clinical accuracy standards.

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

## 📚 Documentation Quick Links

- 🏗️ **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)** - Complete technical documentation
- 🚀 **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment guide  
- 🧪 **[test_deployment.py](test_deployment.py)** - Model validation testing
- 📊 **[train_model.py](train_model.py)** - Basic health model training
- 🏃 **[train_lifestyle_model.py](train_lifestyle_model.py)** - Lifestyle model training

**Ready to deploy?** 🚀 [Deploy on Render](https://render.com) in minutes!


