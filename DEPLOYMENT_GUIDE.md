# 🚀 DiabeticsIntel Deployment Guide

Complete guide to deploy your AI-powered diabetes risk assessment application on various platforms.

## 📋 Table of Contents

1. [Render Deployment (Recommended)](#render-deployment-recommended)
2. [Railway Deployment](#railway-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [PythonAnywhere Deployment](#pythonanywhere-deployment)
5. [Local Development Setup](#local-development-setup)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

---

## 🌟 Render Deployment (Recommended)

### Why Render?
- ✅ **Free Tier Available** - Perfect for demos and testing
- ✅ **Automatic HTTPS** - SSL certificates included
- ✅ **Git Integration** - Auto-deploy on push
- ✅ **Easy Setup** - One-click deployment
- ✅ **Python Support** - Native Flask/gunicorn support

### Step-by-Step Render Deployment

#### 1. Prepare Your Repository
Your GitHub repository is already configured: `https://github.com/kris07hna/diabetics-intel-predictor`

#### 2. Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Connect your GitHub account

#### 3. Deploy Web Service
1. **Click "New +"** → **"Web Service"**
2. **Connect Repository:** `kris07hna/diabetics-intel-predictor`
3. **Configure Settings:**

```yaml
Name: diabetics-intel-predictor
Environment: Python 3
Region: Oregon (US West) or closest to your users
Branch: main
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
```

#### 4. Advanced Settings (Optional)
```yaml
Auto-Deploy: Yes
Environment Variables:
  FLASK_ENV: production
  FLASK_DEBUG: False
  PYTHONUNBUFFERED: 1
```

#### 5. Deploy
1. Click **"Create Web Service"**
2. Wait 3-5 minutes for deployment
3. Your app will be live at: `https://your-app-name.onrender.com`

### 🔧 Render Configuration Files

Create `render.yaml` in your project root (optional):

```yaml
services:
  - type: web
    name: diabetics-intel
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: FLASK_ENV
        value: production
      - key: PYTHONUNBUFFERED
        value: 1
```

---

## 🚄 Railway Deployment

### Step-by-Step Railway Setup

#### 1. Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Install Railway CLI (optional)

#### 2. Deploy from GitHub
1. **New Project** → **Deploy from GitHub repo**
2. **Select:** `kris07hna/diabetics-intel-predictor`
3. **Railway Auto-detects:** Python application

#### 3. Configuration
Railway automatically detects your app, but you can customize:

```yaml
# railway.json (optional)
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app:app --bind 0.0.0.0:$PORT",
    "healthcheckPath": "/health"
  }
}
```

#### 4. Environment Variables
In Railway dashboard:
- `FLASK_ENV=production`
- `PYTHONUNBUFFERED=1`

---

## 🟣 Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Git configured

### Step-by-Step Heroku Setup

#### 1. Install Heroku CLI
```bash
# Windows (PowerShell)
winget install Heroku.CLI

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

#### 2. Create Heroku App
```bash
# Login to Heroku
heroku login

# Create app
heroku create diabetics-intel-app

# Add Python buildpack
heroku buildpacks:set heroku/python
```

#### 3. Create Procfile
Create `Procfile` in project root:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

#### 4. Configure Environment
```bash
heroku config:set FLASK_ENV=production
heroku config:set PYTHONUNBUFFERED=1
```

#### 5. Deploy
```bash
git push heroku main
```

#### 6. Open Application
```bash
heroku open
```

---

## 🐍 PythonAnywhere Deployment

### Step-by-Step PythonAnywhere Setup

#### 1. Create Account
- Go to [pythonanywhere.com](https://pythonanywhere.com)
- Sign up for free account

#### 2. Upload Code
```bash
# In PythonAnywhere console
git clone https://github.com/kris07hna/diabetics-intel-predictor.git
cd diabetics-intel-predictor
```

#### 3. Install Dependencies
```bash
pip3.10 install --user -r requirements.txt
```

#### 4. Configure Web App
1. **Web tab** → **Add a new web app**
2. **Python 3.10** → **Manual configuration**
3. **Source code:** `/home/yourusername/diabetics-intel-predictor`
4. **WSGI configuration file:** Edit to point to your app

#### 5. WSGI Configuration
```python
import sys
import os

# Add your project directory to sys.path
path = '/home/yourusername/diabetics-intel-predictor'
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application
```

#### 6. Static Files
- **URL:** `/static/`
- **Directory:** `/home/yourusername/diabetics-intel-predictor/static/`

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.8+ installed
- Git installed
- Virtual environment (recommended)

### Setup Steps

#### 1. Clone Repository
```bash
git clone https://github.com/kris07hna/diabetics-intel-predictor.git
cd diabetics-intel-predictor
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv diabetes_env
diabetes_env\Scripts\activate

# Linux/Mac  
python3 -m venv diabetes_env
source diabetes_env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Verify Model Files
```bash
python check_pickle.py
```

#### 5. Run Application
```bash
# Development mode
python app.py

# Production mode
gunicorn app:app --bind 0.0.0.0:5000
```

#### 6. Access Application
Open browser: `http://localhost:5000`

---

## 🔧 Environment Variables

### Required Variables
```bash
FLASK_ENV=production          # Set to production for deployment
FLASK_DEBUG=False            # Disable debug mode in production
PYTHONUNBUFFERED=1          # Ensure logs are visible
```

### Optional Variables
```bash
PORT=5000                   # Port number (auto-set by most platforms)
GUNICORN_WORKERS=4          # Number of worker processes
GUNICORN_TIMEOUT=30         # Request timeout
```

### Platform-Specific Setup

#### Render
Set in dashboard under "Environment" tab

#### Railway  
Set in dashboard under "Variables" tab

#### Heroku
```bash
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=False
```

---

## 🔍 Health Check Endpoints

Your application includes built-in health checks:

### `/health` Endpoint
```json
{
  "status": "healthy",
  "basic_model_loaded": true,
  "lifestyle_model_loaded": true,
  "label_encoders_loaded": true,
  "basic_metadata": {...},
  "lifestyle_metadata": {...},
  "python_version": "3.10.0"
}
```

Use this endpoint for:
- **Render:** Automatic health checks
- **Railway:** Health check monitoring  
- **Heroku:** Dyno health verification
- **Load balancers:** Uptime monitoring

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **Build Fails - Dependencies**
```bash
# Error: Package not found
# Solution: Update requirements.txt
pip freeze > requirements.txt
```

#### 2. **Model Files Not Loading**
```bash
# Error: FileNotFoundError for model files
# Solution: Ensure model files are in repository
ls model/
# Should show: diabetes_model.pkl, scaler.pkl, etc.
```

#### 3. **Port Issues**
```python
# Error: Port binding failed
# Solution: Use environment PORT variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

#### 4. **Memory Issues**
```bash
# Error: Memory limit exceeded
# Solution: Optimize model loading or upgrade plan
# For free tiers, ensure models are < 100MB total
```

#### 5. **Static Files Not Loading**
```python
# Error: 404 for CSS/JS files
# Solution: Check Flask static configuration
app = Flask(__name__, static_folder='static')
```

#### 6. **CORS Issues**
```python
# Error: CORS policy blocks requests
# Solution: Add CORS headers (if needed)
from flask_cors import CORS
CORS(app)
```

### Debug Commands

#### Check Application Logs
```bash
# Render: View in dashboard
# Railway: railway logs
# Heroku: heroku logs --tail
# Local: Check terminal output
```

#### Test Health Endpoint
```bash
curl https://your-app-url.com/health
```

#### Verify Model Loading
```bash
python -c "import joblib; print('Models OK')"
```

---

## 📊 Performance Optimization

### Production Recommendations

#### 1. **Gunicorn Configuration**
```bash
# Optimal settings for production
gunicorn app:app \
  --workers=4 \
  --worker-class=sync \
  --timeout=30 \
  --keep-alive=2 \
  --max-requests=1000 \
  --max-requests-jitter=100 \
  --bind 0.0.0.0:$PORT
```

#### 2. **Model Loading Optimization**
```python
# Load models once at startup, not per request
# Already implemented in app.py
```

#### 3. **Static File Optimization**
- Use CDN for static assets (if applicable)
- Minimize CSS/JS files
- Compress images

#### 4. **Caching Strategy**
```python
# Add response caching for static endpoints
from flask import make_response

@app.route('/risk_factors')
def risk_factors():
    response = make_response(jsonify({...}))
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response
```

---

## 🔐 Security Considerations

### Production Security Checklist

- ✅ **HTTPS Enabled** (automatic on Render/Railway/Heroku)
- ✅ **Debug Mode Disabled** (`FLASK_DEBUG=False`)
- ✅ **Secret Key Set** (if using sessions)
- ✅ **Input Validation** (implemented in forms)
- ✅ **No Sensitive Data in Logs**
- ✅ **CORS Configured** (if needed)
- ✅ **Rate Limiting** (consider for production)

### Optional Security Headers
```python
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

---

## 🎯 Post-Deployment Checklist

After successful deployment:

- [ ] **Test Health Endpoint:** `GET /health`
- [ ] **Test Basic Assessment:** `GET /basic`
- [ ] **Test Lifestyle Assessment:** `GET /lifestyle`
- [ ] **Test Prediction API:** `POST /predict`
- [ ] **Verify Model Loading:** Check health endpoint response
- [ ] **Test Mobile Responsiveness**
- [ ] **Check Application Logs** for errors
- [ ] **Set Up Monitoring** (optional)
- [ ] **Configure Custom Domain** (optional)

---

## 📞 Support

If you encounter issues:

1. **Check Application Logs** first
2. **Verify Model Files** are present and valid
3. **Test Health Endpoint** for model status
4. **Review Environment Variables**
5. **Check Platform-Specific Documentation**

### Quick Links
- **Render Docs:** https://render.com/docs
- **Railway Docs:** https://docs.railway.app
- **Heroku Docs:** https://devcenter.heroku.com
- **Flask Deployment:** https://flask.palletsprojects.com/en/2.3.x/deploying/

---

**🚀 Your DiabeticsIntel application is ready for production deployment!**

Choose your preferred platform and follow the guide above. For beginners, **Render** is recommended for its simplicity and free tier.