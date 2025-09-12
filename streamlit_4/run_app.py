# run_app.py
"""
Script to run the Streamlit Field Recommendation App
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required model files exist"""
    required_files = [
        '../models_ANN/subject_scaler.pkl',
        '../models_ANN/field_encoder.pkl',
        '../models_ANN/board_encoder.pkl',
        '../models_ANN/combination_encoder.pkl',
        '../models_ANN/board_ohe.pkl',
        '../models_ANN/combination_ohe.pkl',
        '../models_ANN/subject_columns.pkl'
    ]
    
    # Check for model file (either .h5 or .pkl)
    model_exists = os.path.exists('field_recommendation_model.h5') or os.path.exists('field_recommendation_model.pkl')
    
    if not model_exists:
        print("❌ Model file not found! Please train the model first.")
        return False
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements!")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    print("🚀 Starting Field Recommendation System Deployment...")
    
    # Check if requirements file exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    # Check model files
    print("\n🔍 Checking model files...")
    if not check_requirements():
        print("\n💡 To generate model files, run your training script first!")
        sys.exit(1)
    
    # Run the app
    print("\n🎓 Starting Streamlit app...")
    print("🌐 The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the app\n")
    
    run_streamlit_app()

# ===================================
# .streamlit/config.toml
# (Create this file in a .streamlit folder)
# ===================================

"""
[server]
port = 8501
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
"""

# ===================================
# deploy_instructions.md
# ===================================

DEPLOYMENT_INSTRUCTIONS = """
# Field Recommendation System - Deployment Guide

## Quick Start

1. **Prepare your environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files are present:**
   - Run your training script first to generate model files
   - Required files:
     - `field_recommendation_model.h5` (or `.pkl`)
     - `subject_scaler.pkl`
     - `field_encoder.pkl`
     - `board_encoder.pkl`
     - `combination_encoder.pkl`
     - `board_ohe.pkl`
     - `combination_ohe.pkl`
     - `subject_columns.pkl`

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   Or use the helper script:
   ```bash
   python run_app.py
   ```

## Project Structure
```
your_project/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_app.py               # Helper script to run the app
├── your_training_script.py  # Your original training code
├── dataset/                 # Your dataset folder
├── .streamlit/              # Streamlit config folder
│   └── config.toml         # Streamlit configuration
└── model_files/            # Generated model files
    ├── field_recommendation_model.h5
    ├── subject_scaler.pkl
    ├── field_encoder.pkl
    ├── board_encoder.pkl
    ├── combination_encoder.pkl
    ├── board_ohe.pkl
    ├── combination_ohe.pkl
    └── subject_columns.pkl
```

## Features

### 🎯 Main Features
- **Interactive UI**: Clean, modern interface with sidebar controls
- **Real-time Predictions**: Instant field recommendations
- **Confidence Scoring**: AI confidence levels for predictions
- **Top 3 Recommendations**: Multiple field options
- **Subject Analysis**: Performance breakdown by subject
- **Visualization**: Interactive charts and graphs
- **Data Export**: Download prediction results as JSON

### 📊 Visualizations
- Confidence bar charts
- Subject performance horizontal bar charts
- Color-coded confidence indicators
- Interactive Plotly charts

### 🔧 Technical Features
- **Model Caching**: Efficient model loading with Streamlit caching
- **Error Handling**: Robust error handling and user feedback
- **Responsive Design**: Works on desktop and mobile
- **Custom Styling**: Professional CSS styling

## Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Streamlit Cloud
1. Push your code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### Docker Deployment
Create a Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Heroku Deployment
1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

## Troubleshooting

### Common Issues

1. **Model files not found:**
   - Run your training script first
   - Ensure all .pkl files are in the same directory

2. **Import errors:**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Memory issues:**
   - Consider using smaller model files
   - Implement model compression if needed

4. **Port conflicts:**
   - Use different port: `streamlit run streamlit_app.py --server.port 8502`

### Performance Tips
- Use caching for expensive operations
- Optimize model loading
- Consider model quantization for deployment
- Monitor memory usage

## Customization

### Adding New Features
- Modify `streamlit_app.py` to add new functionality
- Update the `FieldRecommendationSystem` class for new prediction methods
- Add new visualizations using Plotly or Matplotlib

### Styling Changes
- Modify the CSS in the `st.markdown()` sections
- Update colors and themes in the custom CSS
- Add new components as needed

### Configuration
- Update `.streamlit/config.toml` for app settings
- Modify default values in the sidebar inputs
- Add new subject combinations as needed

## Security Considerations
- Validate all user inputs
- Implement rate limiting for production
- Consider authentication for sensitive deployments
- Monitor usage and errors

## Monitoring & Maintenance
- Set up logging for predictions
- Monitor model performance
- Update models regularly
- Track user interactions and feedback
"""

# Save deployment instructions
with open("deploy_instructions.md", "w") as f:
    f.write(DEPLOYMENT_INSTRUCTIONS)

print("✅ Deployment files created successfully!")
print("\nCreated files:")
print("- run_app.py (Helper script to run the app)")
print("- deploy_instructions.md (Deployment guide)")
print("\nNext steps:")
print("1. Save the Streamlit app code as 'streamlit_app.py'")
print("2. Save the requirements as 'requirements.txt'") 
print("3. Ensure your model files are in the same directory")
print("4. Run: python run_app.py")