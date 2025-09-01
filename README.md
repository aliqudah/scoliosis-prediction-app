# Scoliosis Prediction System

An advanced web application for predicting adolescent idiopathic scoliosis curve progression using interactive visualizations and machine learning algorithms.

## Features

### 🔢 Enhanced Data Generation
- **Up to 10,000 synthetic patients** (increased from 1,000)
- Realistic demographics and clinical parameters
- Multiple curve patterns and progression scenarios
- Follow-up data at 6, 12, and 24 months

### 📊 Interactive Visualizations
- **Plotly-based interactive charts** (replacing static matplotlib)
- Zoom, hover, and filter capabilities
- Age distributions and sex demographics
- Cobb angle progression scatter plots
- Risk level pie charts
- Responsive design for all screen sizes

### 🎯 Prediction & Risk Assessment
- Multi-timepoint curve progression prediction
- Risk stratification (Low, Moderate, High)
- Treatment recommendations
- Interactive progression curves
- Risk factor analysis

### 🔧 Technical Improvements
- **Consolidated risk calculation logic** (eliminated duplicates)
- Optimized performance for large datasets
- Professional medical interface
- Data export capabilities (CSV/JSON)

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd scoliosis_deploy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser to `http://localhost:5000`

## Railway Deployment

### Method 1: GitHub Integration (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect the Flask app and deploy

### Method 2: Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

## Configuration Files

- **`requirements.txt`**: Python dependencies including Plotly
- **`Procfile`**: Process configuration for deployment
- **`railway.json`**: Railway-specific deployment settings
- **`app.py`**: Main Flask application with all enhancements

## Application Structure

```
scoliosis_deploy/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Process configuration
├── railway.json          # Railway deployment config
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template with Plotly
│   ├── index.html        # Homepage
│   ├── generate_data.html # Data generation page
│   └── predict.html      # Prediction page
└── static/               # Static assets
    ├── css/
    ├── js/
    └── images/
```

## Key Enhancements Made

### 1. **Increased Sample Capacity**
- Maximum patients increased from 1,000 to 10,000
- Optimized data generation algorithms
- Better memory management for large datasets

### 2. **Interactive Visualizations**
- Replaced static matplotlib with interactive Plotly charts
- Added zoom, pan, hover, and filter capabilities
- Responsive design for mobile and desktop

### 3. **Code Optimization**
- Created `RiskCalculator` utility class
- Eliminated duplicate risk calculation logic
- Improved maintainability and performance

### 4. **Enhanced User Experience**
- Professional medical interface design
- Loading indicators for large operations
- Improved error handling and validation
- Example cases for quick testing

## API Endpoints

- **`POST /api/generate_data`**: Generate synthetic patient data
- **`POST /api/predict`**: Calculate curve progression prediction
- **`POST /api/visualize_data`**: Generate interactive data visualizations
- **`POST /api/visualize_prediction`**: Generate prediction charts
- **`POST /api/download_data`**: Download data as CSV

## Clinical Parameters

### Risk Factors (Scoring System)
- **Age < 13 years**: +2 points
- **Female sex**: +1 point
- **Risser sign ≤ 2**: +3 points
- **Initial Cobb > 25°**: +2 points

### Risk Levels
- **Low Risk**: ≤5° progression over 24 months
- **Moderate Risk**: 5-10° progression over 24 months  
- **High Risk**: >10° progression over 24 months

### Treatment Thresholds
- **Observation**: Cobb angle < 25°
- **Bracing consideration**: Cobb angle 25-45°
- **Surgical consultation**: Cobb angle > 45°

## Dependencies

- **Flask 2.3.3**: Web framework
- **NumPy 1.24.3**: Numerical computations
- **Pandas 2.0.3**: Data manipulation
- **Plotly 5.15.0**: Interactive visualizations
- **Gunicorn 20.1.0**: WSGI HTTP Server for deployment

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is for research and educational purposes. Clinical decisions should always be made by qualified healthcare professionals.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For technical issues or questions about deployment, please check the Railway documentation or create an issue in the repository.
