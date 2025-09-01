from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
import io
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from typing import Dict, List, Any

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scoliosis_prediction_app_2024'

class RiskCalculator:
    """Shared utility class for risk score calculations"""
    
    @staticmethod
    def calculate_risk_score(age: float, sex: str, risser_sign: int, initial_cobb: float) -> int:
        """Calculate risk score based on clinical factors"""
        risk_score = 0
        if age < 13: risk_score += 2
        if sex == 'Female': risk_score += 1
        if risser_sign <= 2: risk_score += 3
        if initial_cobb > 25: risk_score += 2
        return risk_score
    
    @staticmethod
    def get_risk_level(total_progression: float) -> str:
        """Determine risk level based on total progression"""
        if total_progression > 10:
            return 'High'
        elif total_progression > 5:
            return 'Moderate'
        else:
            return 'Low'
    
    @staticmethod
    def get_progression_probability(risk_level: str) -> float:
        """Get progression probability based on risk level"""
        probabilities = {
            'High': 0.85,
            'Moderate': 0.60,
            'Low': 0.25
        }
        return probabilities.get(risk_level, 0.5)

class ScoliosisDataGenerator:
    """Class to generate synthetic scoliosis patient data"""
    
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
    
    def generate_patient_data(self, n_patients: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic patient data with optimized performance for large datasets"""
        np.random.seed(self.seed)
        
        # Generate base demographics using vectorized operations
        ages = np.random.normal(13.2, 1.8, n_patients)
        ages = np.clip(ages, 10, 16).round(1)
        
        sexes = np.random.choice(['Female', 'Male'], n_patients, p=[0.8, 0.2])
        
        risser_signs = np.random.choice([0, 1, 2, 3, 4, 5], n_patients, 
                                       p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
        
        bmis = np.random.normal(20.5, 3.2, n_patients)
        bmis = np.clip(bmis, 15, 30).round(1)
        
        initial_cobbs = np.random.normal(22.5, 8.5, n_patients)
        initial_cobbs = np.clip(initial_cobbs, 10, 40).round(1)
        
        curve_patterns = np.random.choice(['Right thoracic', 'Left thoracic', 'Thoracolumbar', 'Double major'], 
                                         n_patients, p=[0.45, 0.15, 0.25, 0.15])
        
        flexibilities = np.random.normal(65, 15, n_patients)
        flexibilities = np.clip(flexibilities, 30, 90).round(1)
        
        # Vectorized risk score calculation
        risk_scores = np.zeros(n_patients)
        risk_scores += (ages < 13).astype(int) * 2
        risk_scores += (sexes == 'Female').astype(int)
        risk_scores += (risser_signs <= 2).astype(int) * 3
        risk_scores += (initial_cobbs > 25).astype(int) * 2
        
        # Vectorized progression calculation
        base_progression_rates = 0.5 + (risk_scores * 0.3) + np.random.normal(0, 0.2, n_patients)
        
        # Calculate future Cobb angles
        cobb_6m = np.maximum(initial_cobbs + base_progression_rates * 6 + np.random.normal(0, 1, n_patients), initial_cobbs)
        cobb_12m = np.maximum(initial_cobbs + base_progression_rates * 12 + np.random.normal(0, 1.5, n_patients), initial_cobbs)
        cobb_24m = np.maximum(initial_cobbs + base_progression_rates * 24 + np.random.normal(0, 2, n_patients), initial_cobbs)
        
        # Vectorized risk classification
        total_progressions = cobb_24m - initial_cobbs
        risk_levels = np.where(total_progressions > 10, 'High', 
                              np.where(total_progressions > 5, 'Moderate', 'Low'))
        
        # Create patient data efficiently
        patients_data = []
        for i in range(n_patients):
            patient = {
                'patient_id': f'P{i+1:04d}',
                'age': ages[i],
                'sex': sexes[i],
                'risser_sign': int(risser_signs[i]),
                'bmi': bmis[i],
                'initial_cobb': initial_cobbs[i],
                'curve_pattern': curve_patterns[i],
                'flexibility': flexibilities[i],
                'cobb_6m': round(cobb_6m[i], 1),
                'cobb_12m': round(cobb_12m[i], 1),
                'cobb_24m': round(cobb_24m[i], 1),
                'progression_risk': risk_levels[i],
                'risk_score': int(risk_scores[i])
            }
            patients_data.append(patient)
        
        return patients_data

class ScoliosisPredictor:
    """Class to predict scoliosis progression"""
    
    def __init__(self):
        pass
    
    def predict_progression(self, age: float, sex: str, risser_sign: int, bmi: float, 
                          initial_cobb: float, curve_pattern: str, flexibility: float) -> Dict[str, Any]:
        """Predict curve progression for a patient"""
        # Use shared risk calculator
        risk_score = RiskCalculator.calculate_risk_score(age, sex, risser_sign, initial_cobb)
        
        # Base progression rate
        progression_rate = 0.5 + (risk_score * 0.3)
        
        # Adjust for other factors
        if curve_pattern in ['Double major', 'Right thoracic']:
            progression_rate *= 1.1
        if flexibility < 50:
            progression_rate *= 1.2
        if bmi > 25:
            progression_rate *= 0.9
        
        # Calculate future Cobb angles
        cobb_6m = max(initial_cobb + progression_rate * 6, initial_cobb)
        cobb_12m = max(initial_cobb + progression_rate * 12, initial_cobb)
        cobb_24m = max(initial_cobb + progression_rate * 24, initial_cobb)
        
        # Calculate progression and risk level
        total_progression = cobb_24m - initial_cobb
        risk_level = RiskCalculator.get_risk_level(total_progression)
        prob_progression = RiskCalculator.get_progression_probability(risk_level)
        
        # Treatment recommendations
        if cobb_24m > 45:
            recommendation = 'Surgical consultation recommended'
        elif cobb_24m > 25:
            recommendation = 'Bracing may be considered'
        else:
            recommendation = 'Observation with regular follow-up'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'progression_probability': prob_progression,
            'predicted_cobb_6m': round(cobb_6m, 1),
            'predicted_cobb_12m': round(cobb_12m, 1),
            'predicted_cobb_24m': round(cobb_24m, 1),
            'total_progression': round(total_progression, 1),
            'recommendation': recommendation
        }

class PlotlyVisualizer:
    """Class to create interactive Plotly visualizations"""
    
    @staticmethod
    def create_dataset_analysis(df: pd.DataFrame) -> str:
        """Create interactive dataset analysis charts"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'Initial vs 24-month Cobb Angle', 
                           'Progression Risk Distribution', 'Sex Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # 1. Age distribution
        fig.add_trace(
            go.Histogram(x=df['age'], name='Age Distribution', nbinsx=15,
                        marker_color='skyblue', opacity=0.7),
            row=1, col=1
        )
        
        # 2. Initial vs 24-month Cobb angle scatter plot
        risk_colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        for risk in ['Low', 'Moderate', 'High']:
            subset = df[df['progression_risk'] == risk]
            fig.add_trace(
                go.Scatter(x=subset['initial_cobb'], y=subset['cobb_24m'],
                          mode='markers', name=f'{risk} Risk',
                          marker=dict(color=risk_colors[risk], opacity=0.6),
                          hovertemplate='Initial: %{x}°<br>24m: %{y}°<br>Risk: ' + risk),
                row=1, col=2
            )
        
        # Add no progression line
        fig.add_trace(
            go.Scatter(x=[10, 40], y=[10, 40], mode='lines',
                      line=dict(dash='dash', color='black', width=1),
                      name='No progression line', showlegend=False),
            row=1, col=2
        )
        
        # 3. Risk distribution pie chart
        risk_counts = df['progression_risk'].value_counts()
        fig.add_trace(
            go.Pie(labels=risk_counts.index, values=risk_counts.values,
                  marker_colors=[risk_colors[risk] for risk in risk_counts.index],
                  name='Risk Distribution'),
            row=2, col=1
        )
        
        # 4. Sex distribution bar chart
        sex_counts = df['sex'].value_counts()
        fig.add_trace(
            go.Bar(x=sex_counts.index, y=sex_counts.values,
                  marker_color=['pink', 'lightblue'], name='Sex Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Synthetic Dataset Analysis",
            height=700,
            showlegend=True,
            title_x=0.5
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Age (years)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Initial Cobb Angle (°)", row=1, col=2)
        fig.update_yaxes(title_text="24-month Cobb Angle (°)", row=1, col=2)
        
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    @staticmethod
    def create_prediction_visualization(prediction: Dict[str, Any], patient_data: Dict[str, Any]) -> str:
        """Create interactive prediction visualization"""
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Predicted Progression Curve', 'Risk Factors Present'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. Progression curve
        months = [0, 6, 12, 24]
        cobb_values = [
            patient_data['initial_cobb'],
            prediction['predicted_cobb_6m'],
            prediction['predicted_cobb_12m'],
            prediction['predicted_cobb_24m']
        ]
        
        risk_colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        color = risk_colors[prediction['risk_level']]
        
        # Main progression line
        fig.add_trace(
            go.Scatter(x=months, y=cobb_values, mode='lines+markers',
                      line=dict(color=color, width=3),
                      marker=dict(size=8, color=color),
                      name=f'{prediction["risk_level"]} Risk Progression',
                      hovertemplate='Month %{x}<br>Cobb Angle: %{y}°'),
            row=1, col=1
        )
        
        # Fill area under curve
        fig.add_trace(
            go.Scatter(x=months, y=cobb_values, fill='tonexty',
                      fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.3])}',
                      line=dict(color='rgba(0,0,0,0)'),
                      showlegend=False, hoverinfo='skip'),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=25, line_dash="dash", line_color="orange",
                     annotation_text="Bracing threshold", row=1, col=1)
        fig.add_hline(y=45, line_dash="dash", line_color="red",
                     annotation_text="Surgical threshold", row=1, col=1)
        
        # 2. Risk factors visualization
        risk_factors = ['Age < 13', 'Female', 'Risser ≤ 2', 'Cobb > 25°']
        risk_values = [
            1 if patient_data['age'] < 13 else 0,
            1 if patient_data['sex'] == 'Female' else 0,
            1 if patient_data['risser_sign'] <= 2 else 0,
            1 if patient_data['initial_cobb'] > 25 else 0
        ]
        
        colors = ['red' if val else 'lightgray' for val in risk_values]
        
        fig.add_trace(
            go.Bar(x=risk_factors, y=risk_values, marker_color=colors,
                  name='Risk Factors', opacity=0.7,
                  hovertemplate='%{x}<br>Present: %{y}'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Prediction Analysis - Risk Score: {prediction['risk_score']}",
            height=500,
            title_x=0.5
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (months)", row=1, col=1)
        fig.update_yaxes(title_text="Cobb Angle (°)", row=1, col=1)
        
        fig.update_xaxes(title_text="Risk Factors", row=1, col=2)
        fig.update_yaxes(title_text="Present (1) / Absent (0)", range=[0, 1.2], row=1, col=2)
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Initialize models
data_generator = ScoliosisDataGenerator()
predictor = ScoliosisPredictor()
visualizer = PlotlyVisualizer()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate_data')
def generate_data_page():
    """Data generation page"""
    return render_template('generate_data.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html')

@app.route('/api/generate_data', methods=['POST'])
def api_generate_data():
    """API endpoint to generate synthetic data"""
    try:
        data = request.get_json()
        n_patients = int(data.get('n_patients', 100))
        n_patients = min(max(n_patients, 10), 10000)  # Limit between 10 and 10000
        
        patients_data = data_generator.generate_patient_data(n_patients)
        
        return jsonify({
            'success': True,
            'data': patients_data,
            'summary': {
                'total_patients': len(patients_data),
                'high_risk': len([p for p in patients_data if p['progression_risk'] == 'High']),
                'moderate_risk': len([p for p in patients_data if p['progression_risk'] == 'Moderate']),
                'low_risk': len([p for p in patients_data if p['progression_risk'] == 'Low']),
                'avg_age': round(np.mean([p['age'] for p in patients_data]), 1),
                'avg_initial_cobb': round(np.mean([p['initial_cobb'] for p in patients_data]), 1)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        
        # Extract patient data
        age = float(data['age'])
        sex = data['sex']
        risser_sign = int(data['risser_sign'])
        bmi = float(data['bmi'])
        initial_cobb = float(data['initial_cobb'])
        curve_pattern = data['curve_pattern']
        flexibility = float(data['flexibility'])
        
        # Make prediction
        prediction = predictor.predict_progression(
            age, sex, risser_sign, bmi, initial_cobb, curve_pattern, flexibility
        )
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize_data', methods=['POST'])
def api_visualize_data():
    """API endpoint to create interactive data visualizations"""
    try:
        data = request.get_json()
        patients_data = data['patients_data']
        
        # Create DataFrame
        df = pd.DataFrame(patients_data)
        
        # Create interactive visualization
        plot_json = visualizer.create_dataset_analysis(df)
        
        return jsonify({
            'success': True,
            'plot_json': plot_json
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize_prediction', methods=['POST'])
def api_visualize_prediction():
    """API endpoint to visualize prediction results"""
    try:
        data = request.get_json()
        prediction = data['prediction']
        patient_data = data['patient_data']
        
        # Create interactive prediction visualization
        plot_json = visualizer.create_prediction_visualization(prediction, patient_data)
        
        return jsonify({
            'success': True,
            'plot_json': plot_json
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_data', methods=['POST'])
def api_download_data():
    """API endpoint to download generated data as CSV"""
    try:
        data = request.get_json()
        patients_data = data['patients_data']
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(patients_data)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Create a BytesIO object for the file
        csv_buffer = io.BytesIO()
        csv_buffer.write(output.getvalue().encode('utf-8'))
        csv_buffer.seek(0)
        
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'scoliosis_synthetic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Make the app available for WSGI deployment
application = app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
