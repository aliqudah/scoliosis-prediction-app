from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import io
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scoliosis_prediction_app_2024'

class RiskCalculator:
    """Shared utility class for calculating scoliosis progression risk scores"""
    
    @staticmethod
    def calculate_risk_score(age, sex, risser_sign, initial_cobb):
        """Calculate risk score based on clinical factors"""
        risk_score = 0
        if age < 13: risk_score += 2
        if sex == 'Female': risk_score += 1
        if risser_sign <= 2: risk_score += 3
        if initial_cobb > 25: risk_score += 2
        return risk_score
    
    @staticmethod
    def get_risk_level(total_progression):
        """Determine risk level based on total progression"""
        if total_progression > 10:
            return 'High'
        elif total_progression > 5:
            return 'Moderate'
        else:
            return 'Low'

class ScoliosisDataGenerator:
    """Class to generate synthetic scoliosis patient data"""
    
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
    
    def generate_patient_data(self, n_patients=100):
        """Generate synthetic patient data"""
        # Limit to 10,000 patients for performance
        n_patients = min(max(n_patients, 10), 10000)
        np.random.seed(self.seed)
        
        # Generate base demographics
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
        
        # Calculate progression based on risk factors
        patients_data = []
        for i in range(n_patients):
            # Use shared risk calculator
            risk_score = RiskCalculator.calculate_risk_score(
                ages[i], sexes[i], risser_signs[i], initial_cobbs[i]
            )
            
            # Progression calculation
            base_progression = initial_cobbs[i]
            progression_rate = 0.5 + (risk_score * 0.3) + np.random.normal(0, 0.2)
            
            # Calculate future Cobb angles
            cobb_6m = max(base_progression + progression_rate * 6 + np.random.normal(0, 1), base_progression)
            cobb_12m = max(base_progression + progression_rate * 12 + np.random.normal(0, 1.5), base_progression)
            cobb_24m = max(base_progression + progression_rate * 24 + np.random.normal(0, 2), base_progression)
            
            # Risk classification using shared utility
            total_progression = cobb_24m - base_progression
            risk_level = RiskCalculator.get_risk_level(total_progression)
            
            patient = {
                'patient_id': f'P{i+1:04d}',
                'age': ages[i],
                'sex': sexes[i],
                'risser_sign': int(risser_signs[i]),
                'bmi': bmis[i],
                'initial_cobb': initial_cobbs[i],
                'curve_pattern': curve_patterns[i],
                'flexibility': flexibilities[i],
                'cobb_6m': round(cobb_6m, 1),
                'cobb_12m': round(cobb_12m, 1),
                'cobb_24m': round(cobb_24m, 1),
                'progression_risk': risk_level,
                'risk_score': risk_score
            }
            patients_data.append(patient)
        
        return patients_data

class ScoliosisPredictor:
    """Class to predict scoliosis progression"""
    
    def predict_progression(self, age, sex, risser_sign, bmi, initial_cobb, curve_pattern, flexibility):
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
        
        # Calculate progression probabilities
        if total_progression > 10:
            prob_progression = 0.85
        elif total_progression > 5:
            prob_progression = 0.60
        else:
            prob_progression = 0.25
        
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

# Initialize models
data_generator = ScoliosisDataGenerator()
predictor = ScoliosisPredictor()

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
    """API endpoint to create interactive visualizations"""
    try:
        data = request.get_json()
        patients_data = data['patients_data']
        
        # Create DataFrame
        df = pd.DataFrame(patients_data)
        
        # Create interactive visualizations using Plotly
        # 1. Age distribution
        fig1 = px.histogram(df, x='age', nbins=15, title='Age Distribution',
                           labels={'age': 'Age (years)', 'count': 'Frequency'},
                           color_discrete_sequence=['skyblue'])
        
        # 2. Initial vs 24-month Cobb angle scatter
        fig2 = px.scatter(df, x='initial_cobb', y='cobb_24m', color='progression_risk',
                         title='Initial vs 24-month Cobb Angle',
                         labels={'initial_cobb': 'Initial Cobb Angle (°)', 
                                'cobb_24m': '24-month Cobb Angle (°)'},
                         color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
        
        # Add diagonal line for no progression
        fig2.add_shape(type="line", x0=10, y0=10, x1=40, y1=40,
                      line=dict(color="black", width=2, dash="dash"))
        
        # 3. Risk distribution pie chart
        risk_counts = df['progression_risk'].value_counts()
        fig3 = px.pie(values=risk_counts.values, names=risk_counts.index,
                     title='Progression Risk Distribution',
                     color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
        
        # 4. Sex distribution bar chart
        sex_counts = df['sex'].value_counts()
        fig4 = px.bar(x=sex_counts.index, y=sex_counts.values,
                     title='Sex Distribution',
                     labels={'x': 'Sex', 'y': 'Count'},
                     color=sex_counts.index,
                     color_discrete_map={'Female': 'pink', 'Male': 'lightblue'})
        
        # Convert plots to JSON for frontend
        graphs = {
            'age_dist': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
            'cobb_scatter': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
            'risk_pie': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
            'sex_bar': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        return jsonify({
            'success': True,
            'graphs': graphs
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
        
        # Create progression curve
        months = [0, 6, 12, 24]
        cobb_values = [
            patient_data['initial_cobb'],
            prediction['predicted_cobb_6m'],
            prediction['predicted_cobb_12m'],
            prediction['predicted_cobb_24m']
        ]
        
        risk_colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        color = risk_colors[prediction['risk_level']]
        
        # Progression curve
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=months, y=cobb_values, mode='lines+markers',
                                 name='Progression', line=dict(color=color, width=3),
                                 marker=dict(size=8)))
        
        # Add threshold lines
        fig1.add_hline(y=25, line_dash="dash", line_color="orange", 
                      annotation_text="Bracing threshold")
        fig1.add_hline(y=45, line_dash="dash", line_color="red",
                      annotation_text="Surgical threshold")
        
        fig1.update_layout(title=f'Predicted Progression Curve ({prediction["risk_level"]} Risk)',
                          xaxis_title='Time (months)',
                          yaxis_title='Cobb Angle (°)')
        
        # Risk factors visualization
        risk_factors = ['Age < 13', 'Female', 'Risser ≤ 2', 'Cobb > 25°']
        risk_values = [
            1 if patient_data['age'] < 13 else 0,
            1 if patient_data['sex'] == 'Female' else 0,
            1 if patient_data['risser_sign'] <= 2 else 0,
            1 if patient_data['initial_cobb'] > 25 else 0
        ]
        
        colors = ['red' if val else 'lightgray' for val in risk_values]
        
        fig2 = go.Figure(data=[go.Bar(x=risk_factors, y=risk_values, marker_color=colors)])
        fig2.update_layout(title=f'Risk Factors Present (Total Score: {prediction["risk_score"]})',
                          yaxis_title='Present (1) / Absent (0)',
                          yaxis=dict(range=[0, 1.2]))
        
        # Convert plots to JSON
        graphs = {
            'progression': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
            'risk_factors': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        return jsonify({
            'success': True,
            'graphs': graphs
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
