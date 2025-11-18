from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scoliosis_prediction_app_2024_enhanced'

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class ScoliosisDataGenerator:
    """Enhanced class to generate synthetic scoliosis patient data"""
    
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
    
    def generate_patient_data(self, n_patients=100):
        """Generate synthetic patient data - now supports up to 10,000 patients"""
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
            # Risk score calculation
            risk_score = 0
            if ages[i] < 13: risk_score += 2
            if sexes[i] == 'Female': risk_score += 1
            if risser_signs[i] <= 2: risk_score += 3
            if initial_cobbs[i] > 25: risk_score += 2
            
            # Progression calculation
            base_progression = initial_cobbs[i]
            progression_rate = 0.5 + (risk_score * 0.3) + np.random.normal(0, 0.2)
            
            # Calculate future Cobb angles
            cobb_6m = max(base_progression + progression_rate * 6 + np.random.normal(0, 1), base_progression)
            cobb_12m = max(base_progression + progression_rate * 12 + np.random.normal(0, 1.5), base_progression)
            cobb_24m = max(base_progression + progression_rate * 24 + np.random.normal(0, 2), base_progression)
            
            # Risk classification
            total_progression = cobb_24m - base_progression
            if total_progression > 10:
                risk_level = 'High'
            elif total_progression > 5:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            
            patient = {
                'patient_id': f'P{i+1:05d}',  # Support up to 99,999 patients
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
    """Enhanced class to predict scoliosis progression with comprehensive figure generation"""
    
    def __init__(self):
        pass
    
    def calculate_risk_score(self, age, sex, risser_sign, initial_cobb):
        """Calculate risk score based on clinical factors"""
        risk_score = 0
        if age < 13: risk_score += 2
        if sex == 'Female': risk_score += 1
        if risser_sign <= 2: risk_score += 3
        if initial_cobb > 25: risk_score += 2
        return risk_score
    
    def predict_progression(self, age, sex, risser_sign, bmi, initial_cobb, curve_pattern, flexibility):
        """Predict curve progression for a patient"""
        risk_score = self.calculate_risk_score(age, sex, risser_sign, initial_cobb)
        
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
        
        # Calculate progression probabilities
        total_progression = cobb_24m - initial_cobb
        if total_progression > 10:
            risk_level = 'High'
            prob_progression = 0.85
        elif total_progression > 5:
            risk_level = 'Moderate'
            prob_progression = 0.60
        else:
            risk_level = 'Low'
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
    
    def generate_comprehensive_figures(self, prediction, patient_data):
        """Generate all paper figures for prediction results"""
        figures = {}
        
        # 1. Model Architecture Figure (static - from paper)
        figures['model_architecture'] = '/static/images/high_res_figures/detailed_model_architecture_programmatic.png'
        
        # 2. Patient Progression Curve
        fig1, ax1 = plt.subplots(figsize=(12, 8), dpi=150)
        months = [0, 6, 12, 24]
        cobb_values = [
            patient_data['initial_cobb'],
            prediction['predicted_cobb_6m'],
            prediction['predicted_cobb_12m'],
            prediction['predicted_cobb_24m']
        ]
        
        risk_colors = {'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        color = risk_colors[prediction['risk_level']]
        
        ax1.plot(months, cobb_values, marker='o', linewidth=4, color=color, markersize=10, label=f'{prediction["risk_level"]} Risk Patient')
        ax1.fill_between(months, cobb_values, alpha=0.3, color=color)
        ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Bracing threshold (25°)')
        ax1.axhline(y=45, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Surgical threshold (45°)')
        ax1.set_title(f'Predicted Curve Progression\n({prediction["risk_level"]} Risk - {prediction["progression_probability"]*100:.0f}% probability)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time (months)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cobb Angle (degrees)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, 25)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Add annotations
        for i, (month, cobb) in enumerate(zip(months, cobb_values)):
            if i > 0:
                progression = cobb - patient_data['initial_cobb']
                ax1.annotate(f'+{progression:.1f}°', (month, cobb), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        img_buffer1 = io.BytesIO()
        plt.savefig(img_buffer1, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer1.seek(0)
        figures['progression_curve'] = base64.b64encode(img_buffer1.getvalue()).decode()
        plt.close()
        
        # 3. Risk Factor Analysis
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
        
        # Risk factors bar chart
        risk_factors = ['Age < 13 years', 'Female sex', 'Risser ≤ 2', 'Initial Cobb > 25°']
        risk_values = [
            1 if patient_data['age'] < 13 else 0,
            1 if patient_data['sex'] == 'Female' else 0,
            1 if patient_data['risser_sign'] <= 2 else 0,
            1 if patient_data['initial_cobb'] > 25 else 0
        ]
        
        colors = ['#e74c3c' if val else '#bdc3c7' for val in risk_values]
        bars = ax2.bar(risk_factors, risk_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_title(f'Risk Factors Present\n(Total Score: {prediction["risk_score"]}/8)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Present (1) / Absent (0)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.2)
        
        # Add value labels
        for bar, val, factor in zip(bars, risk_values, risk_factors):
            if val:
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.05,
                        'PRESENT', ha='center', va='bottom', fontweight='bold', color='red')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2., 0.05,
                        'ABSENT', ha='center', va='bottom', fontweight='bold', color='gray')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        # Risk level pie chart
        risk_distribution = {'Low': 0, 'Moderate': 0, 'High': 0}
        risk_distribution[prediction['risk_level']] = 1
        
        colors_pie = ['#27ae60', '#f39c12', '#e74c3c']
        sizes = [risk_distribution['Low'], risk_distribution['Moderate'], risk_distribution['High']]
        labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        
        # Only show the current patient's risk level
        explode = [0.1 if prediction['risk_level'] == level else 0 for level in ['Low', 'Moderate', 'High']]
        
        wedges, texts, autotexts = ax3.pie([1], labels=[f'{prediction["risk_level"]} Risk\n({prediction["progression_probability"]*100:.0f}% probability)'], 
                                          colors=[risk_colors[prediction['risk_level']]], startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax3.set_title('Patient Risk Classification', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        img_buffer2 = io.BytesIO()
        plt.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer2.seek(0)
        figures['risk_analysis'] = base64.b64encode(img_buffer2.getvalue()).decode()
        plt.close()
        
        # 4. Treatment Timeline
        fig3, ax4 = plt.subplots(figsize=(14, 8), dpi=150)
        
        # Create treatment timeline
        timepoints = ['Baseline', '6 months', '12 months', '24 months']
        cobb_angles = [patient_data['initial_cobb'], prediction['predicted_cobb_6m'], 
                      prediction['predicted_cobb_12m'], prediction['predicted_cobb_24m']]
        
        # Treatment zones
        ax4.axhspan(0, 25, alpha=0.2, color='green', label='Observation zone (< 25°)')
        ax4.axhspan(25, 45, alpha=0.2, color='orange', label='Bracing zone (25-45°)')
        ax4.axhspan(45, 60, alpha=0.2, color='red', label='Surgical zone (> 45°)')
        
        # Plot progression
        ax4.plot(timepoints, cobb_angles, marker='o', linewidth=4, markersize=12, 
                color=risk_colors[prediction['risk_level']], label=f'Patient progression ({prediction["risk_level"]} risk)')
        
        # Add treatment recommendations at each timepoint
        recommendations = []
        for cobb in cobb_angles:
            if cobb > 45:
                recommendations.append('Surgery')
            elif cobb > 25:
                recommendations.append('Bracing')
            else:
                recommendations.append('Observation')
        
        for i, (tp, cobb, rec) in enumerate(zip(timepoints, cobb_angles, recommendations)):
            ax4.annotate(f'{cobb}°\n{rec}', (i, cobb), 
                        textcoords="offset points", xytext=(0,15), ha='center', 
                        fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax4.set_title('Treatment Timeline and Recommendations', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Cobb Angle (degrees)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Follow-up Timepoints', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=12, loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(60, max(cobb_angles) + 10))
        
        plt.tight_layout()
        img_buffer3 = io.BytesIO()
        plt.savefig(img_buffer3, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer3.seek(0)
        figures['treatment_timeline'] = base64.b64encode(img_buffer3.getvalue()).decode()
        plt.close()
        
        # 5. Comparison with Population Data
        fig4, ax5 = plt.subplots(figsize=(12, 8), dpi=150)
        
        # Generate comparison data
        np.random.seed(42)
        n_comparison = 100
        comparison_progressions = []
        
        for _ in range(n_comparison):
            # Generate random patients with similar characteristics
            comp_age = np.random.normal(patient_data['age'], 1)
            comp_sex = np.random.choice(['Female', 'Male'], p=[0.8, 0.2])
            comp_risser = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
            comp_initial = np.random.normal(patient_data['initial_cobb'], 5)
            
            # Calculate risk score
            comp_risk = 0
            if comp_age < 13: comp_risk += 2
            if comp_sex == 'Female': comp_risk += 1
            if comp_risser <= 2: comp_risk += 3
            if comp_initial > 25: comp_risk += 2
            
            # Calculate progression
            comp_rate = 0.5 + (comp_risk * 0.3) + np.random.normal(0, 0.2)
            comp_24m = max(comp_initial + comp_rate * 24, comp_initial)
            comparison_progressions.append(comp_24m - comp_initial)
        
        # Plot histogram of comparison data
        ax5.hist(comparison_progressions, bins=20, alpha=0.6, color='lightblue', 
                edgecolor='black', label='Similar patients (n=100)')
        
        # Mark current patient
        patient_progression = prediction['predicted_cobb_24m'] - patient_data['initial_cobb']
        ax5.axvline(patient_progression, color=risk_colors[prediction['risk_level']], 
                   linewidth=4, label=f'Current patient (+{patient_progression:.1f}°)')
        
        ax5.set_title('Patient Progression Compared to Similar Cases', fontsize=16, fontweight='bold', pad=20)
        ax5.set_xlabel('24-month Progression (degrees)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Number of Patients', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Add percentile information
        percentile = (np.sum(np.array(comparison_progressions) < patient_progression) / len(comparison_progressions)) * 100
        ax5.text(0.7, 0.9, f'Patient is in {percentile:.0f}th percentile\nfor progression', 
                transform=ax5.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        img_buffer4 = io.BytesIO()
        plt.savefig(img_buffer4, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer4.seek(0)
        figures['population_comparison'] = base64.b64encode(img_buffer4.getvalue()).decode()
        plt.close()
        
        return figures

# Initialize models
data_generator = ScoliosisDataGenerator()
predictor = ScoliosisPredictor()

@app.route('/')
def index():
    """Main page with paper figures"""
    return render_template('index.html')

@app.route('/generate_data')
def generate_data_page():
    """Data generation page"""
    return render_template('generate_data.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html')

@app.route('/paper_figures')
def paper_figures_page():
    """Page displaying all paper figures"""
    return render_template('paper_figures.html')

@app.route('/api/generate_data', methods=['POST'])
def api_generate_data():
    """API endpoint to generate synthetic data - now supports up to 10,000 patients"""
    try:
        data = request.get_json()
        n_patients = int(data.get('n_patients', 100))
        n_patients = min(max(n_patients, 10), 10000)  # Limit between 10 and 10,000
        
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
                'avg_initial_cobb': round(np.mean([p['initial_cobb'] for p in patients_data]), 1),
                'female_percentage': round(len([p for p in patients_data if p['sex'] == 'Female']) / len(patients_data) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction with comprehensive figure generation"""
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
        
        patient_data = {
            'age': age,
            'sex': sex,
            'risser_sign': risser_sign,
            'bmi': bmi,
            'initial_cobb': initial_cobb,
            'curve_pattern': curve_pattern,
            'flexibility': flexibility
        }
        
        # Make prediction
        prediction = predictor.predict_progression(
            age, sex, risser_sign, bmi, initial_cobb, curve_pattern, flexibility
        )
        
        # Generate comprehensive figures
        figures = predictor.generate_comprehensive_figures(prediction, patient_data)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'figures': figures
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize_data', methods=['POST'])
def api_visualize_data():
    """Enhanced API endpoint to create comprehensive data visualizations"""
    try:
        data = request.get_json()
        patients_data = data['patients_data']
        
        # Create DataFrame
        df = pd.DataFrame(patients_data)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16), dpi=150)
        fig.suptitle('Comprehensive Synthetic Dataset Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.08)
        
        # 1. Age distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Age Distribution', fontweight='bold')
        ax1.set_xlabel('Age (years)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Initial Cobb vs 24m Cobb by risk
        ax2 = fig.add_subplot(gs[0, 1])
        risk_colors = {'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#e74c3c'}
        for risk in ['Low', 'Moderate', 'High']:
            subset = df[df['progression_risk'] == risk]
            ax2.scatter(subset['initial_cobb'], subset['cobb_24m'], 
                       label=f'{risk} Risk', alpha=0.7, color=risk_colors[risk], s=50)
        ax2.plot([10, 40], [10, 40], 'k--', alpha=0.5, label='No progression')
        ax2.set_title('Initial vs 24-month Cobb Angle', fontweight='bold')
        ax2.set_xlabel('Initial Cobb Angle (°)')
        ax2.set_ylabel('24-month Cobb Angle (°)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk distribution pie chart
        ax3 = fig.add_subplot(gs[0, 2])
        risk_counts = df['progression_risk'].value_counts()
        colors = [risk_colors[risk] for risk in risk_counts.index]
        wedges, texts, autotexts = ax3.pie(risk_counts.values, labels=risk_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Risk Distribution', fontweight='bold')
        
        # 4. Sex distribution
        ax4 = fig.add_subplot(gs[1, 0])
        sex_counts = df['sex'].value_counts()
        bars = ax4.bar(sex_counts.index, sex_counts.values, color=['pink', 'lightblue'], alpha=0.7)
        ax4.set_title('Sex Distribution', fontweight='bold')
        ax4.set_ylabel('Count')
        for bar, count in zip(bars, sex_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Risser sign distribution
        ax5 = fig.add_subplot(gs[1, 1])
        risser_counts = df['risser_sign'].value_counts().sort_index()
        ax5.bar(risser_counts.index, risser_counts.values, color='lightgreen', alpha=0.7)
        ax5.set_title('Risser Sign Distribution', fontweight='bold')
        ax5.set_xlabel('Risser Sign')
        ax5.set_ylabel('Count')
        
        # 6. BMI distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(df['bmi'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_title('BMI Distribution', fontweight='bold')
        ax6.set_xlabel('BMI (kg/m²)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        # 7. Progression over time
        ax7 = fig.add_subplot(gs[2, :])
        months = [0, 6, 12, 24]
        
        # Sample some patients for progression curves
        sample_size = min(50, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for _, patient in sample_df.iterrows():
            cobb_values = [patient['initial_cobb'], patient['cobb_6m'], 
                          patient['cobb_12m'], patient['cobb_24m']]
            color = risk_colors[patient['progression_risk']]
            alpha = 0.3 if patient['progression_risk'] == 'Low' else 0.6
            ax7.plot(months, cobb_values, color=color, alpha=alpha, linewidth=1)
        
        # Add average progression lines
        for risk in ['Low', 'Moderate', 'High']:
            subset = df[df['progression_risk'] == risk]
            if len(subset) > 0:
                avg_progression = [
                    subset['initial_cobb'].mean(),
                    subset['cobb_6m'].mean(),
                    subset['cobb_12m'].mean(),
                    subset['cobb_24m'].mean()
                ]
                ax7.plot(months, avg_progression, color=risk_colors[risk], 
                        linewidth=4, label=f'{risk} Risk (Average)', marker='o', markersize=8)
        
        ax7.axhline(y=25, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Bracing threshold')
        ax7.axhline(y=45, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Surgical threshold')
        ax7.set_title('Progression Curves by Risk Level', fontweight='bold', fontsize=16)
        ax7.set_xlabel('Time (months)', fontweight='bold')
        ax7.set_ylabel('Cobb Angle (degrees)', fontweight='bold')
        ax7.legend(fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': img_str
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
    app.run(host='0.0.0.0', port=5000, debug=False)

