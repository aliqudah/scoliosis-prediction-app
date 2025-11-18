import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for professional appearance
plt.style.use('default')
sns.set_palette("husl")

# Generate sample data that matches our synthetic dataset
np.random.seed(42)

# Create sample patient data
n_samples = 10
sample_data = {
    'Patient_ID': [f'P{i+1:03d}' for i in range(n_samples)],
    'Age': np.random.normal(13.2, 1.8, n_samples).round(1),
    'Sex': np.random.choice(['Female', 'Male'], n_samples, p=[0.8, 0.2]),
    'Risser_Sign': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
    'BMI': np.random.normal(20.5, 3.2, n_samples).round(1),
    'Initial_Cobb': np.random.normal(22.5, 8.5, n_samples).round(1),
    'Curve_Pattern': np.random.choice(['Right thoracic', 'Left thoracic', 'Thoracolumbar', 'Double major'], 
                                     n_samples, p=[0.45, 0.15, 0.25, 0.15]),
    'Flexibility': np.random.normal(65, 15, n_samples).round(1),
    'Cobb_6m': [],
    'Cobb_12m': [],
    'Cobb_24m': [],
    'Progression_Risk': []
}

# Calculate progression based on risk factors
for i in range(n_samples):
    # Risk score calculation (simplified)
    risk_score = 0
    if sample_data['Age'][i] < 13: risk_score += 2
    if sample_data['Sex'][i] == 'Female': risk_score += 1
    if sample_data['Risser_Sign'][i] <= 2: risk_score += 3
    if sample_data['Initial_Cobb'][i] > 25: risk_score += 2
    
    # Progression calculation
    base_progression = sample_data['Initial_Cobb'][i]
    progression_rate = 0.5 + (risk_score * 0.3) + np.random.normal(0, 0.2)
    
    cobb_6m = base_progression + progression_rate * 6 + np.random.normal(0, 1)
    cobb_12m = base_progression + progression_rate * 12 + np.random.normal(0, 1.5)
    cobb_24m = base_progression + progression_rate * 24 + np.random.normal(0, 2)
    
    sample_data['Cobb_6m'].append(round(max(cobb_6m, base_progression), 1))
    sample_data['Cobb_12m'].append(round(max(cobb_12m, base_progression), 1))
    sample_data['Cobb_24m'].append(round(max(cobb_24m, base_progression), 1))
    
    # Risk classification
    total_progression = cobb_24m - base_progression
    if total_progression > 10:
        risk = 'High'
    elif total_progression > 5:
        risk = 'Moderate'
    else:
        risk = 'Low'
    sample_data['Progression_Risk'].append(risk)

# Create DataFrame
df = pd.DataFrame(sample_data)

# Create the figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.suptitle('SAMPLE PATIENTS FROM SYNTHETIC DATASET', fontsize=20, fontweight='bold', y=0.95)

# 1. Patient Table (top-left)
ax1.axis('tight')
ax1.axis('off')
ax1.set_title('Sample Patient Records', fontsize=16, fontweight='bold', pad=20)

# Create table data
table_data = []
headers = ['ID', 'Age', 'Sex', 'Risser', 'BMI', 'Init Cobb', 'Pattern', 'Risk']
for i in range(min(8, len(df))):
    row = [
        df.iloc[i]['Patient_ID'],
        f"{df.iloc[i]['Age']:.1f}",
        df.iloc[i]['Sex'][0],  # F or M
        str(df.iloc[i]['Risser_Sign']),
        f"{df.iloc[i]['BMI']:.1f}",
        f"{df.iloc[i]['Initial_Cobb']:.1f}°",
        df.iloc[i]['Curve_Pattern'][:8] + '...' if len(df.iloc[i]['Curve_Pattern']) > 8 else df.iloc[i]['Curve_Pattern'],
        df.iloc[i]['Progression_Risk']
    ]
    table_data.append(row)

table = ax1.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                 colWidths=[0.08, 0.08, 0.06, 0.08, 0.08, 0.1, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4A9BE8')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 2. Progression Curves (top-right)
ax2.set_title('Individual Patient Progression Curves', fontsize=16, fontweight='bold')
months = [0, 6, 12, 24]

for i in range(min(5, len(df))):
    cobb_values = [
        df.iloc[i]['Initial_Cobb'],
        df.iloc[i]['Cobb_6m'],
        df.iloc[i]['Cobb_12m'],
        df.iloc[i]['Cobb_24m']
    ]
    
    color = 'red' if df.iloc[i]['Progression_Risk'] == 'High' else 'orange' if df.iloc[i]['Progression_Risk'] == 'Moderate' else 'green'
    ax2.plot(months, cobb_values, marker='o', linewidth=2, alpha=0.7, 
             label=f"{df.iloc[i]['Patient_ID']} ({df.iloc[i]['Progression_Risk']})", color=color)

ax2.set_xlabel('Time (months)', fontsize=12)
ax2.set_ylabel('Cobb Angle (degrees)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(-1, 25)

# Add threshold lines
ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Bracing threshold')
ax2.axhline(y=45, color='red', linestyle='--', alpha=0.7, label='Surgical threshold')

# 3. Risk Factor Distribution (bottom-left)
ax3.set_title('Distribution of Risk Factors in Sample', fontsize=16, fontweight='bold')

# Create risk factor visualization
risk_factors = ['Age < 13', 'Female', 'Risser ≤ 2', 'Cobb > 25°']
risk_counts = [
    sum(df['Age'] < 13),
    sum(df['Sex'] == 'Female'),
    sum(df['Risser_Sign'] <= 2),
    sum(df['Initial_Cobb'] > 25)
]

bars = ax3.bar(risk_factors, risk_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax3.set_ylabel('Number of Patients', fontsize=12)
ax3.set_ylim(0, len(df))

# Add value labels on bars
for bar, count in zip(bars, risk_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# 4. Outcome Distribution (bottom-right)
ax4.set_title('Progression Risk Distribution', fontsize=16, fontweight='bold')

risk_counts = df['Progression_Risk'].value_counts()
colors = ['green', 'orange', 'red']
wedges, texts, autotexts = ax4.pie(risk_counts.values, labels=risk_counts.index, 
                                  autopct='%1.1f%%', colors=colors, startangle=90)

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

plt.tight_layout()
plt.savefig('/home/ubuntu/scoliosis_prediction/high_res_figures/dataset_samples.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Dataset samples figure created successfully!")

