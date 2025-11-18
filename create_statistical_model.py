import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Create figure for statistical model explanation
fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
plt.text(8, 9.5, 'STATISTICAL MODEL FOR SYNTHETIC DATASET GENERATION', 
         fontsize=20, fontweight='bold', ha='center')

# Define colors
input_color = '#E3F2FD'  # Light blue
process_color = '#E8F5E8'  # Light green
model_color = '#F3E5F5'  # Light purple
output_color = '#FFF3E0'  # Light orange

# Function to create rounded rectangle
def create_rounded_box(x, y, width, height, title, content_lines, color, title_color='black'):
    # Main box
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                        facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Title
    plt.text(x + width/2, y + height - 0.5, title, 
             fontsize=14, fontweight='bold', ha='center', va='center', color=title_color)
    
    # Content
    line_height = (height - 1) / (len(content_lines) + 1)
    for i, line in enumerate(content_lines):
        plt.text(x + width/2, y + height - 1.2 - (i * line_height), 
                 line, fontsize=11, ha='center', va='center')

# Function to create arrow
def create_arrow(start_x, start_y, end_x, end_y, label=''):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    if label:
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        plt.text(mid_x, mid_y + 0.2, label, fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 1. Literature Review Input
create_rounded_box(1, 7, 3, 2.5, 'LITERATURE REVIEW',
                  ['• Weinstein et al. (2003)', '• Lonstein & Carlson (1984)', 
                   '• Sanders et al. (2007)', '• Nault et al. (2002)',
                   '• Risk factor analysis', '• Progression rates'], input_color)

# 2. Statistical Parameters
create_rounded_box(5.5, 7, 5, 2.5, 'STATISTICAL PARAMETERS EXTRACTION',
                  ['Age: μ=13.2, σ=1.8 (Normal distribution)',
                   'Initial Cobb: μ=22.5, σ=8.5 (Normal distribution)',
                   'Sex ratio: 80% Female, 20% Male',
                   'Risser distribution: Weighted toward lower stages',
                   'BMI: μ=20.5, σ=3.2 (Normal distribution)',
                   'Progression rates: Risk-factor dependent'], process_color)

# 3. Risk Score Model
create_rounded_box(12, 7, 3, 2.5, 'RISK SCORE MODEL',
                  ['Age < 13: +2 points', 'Female sex: +1 point',
                   'Risser ≤ 2: +3 points', 'Cobb > 25°: +2 points',
                   'Total: 0-8 points'], model_color)

# 4. Progression Model
create_rounded_box(1, 4, 4.5, 2.5, 'PROGRESSION MODEL',
                  ['Base rate: 0.5°/month',
                   'Risk multiplier: 0.3 × risk_score',
                   'Random variation: N(0, 0.2)',
                   'Formula: Δ = (0.5 + 0.3×risk) × time + ε',
                   'Non-negative constraint applied'], model_color)

# 5. Validation Process
create_rounded_box(6.5, 4, 4, 2.5, 'VALIDATION PROCESS',
                  ['Statistical tests vs. literature',
                   'Distribution matching (KS test)',
                   'Correlation analysis',
                   'Clinical plausibility check',
                   'Expert review'], process_color)

# 6. Final Dataset
create_rounded_box(12, 4, 3, 2.5, 'FINAL DATASET',
                  ['1,000 patients', 'Baseline + 3 timepoints',
                   '70% Training', '15% Validation',
                   '15% Test'], output_color)

# 7. Mathematical Formulation
create_rounded_box(2, 0.5, 12, 2.5, 'MATHEMATICAL FORMULATION',
                  ['Patient i at time t: Cobb_i(t) = Cobb_i(0) + Progression_rate_i × t + ε_i(t)',
                   'Progression_rate_i = 0.5 + 0.3 × Risk_score_i + η_i, where η_i ~ N(0, 0.2)',
                   'Risk_score_i = 2×I(Age_i < 13) + 1×I(Sex_i = Female) + 3×I(Risser_i ≤ 2) + 2×I(Cobb_i(0) > 25)',
                   'ε_i(t) ~ N(0, σ_t²), where σ_6m = 1°, σ_12m = 1.5°, σ_24m = 2°',
                   'Constraint: Cobb_i(t) ≥ Cobb_i(0) (no improvement without intervention)'], 
                  '#F5F5F5')

# Create arrows
create_arrow(4, 8.2, 5.5, 8.2, 'Extract')
create_arrow(10.5, 8.2, 12, 8.2, 'Define')
create_arrow(8, 7, 8, 6.5, 'Apply')
create_arrow(5.5, 5.2, 6.5, 5.2, 'Validate')
create_arrow(10.5, 5.2, 12, 5.2, 'Generate')
create_arrow(8, 4, 8, 3, 'Implement')

# Add legend
legend_elements = [
    mpatches.Patch(color=input_color, label='Input Sources'),
    mpatches.Patch(color=process_color, label='Processing Steps'),
    mpatches.Patch(color=model_color, label='Statistical Models'),
    mpatches.Patch(color=output_color, label='Final Output')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('/home/ubuntu/scoliosis_prediction/high_res_figures/statistical_model_explanation.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Statistical model explanation figure created successfully!")

