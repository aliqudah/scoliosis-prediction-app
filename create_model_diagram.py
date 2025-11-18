import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as PathEffects

# Set up the figure with high resolution
plt.figure(figsize=(16, 10), dpi=300)
ax = plt.gca()
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
input_color = '#4A9BE8'  # blue
numerical_color = '#57C278'  # green
categorical_color = '#57C278'  # green
fusion_color = '#9575CD'  # purple
temporal_color = '#9575CD'  # purple
output_color = '#FFB74D'  # orange

# Title
plt.text(8, 9.5, 'DETAILED DEEP LEARNING MODEL FOR SCOLIOSIS CURVE PROGRESSION PREDICTION', 
         fontsize=20, fontweight='bold', ha='center')

# Function to create a box with title and content
def create_box(x, y, width, height, title, title_color, content_lines, box_alpha=0.2):
    # Main box
    rect = Rectangle((x, y), width, height, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Title box
    title_height = 0.8
    title_rect = Rectangle((x, y+height-title_height), width, title_height, 
                          facecolor=title_color, edgecolor='black', alpha=1, linewidth=2)
    ax.add_patch(title_rect)
    
    # Title text
    plt.text(x + width/2, y + height - title_height/2, title, 
             fontsize=16, fontweight='bold', ha='center', va='center', color='white')
    
    # Content text
    line_height = (height - title_height) / (len(content_lines) + 1)
    for i, line in enumerate(content_lines):
        plt.text(x + width/2, y + (height - title_height) - (i+1) * line_height, 
                 line, fontsize=12, ha='center', va='center')

# Function to create an arrow
def create_arrow(start_x, start_y, end_x, end_y):
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                           arrowstyle='->', linewidth=2, color='black',
                           connectionstyle='arc3,rad=0')
    ax.add_patch(arrow)

# INPUT SECTION
input_x, input_y, input_width, input_height = 1, 2, 2.5, 6
create_box(input_x, input_y, input_width, input_height, 'INPUT', input_color, [])

# Clinical Features
clinical_x, clinical_y, clinical_width, clinical_height = input_x + 0.25, input_y + 3, input_width - 0.5, 2.5
clinical_rect = Rectangle((clinical_x, clinical_y), clinical_width, clinical_height, 
                         facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(clinical_rect)
plt.text(clinical_x + clinical_width/2, clinical_y + clinical_height - 0.3, 'Clinical Features', 
         fontsize=14, fontweight='bold', ha='center', va='center')

clinical_features = ['Age (numeric)', 'Sex (categorical)', 'Risser sign (0-5)', 'BMI (numeric)']
for i, feature in enumerate(clinical_features):
    plt.text(clinical_x + clinical_width/2, clinical_y + clinical_height - 0.7 - 0.4*i, 
             feature, fontsize=11, ha='center', va='center')

# Radiographic Features
radio_x, radio_y, radio_width, radio_height = input_x + 0.25, input_y + 0.25, input_width - 0.5, 2.5
radio_rect = Rectangle((radio_x, radio_y), radio_width, radio_height, 
                      facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(radio_rect)
plt.text(radio_x + radio_width/2, radio_y + radio_height - 0.3, 'Radiographic Features', 
         fontsize=14, fontweight='bold', ha='center', va='center')

radio_features = ['Initial Cobb angle (°)', 'Curve pattern (categorical)', 'Curve flexibility (%)', 'Vertebral rotation (°)']
for i, feature in enumerate(radio_features):
    plt.text(radio_x + radio_width/2, radio_y + radio_height - 0.7 - 0.4*i, 
             feature, fontsize=11, ha='center', va='center')

# FEATURE EXTRACTION SECTION
extract_x, extract_y, extract_width, extract_height = 5, 2, 3, 6
create_box(extract_x, extract_y, extract_width, extract_height, 'FEATURE EXTRACTION', numerical_color, [])

# Numerical Processing
numerical_x, numerical_y, numerical_width, numerical_height = extract_x + 0.25, extract_y + 3, extract_width - 0.5, 2.5
numerical_rect = Rectangle((numerical_x, numerical_y), numerical_width, numerical_height, 
                          facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(numerical_rect)
plt.text(numerical_x + numerical_width/2, numerical_y + numerical_height - 0.3, 'Numerical Processing', 
         fontsize=14, fontweight='bold', ha='center', va='center')

numerical_layers = [
    'Normalization',
    'Dense (64 units, ReLU)',
    'Batch Normalization',
    'Dropout (0.3)',
    'Dense (32 units, ReLU)'
]
for i, layer in enumerate(numerical_layers):
    plt.text(numerical_x + numerical_width/2, numerical_y + numerical_height - 0.7 - 0.35*i, 
             layer, fontsize=11, ha='center', va='center')

# Categorical Processing
categorical_x, categorical_y, categorical_width, categorical_height = extract_x + 0.25, extract_y + 0.25, extract_width - 0.5, 2.5
categorical_rect = Rectangle((categorical_x, categorical_y), categorical_width, categorical_height, 
                            facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(categorical_rect)
plt.text(categorical_x + categorical_width/2, categorical_y + categorical_height - 0.3, 'Categorical Processing', 
         fontsize=14, fontweight='bold', ha='center', va='center')

categorical_layers = [
    'One-hot encoding',
    'Embedding (16 dim)',
    'Flatten',
    'Dense (16 units, ReLU)',
    'Batch Normalization'
]
for i, layer in enumerate(categorical_layers):
    plt.text(categorical_x + categorical_width/2, categorical_y + categorical_height - 0.7 - 0.35*i, 
             layer, fontsize=11, ha='center', va='center')

# MODEL CORE SECTION
model_x, model_y, model_width, model_height = 9.5, 2, 3, 6
create_box(model_x, model_y, model_width, model_height, 'MODEL CORE', fusion_color, [])

# Feature Fusion
fusion_x, fusion_y, fusion_width, fusion_height = model_x + 0.25, model_y + 3, model_width - 0.5, 2.5
fusion_rect = Rectangle((fusion_x, fusion_y), fusion_width, fusion_height, 
                       facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(fusion_rect)
plt.text(fusion_x + fusion_width/2, fusion_y + fusion_height - 0.3, 'Feature Fusion', 
         fontsize=14, fontweight='bold', ha='center', va='center')

fusion_layers = [
    'Concatenation',
    'Dense (64 units, ReLU)',
    'Batch Normalization',
    'Dropout (0.3)',
    'Reshape for sequence'
]
for i, layer in enumerate(fusion_layers):
    plt.text(fusion_x + fusion_width/2, fusion_y + fusion_height - 0.7 - 0.35*i, 
             layer, fontsize=11, ha='center', va='center')

# Temporal Modeling
temporal_x, temporal_y, temporal_width, temporal_height = model_x + 0.25, model_y + 0.25, model_width - 0.5, 2.5
temporal_rect = Rectangle((temporal_x, temporal_y), temporal_width, temporal_height, 
                         facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(temporal_rect)
plt.text(temporal_x + temporal_width/2, temporal_y + temporal_height - 0.3, 'Temporal Modeling', 
         fontsize=14, fontweight='bold', ha='center', va='center')

temporal_layers = [
    'Bidirectional LSTM (32 units)',
    'Dropout (0.3)',
    'Time-distributed Dense',
    'Attention mechanism'
]
for i, layer in enumerate(temporal_layers):
    plt.text(temporal_x + temporal_width/2, temporal_y + temporal_height - 0.7 - 0.4*i, 
             layer, fontsize=11, ha='center', va='center')

# OUTPUT SECTION
output_x, output_y, output_width, output_height = 14, 2, 2.5, 6
create_box(output_x, output_y, output_width, output_height, 'OUTPUT', output_color, [])

# Regression Output
regression_x, regression_y, regression_width, regression_height = output_x + 0.25, output_y + 3, output_width - 0.5, 2.5
regression_rect = Rectangle((regression_x, regression_y), regression_width, regression_height, 
                           facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(regression_rect)
plt.text(regression_x + regression_width/2, regression_y + regression_height - 0.3, 'Regression Output', 
         fontsize=14, fontweight='bold', ha='center', va='center')

regression_details = [
    'Dense (16 units, ReLU)',
    'Dropout (0.2)',
    'Dense (3 units, Linear)',
    'Future Cobb angles',
    '(6, 12, 24 months)'
]
for i, detail in enumerate(regression_details):
    plt.text(regression_x + regression_width/2, regression_y + regression_height - 0.7 - 0.35*i, 
             detail, fontsize=11, ha='center', va='center')

# Classification Output
classification_x, classification_y, classification_width, classification_height = output_x + 0.25, output_y + 0.25, output_width - 0.5, 2.5
classification_rect = Rectangle((classification_x, classification_y), classification_width, classification_height, 
                               facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(classification_rect)
plt.text(classification_x + classification_width/2, classification_y + classification_height - 0.3, 'Classification Output', 
         fontsize=14, fontweight='bold', ha='center', va='center')

classification_details = [
    'Dense (16 units, ReLU)',
    'Dropout (0.2)',
    'Dense (3 units, Sigmoid)',
    'Progression probabilities',
    '(6, 12, 24 months)'
]
for i, detail in enumerate(classification_details):
    plt.text(classification_x + classification_width/2, classification_y + classification_height - 0.7 - 0.35*i, 
             detail, fontsize=11, ha='center', va='center')

# Create arrows
# Input to Feature Extraction
create_arrow(input_x + input_width, input_y + 4.5, extract_x, input_y + 4.5)  # Clinical to Numerical
create_arrow(input_x + input_width, input_y + 1.5, extract_x, input_y + 1.5)  # Radio to Categorical

# Feature Extraction to Model Core
create_arrow(extract_x + extract_width, model_y + 4.5, model_x, model_y + 4.5)  # Numerical to Fusion
create_arrow(extract_x + extract_width, model_y + 1.5, model_x, model_y + 1.5)  # Categorical to Temporal

# Model Core to Output
create_arrow(model_x + model_width, model_y + 4.5, output_x, model_y + 4.5)  # Fusion to Regression
create_arrow(model_x + model_width, model_y + 1.5, output_x, model_y + 1.5)  # Temporal to Classification

# Add tensor shapes at key points
def add_tensor_shape(x, y, shape_text, direction='above'):
    offset = 0.3 if direction == 'above' else -0.3
    text = plt.text(x, y + offset, shape_text, fontsize=9, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.2'))
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# Input tensor shapes
add_tensor_shape(input_x + input_width/2, input_y + 5.5, '[batch_size, n_clinical]')
add_tensor_shape(input_x + input_width/2, input_y + 0.1, '[batch_size, n_radio]', 'below')

# Feature extraction output shapes
add_tensor_shape(extract_x + extract_width/2, extract_y + 5.5, '[batch_size, 32]')
add_tensor_shape(extract_x + extract_width/2, extract_y + 0.1, '[batch_size, 16]', 'below')

# Model core output shapes
add_tensor_shape(model_x + model_width/2, model_y + 5.5, '[batch_size, time_steps, 64]')
add_tensor_shape(model_x + model_width/2, model_y + 0.1, '[batch_size, 32]', 'below')

# Output shapes
add_tensor_shape(output_x + output_width/2, output_y + 5.5, '[batch_size, 3]')
add_tensor_shape(output_x + output_width/2, output_y + 0.1, '[batch_size, 3]', 'below')

# Save the figure
plt.tight_layout()
plt.savefig('/home/ubuntu/scoliosis_prediction/high_res_figures/detailed_model_architecture_programmatic.png', 
            dpi=300, bbox_inches='tight')

