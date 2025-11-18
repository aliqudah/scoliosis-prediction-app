"""
Synthetic Dataset Generator for Adolescent Idiopathic Scoliosis Curve Progression

This script generates a synthetic dataset that mimics the statistical properties
and relationships observed in published studies on AIS progression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_dataset(n_samples=1000):
    """
    Generate a synthetic dataset of AIS patients with initial features and progression outcomes.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic patients to generate
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing synthetic patient data
    """
    # Generate demographic features
    # Age distribution: 10-16 years (peak onset during growth spurt)
    age = np.random.normal(13, 1.5, n_samples)
    age = np.clip(age, 10, 16)
    
    # Sex distribution: ~80% female (higher prevalence in females)
    sex = np.random.choice(['F', 'M'], size=n_samples, p=[0.8, 0.2])
    
    # Risser sign (0-5, indicating skeletal maturity)
    risser = np.random.choice(range(6), size=n_samples)
    
    # Menarchal status (for females only)
    menarchal_status = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if sex[i] == 'F':
            # Probability of having reached menarche increases with age
            prob_menarche = max(0, min(1, (age[i] - 10) / 5))
            menarchal_status[i] = np.random.choice([0, 1], p=[1-prob_menarche, prob_menarche])
    
    # Height and weight (age-appropriate)
    # Simplified model based on CDC growth charts
    height_mean_f = 152 + (age - 10) * 6  # cm
    height_mean_m = 154 + (age - 10) * 7  # cm
    
    height = np.zeros(n_samples)
    weight = np.zeros(n_samples)
    
    for i in range(n_samples):
        if sex[i] == 'F':
            height[i] = np.random.normal(height_mean_f[i], 6)
            # Weight based on height with some variation
            weight[i] = 0.5 * (height[i] - 100) + np.random.normal(0, 5)
        else:
            height[i] = np.random.normal(height_mean_m[i], 7)
            weight[i] = 0.6 * (height[i] - 100) + np.random.normal(0, 6)
    
    # Ensure reasonable values
    height = np.clip(height, 120, 190)
    weight = np.clip(weight, 25, 90)
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Generate radiographic features
    
    # Initial Cobb angle (10-40 degrees, with most cases being mild)
    # Distribution skewed towards lower values
    initial_cobb = np.random.gamma(shape=2, scale=7, size=n_samples) + 10
    initial_cobb = np.clip(initial_cobb, 10, 40)
    
    # Curve pattern
    # 1: Thoracic, 2: Thoracolumbar, 3: Lumbar, 4: Double major
    curve_pattern_probs = [0.45, 0.25, 0.15, 0.15]  # Based on literature
    curve_pattern = np.random.choice([1, 2, 3, 4], size=n_samples, p=curve_pattern_probs)
    
    # Curve flexibility (percentage reduction on bending)
    # Higher flexibility in younger patients and lumbar curves
    flexibility_base = np.random.normal(30, 10, n_samples)
    
    # Adjust flexibility based on curve pattern (lumbar curves more flexible)
    flexibility_adjustment = np.zeros(n_samples)
    for i in range(n_samples):
        if curve_pattern[i] == 3:  # Lumbar
            flexibility_adjustment[i] = 10
        elif curve_pattern[i] == 2:  # Thoracolumbar
            flexibility_adjustment[i] = 5
        elif curve_pattern[i] == 4:  # Double major
            flexibility_adjustment[i] = -5
    
    # Adjust flexibility based on age (younger = more flexible)
    age_adjustment = (16 - age) * 2
    
    flexibility = flexibility_base + flexibility_adjustment + age_adjustment
    flexibility = np.clip(flexibility, 10, 70)
    
    # Thoracic kyphosis angle (20-40 degrees normally)
    thoracic_kyphosis = np.random.normal(30, 7, n_samples)
    
    # Lumbar lordosis angle (40-60 degrees normally)
    lumbar_lordosis = np.random.normal(50, 7, n_samples)
    
    # Number of vertebral levels involved (typically 5-9)
    levels_involved = np.random.randint(5, 10, n_samples)
    
    # Generate progression outcomes
    
    # Factors that increase progression risk:
    # - Young age
    # - Female sex
    # - Premenarchal status
    # - Low Risser sign
    # - High initial Cobb angle
    # - Thoracic curve
    # - Low flexibility
    # - Double major curves
    
    # Calculate progression risk score
    progression_risk = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Age factor (younger = higher risk)
        age_factor = max(0, (16 - age[i]) / 6) * 3
        
        # Sex factor
        sex_factor = 2 if sex[i] == 'F' else 0
        
        # Menarchal status factor
        menarchal_factor = 2 if sex[i] == 'F' and menarchal_status[i] == 0 else 0
        
        # Risser sign factor
        risser_factor = max(0, (5 - risser[i]) / 5) * 3
        
        # Initial Cobb angle factor
        cobb_factor = (initial_cobb[i] - 10) / 30 * 4
        
        # Curve pattern factor
        pattern_factor = 0
        if curve_pattern[i] == 1:  # Thoracic
            pattern_factor = 2
        elif curve_pattern[i] == 4:  # Double major
            pattern_factor = 1.5
            
        # Flexibility factor (less flexible = higher risk)
        flexibility_factor = max(0, (70 - flexibility[i]) / 60) * 2
        
        # Sum all factors with some random variation
        progression_risk[i] = (
            age_factor + sex_factor + menarchal_factor + risser_factor + 
            cobb_factor + pattern_factor + flexibility_factor + 
            np.random.normal(0, 1)  # Add some noise
        )
    
    # Normalize risk score to 0-10 range
    progression_risk = (progression_risk - progression_risk.min()) / (progression_risk.max() - progression_risk.min()) * 10
    
    # Calculate future Cobb angles at 6, 12, and 24 months
    # Higher risk scores lead to more progression
    
    cobb_6m = initial_cobb + progression_risk * np.random.uniform(0.5, 1.5, n_samples)
    cobb_12m = cobb_6m + progression_risk * np.random.uniform(0.3, 1.2, n_samples)
    cobb_24m = cobb_12m + progression_risk * np.random.uniform(0.2, 1.0, n_samples)
    
    # Add some random variation
    cobb_6m += np.random.normal(0, 2, n_samples)
    cobb_12m += np.random.normal(0, 3, n_samples)
    cobb_24m += np.random.normal(0, 4, n_samples)
    
    # Ensure Cobb angles don't decrease over time (progression is non-negative)
    for i in range(n_samples):
        cobb_6m[i] = max(cobb_6m[i], initial_cobb[i])
        cobb_12m[i] = max(cobb_12m[i], cobb_6m[i])
        cobb_24m[i] = max(cobb_24m[i], cobb_12m[i])
    
    # Create binary progression indicators
    # Progression defined as increase of 5+ degrees
    progression_6m = (cobb_6m - initial_cobb >= 5).astype(int)
    progression_12m = (cobb_12m - initial_cobb >= 5).astype(int)
    progression_24m = (cobb_24m - initial_cobb >= 5).astype(int)
    
    # Progression to bracing threshold (25+ degrees)
    bracing_threshold_6m = (cobb_6m >= 25).astype(int)
    bracing_threshold_12m = (cobb_12m >= 25).astype(int)
    bracing_threshold_24m = (cobb_24m >= 25).astype(int)
    
    # Progression to surgical threshold (45+ degrees)
    surgical_threshold_6m = (cobb_6m >= 45).astype(int)
    surgical_threshold_12m = (cobb_12m >= 45).astype(int)
    surgical_threshold_24m = (cobb_24m >= 45).astype(int)
    
    # Create DataFrame
    data = {
        # Demographic features
        'age': age,
        'sex': sex,
        'risser_sign': risser,
        'menarchal_status': menarchal_status,
        'height_cm': height,
        'weight_kg': weight,
        'bmi': bmi,
        
        # Radiographic features
        'initial_cobb_angle': initial_cobb,
        'curve_pattern': curve_pattern,
        'curve_flexibility': flexibility,
        'thoracic_kyphosis': thoracic_kyphosis,
        'lumbar_lordosis': lumbar_lordosis,
        'levels_involved': levels_involved,
        
        # Progression risk
        'progression_risk_score': progression_risk,
        
        # Outcome variables
        'cobb_angle_6m': cobb_6m,
        'cobb_angle_12m': cobb_12m,
        'cobb_angle_24m': cobb_24m,
        'progression_6m': progression_6m,
        'progression_12m': progression_12m,
        'progression_24m': progression_24m,
        'bracing_threshold_6m': bracing_threshold_6m,
        'bracing_threshold_12m': bracing_threshold_12m,
        'bracing_threshold_24m': bracing_threshold_24m,
        'surgical_threshold_6m': surgical_threshold_6m,
        'surgical_threshold_12m': surgical_threshold_12m,
        'surgical_threshold_24m': surgical_threshold_24m
    }
    
    df = pd.DataFrame(data)
    
    # Convert categorical variables
    df['sex'] = df['sex'].astype('category')
    df['curve_pattern'] = df['curve_pattern'].astype('category')
    
    return df

def plot_dataset_distributions(df, output_dir):
    """
    Plot distributions of key variables in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing synthetic patient data
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot initial Cobb angle distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['initial_cobb_angle'], bins=30, alpha=0.7)
    plt.title('Distribution of Initial Cobb Angles')
    plt.xlabel('Cobb Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'initial_cobb_distribution.png'), dpi=300)
    plt.close()
    
    # Plot progression over time
    plt.figure(figsize=(10, 6))
    time_points = ['initial_cobb_angle', 'cobb_angle_6m', 'cobb_angle_12m', 'cobb_angle_24m']
    labels = ['Initial', '6 months', '12 months', '24 months']
    
    # Random sample of 50 patients for visualization
    sample_indices = np.random.choice(len(df), size=50, replace=False)
    sample_df = df.iloc[sample_indices]
    
    for i in sample_indices:
        plt.plot([0, 6, 12, 24], 
                 [df.loc[i, 'initial_cobb_angle'], 
                  df.loc[i, 'cobb_angle_6m'], 
                  df.loc[i, 'cobb_angle_12m'], 
                  df.loc[i, 'cobb_angle_24m']], 
                 'o-', alpha=0.3)
    
    # Plot mean progression
    plt.plot([0, 6, 12, 24], 
             [df['initial_cobb_angle'].mean(), 
              df['cobb_angle_6m'].mean(), 
              df['cobb_angle_12m'].mean(), 
              df['cobb_angle_24m'].mean()], 
             'o-', color='red', linewidth=3, label='Mean progression')
    
    plt.title('Cobb Angle Progression Over Time')
    plt.xlabel('Time (months)')
    plt.ylabel('Cobb Angle (degrees)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'progression_over_time.png'), dpi=300)
    plt.close()
    
    # Plot progression risk vs actual progression
    plt.figure(figsize=(10, 6))
    plt.scatter(df['progression_risk_score'], 
                df['cobb_angle_24m'] - df['initial_cobb_angle'],
                alpha=0.5)
    plt.title('Progression Risk Score vs Actual Progression')
    plt.xlabel('Progression Risk Score')
    plt.ylabel('Actual Progression (24 months)')
    plt.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['progression_risk_score'], 
                   df['cobb_angle_24m'] - df['initial_cobb_angle'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(df['progression_risk_score']), 
             p(np.sort(df['progression_risk_score'])), 
             "r--", linewidth=2)
    
    plt.savefig(os.path.join(output_dir, 'risk_vs_progression.png'), dpi=300)
    plt.close()
    
    # Plot age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=20, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=300)
    plt.close()
    
    # Plot sex distribution
    plt.figure(figsize=(8, 6))
    df['sex'].value_counts().plot(kind='bar')
    plt.title('Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sex_distribution.png'), dpi=300)
    plt.close()
    
    # Plot curve pattern distribution
    plt.figure(figsize=(10, 6))
    pattern_counts = df['curve_pattern'].value_counts().sort_index()
    pattern_labels = ['Thoracic', 'Thoracolumbar', 'Lumbar', 'Double major']
    plt.bar(pattern_labels, pattern_counts)
    plt.title('Curve Pattern Distribution')
    plt.xlabel('Curve Pattern')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'curve_pattern_distribution.png'), dpi=300)
    plt.close()
    
    # Plot progression rates
    plt.figure(figsize=(10, 6))
    progression_rates = [
        df['progression_6m'].mean() * 100,
        df['progression_12m'].mean() * 100,
        df['progression_24m'].mean() * 100
    ]
    plt.bar(['6 months', '12 months', '24 months'], progression_rates)
    plt.title('Progression Rates (≥5° increase)')
    plt.xlabel('Time Point')
    plt.ylabel('Percentage of Patients (%)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'progression_rates.png'), dpi=300)
    plt.close()

def split_and_save_dataset(df, output_dir):
    """
    Split the dataset into training, validation, and test sets and save them.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing synthetic patient data
    output_dir : str
        Directory to save the datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into train (70%), validation (15%), and test (15%) sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save datasets
    train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
    
    print(f"Dataset split and saved:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=1000)
    
    # Plot distributions
    print("Plotting dataset distributions...")
    plot_dataset_distributions(df, output_dir='../figures')
    
    # Split and save dataset
    print("Splitting and saving dataset...")
    train_df, val_df, test_df = split_and_save_dataset(df, output_dir='../data')
    
    print("Done!")

