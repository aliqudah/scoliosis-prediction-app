"""
Main script for the AIS Curve Progression Prediction project.

This script runs the entire pipeline:
1. Generate synthetic dataset
2. Train and evaluate the deep learning model
3. Generate visualizations and results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, roc_auc_score, 
    accuracy_score, precision_score, recall_score
)

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from generate_dataset import generate_synthetic_dataset, plot_dataset_distributions, split_and_save_dataset
from ais_progression_model import AISProgressionModel

def run_pipeline():
    """
    Run the entire pipeline from data generation to model evaluation.
    """
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Generate synthetic dataset
    print("\n=== Generating Synthetic Dataset ===")
    df = generate_synthetic_dataset(n_samples=1000)
    
    # Plot dataset distributions
    print("Plotting dataset distributions...")
    plot_dataset_distributions(df, output_dir='figures')
    
    # Split and save dataset
    print("Splitting and saving dataset...")
    train_df, val_df, test_df = split_and_save_dataset(df, output_dir='data')
    
    # Step 2: Train and evaluate the model
    print("\n=== Training and Evaluating Model ===")
    
    # Initialize model
    model = AISProgressionModel(model_dir='models')
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, y_train = model.preprocess_data(train_df, is_training=True)
    X_val, y_val = model.preprocess_data(val_df)
    X_test, y_test = model.preprocess_data(test_df)
    
    # Build model
    print("Building model...")
    input_shapes = {key: (1, val.shape[1]) if len(val.shape) == 2 else (1,) 
                   for key, val in X_train.items()}
    output_shapes = {key: (1,) for key in y_train.keys()}
    
    model.build_model(input_shapes, output_shapes)
    
    # Print model summary
    model.model.summary()
    
    # Train model
    print("Training model...")
    history = model.train(
        (X_train, y_train),
        (X_val, y_val),
        epochs=50,
        batch_size=32
    )
    
    # Plot training history
    print("Plotting training history...")
    model.plot_training_history(history, output_dir='figures')
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate((X_test, y_test))
    
    # Save evaluation results
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    results_df.to_csv('results/evaluation_results.csv', index=False)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    print("Plotting predictions...")
    model.plot_predictions((X_test, y_test), output_dir='figures')
    
    # Step 3: Generate additional visualizations
    print("\n=== Generating Additional Visualizations ===")
    
    # Feature importance analysis (simplified version)
    print("Analyzing feature importance...")
    analyze_feature_importance(model, X_test, y_test)
    
    print("\nPipeline completed successfully!")

def analyze_feature_importance(model, X_test, y_test, output_dir='figures'):
    """
    Perform a simplified feature importance analysis by perturbing input features.
    
    Parameters:
    -----------
    model : AISProgressionModel
        Trained model
    X_test : dict
        Test input features
    y_test : dict
        Test target variables
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get baseline predictions
    baseline_preds = model.model.predict(X_test)
    
    # Organize baseline predictions by output name
    baseline_pred_dict = {}
    for i, name in enumerate(model.model.output_names):
        baseline_pred_dict[name] = baseline_preds[i]
    
    # Select a regression target for analysis
    target = 'regression_cobb_angle_24m'
    baseline_target_preds = baseline_pred_dict[target].flatten()
    
    # Numerical features to analyze
    numerical_features = [
        'age', 'initial_cobb_angle', 'curve_flexibility',
        'thoracic_kyphosis', 'lumbar_lordosis',
        'levels_involved', 'risser_sign'
    ]
    
    # Store feature importance scores
    importance_scores = {}
    
    # Perturb each feature and measure the effect on predictions
    for feature_idx, feature_name in enumerate(numerical_features):
        # Create perturbed input
        X_perturbed = {k: v.copy() for k, v in X_test.items()}
        
        # Add random noise to the feature
        noise = np.random.normal(0, 0.5, X_perturbed['numerical_features'].shape[0])
        X_perturbed['numerical_features'][:, feature_idx] += noise
        
        # Get predictions with perturbed input
        perturbed_preds = model.model.predict(X_perturbed)
        
        # Organize perturbed predictions
        perturbed_pred_dict = {}
        for i, name in enumerate(model.model.output_names):
            perturbed_pred_dict[name] = perturbed_preds[i]
        
        # Calculate the mean absolute difference in predictions
        perturbed_target_preds = perturbed_pred_dict[target].flatten()
        importance = np.mean(np.abs(perturbed_target_preds - baseline_target_preds))
        
        importance_scores[feature_name] = importance
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # Sort by importance
    sorted_indices = np.argsort(scores)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    plt.barh(sorted_features, sorted_scores)
    plt.xlabel('Feature Importance (Mean Absolute Prediction Difference)')
    plt.title('Feature Importance Analysis')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # Save feature importance scores
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': scores
    })
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    importance_df.to_csv('results/feature_importance.csv', index=False)

if __name__ == "__main__":
    run_pipeline()

