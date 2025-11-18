"""
Deep Learning Model for Predicting Adolescent Idiopathic Scoliosis Curve Progression

This script implements a hybrid CNN-RNN model for predicting the progression of
spinal curvature in AIS patients based on initial radiographic and clinical features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, 
    Conv1D, MaxPooling1D, Flatten, LSTM, GRU,
    Concatenate, Embedding, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, roc_auc_score, 
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AISProgressionModel:
    """
    A class to build, train, and evaluate a deep learning model for
    predicting AIS curve progression.
    """
    
    def __init__(self, model_dir='../models'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save model checkpoints
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize preprocessing objects
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        
        # Initialize model
        self.model = None
    
    def preprocess_data(self, df, is_training=False):
        """
        Preprocess the data for model input.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing patient data
        is_training : bool
            Whether this is training data (to fit preprocessors)
            
        Returns:
        --------
        X : dict
            Dictionary of preprocessed input features
        y : dict
            Dictionary of target variables
        """
        # Separate numerical and categorical features
        numerical_features = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'initial_cobb_angle', 'curve_flexibility',
            'thoracic_kyphosis', 'lumbar_lordosis',
            'levels_involved', 'risser_sign'
        ]
        
        categorical_features = [
            'sex', 'curve_pattern', 'menarchal_status'
        ]
        
        # Target variables
        regression_targets = [
            'cobb_angle_6m', 'cobb_angle_12m', 'cobb_angle_24m'
        ]
        
        classification_targets = [
            'progression_6m', 'progression_12m', 'progression_24m',
            'bracing_threshold_6m', 'bracing_threshold_12m', 'bracing_threshold_24m',
            'surgical_threshold_6m', 'surgical_threshold_12m', 'surgical_threshold_24m'
        ]
        
        # Preprocess numerical features
        numerical_data = df[numerical_features].values
        if is_training:
            numerical_data_scaled = self.numerical_scaler.fit_transform(numerical_data)
        else:
            numerical_data_scaled = self.numerical_scaler.transform(numerical_data)
        
        # Preprocess categorical features
        categorical_data = {}
        for feature in categorical_features:
            if is_training:
                # Create and fit one-hot encoder for this feature
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[[feature]])
                self.categorical_encoders[feature] = encoder
            else:
                # Use pre-fitted encoder
                encoder = self.categorical_encoders[feature]
                encoded_data = encoder.transform(df[[feature]])
            
            categorical_data[feature] = encoded_data
        
        # Prepare input dictionary
        X = {
            'numerical_features': numerical_data_scaled
        }
        
        # Add categorical features to input dictionary
        for feature, encoded_data in categorical_data.items():
            X[f'categorical_{feature}'] = encoded_data
        
        # Prepare target dictionary
        y = {}
        
        # Regression targets
        for target in regression_targets:
            y[f'regression_{target}'] = df[target].values
        
        # Classification targets
        for target in classification_targets:
            y[f'classification_{target}'] = df[target].values
        
        return X, y
    
    def build_model(self, input_shapes, output_shapes):
        """
        Build the hybrid CNN-RNN model.
        
        Parameters:
        -----------
        input_shapes : dict
            Dictionary of input shapes
        output_shapes : dict
            Dictionary of output shapes
            
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled model
        """
        # Input layers
        numerical_input = Input(shape=(input_shapes['numerical_features'][1],), 
                               name='numerical_features')
        
        categorical_inputs = {}
        for name, shape in input_shapes.items():
            if name.startswith('categorical_'):
                categorical_inputs[name] = Input(shape=(shape[1],), name=name)
        
        # Process numerical features
        x_numerical = Dense(64, activation='relu')(numerical_input)
        x_numerical = BatchNormalization()(x_numerical)
        x_numerical = Dropout(0.3)(x_numerical)
        x_numerical = Dense(32, activation='relu')(x_numerical)
        
        # Process categorical features and concatenate
        categorical_outputs = []
        for name, input_layer in categorical_inputs.items():
            x_cat = Dense(16, activation='relu')(input_layer)
            x_cat = BatchNormalization()(x_cat)
            categorical_outputs.append(x_cat)
        
        # Concatenate all features
        if categorical_outputs:
            all_features = Concatenate()([x_numerical] + categorical_outputs)
        else:
            all_features = x_numerical
        
        # Shared layers for feature extraction
        shared_features = Dense(64, activation='relu')(all_features)
        shared_features = BatchNormalization()(shared_features)
        shared_features = Dropout(0.3)(shared_features)
        shared_features = Dense(32, activation='relu')(shared_features)
        
        # Reshape for sequence modeling
        # We'll treat the features as a sequence of length 1 for now
        # In a real model with time-series data, this would be longer
        sequence_features = tf.expand_dims(shared_features, axis=1)
        
        # Bidirectional LSTM for temporal modeling
        temporal_features = Bidirectional(LSTM(32, return_sequences=False))(sequence_features)
        temporal_features = Dropout(0.3)(temporal_features)
        
        # Output layers
        outputs = {}
        
        # Regression outputs (Cobb angles at different time points)
        regression_features = Dense(32, activation='relu')(temporal_features)
        
        for name, shape in output_shapes.items():
            if name.startswith('regression_'):
                outputs[name] = Dense(1, activation='linear', name=name)(regression_features)
        
        # Classification outputs (progression, bracing threshold, surgical threshold)
        classification_features = Dense(32, activation='relu')(temporal_features)
        
        for name, shape in output_shapes.items():
            if name.startswith('classification_'):
                outputs[name] = Dense(1, activation='sigmoid', name=name)(classification_features)
        
        # Create and compile model
        model = Model(
            inputs=[numerical_input] + list(categorical_inputs.values()),
            outputs=list(outputs.values())
        )
        
        # Define loss functions and metrics
        losses = {}
        metrics = {}
        
        for name in outputs.keys():
            if name.startswith('regression_'):
                losses[name] = 'mean_squared_error'
                metrics[name] = ['mae']
            else:
                losses[name] = 'binary_crossentropy'
                metrics[name] = ['accuracy', tf.keras.metrics.AUC()]
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=losses,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, epochs=100, batch_size=32):
        """
        Train the model.
        
        Parameters:
        -----------
        train_data : tuple
            Tuple of (X_train, y_train)
        val_data : tuple
            Tuple of (X_val, y_val)
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        test_data : tuple
            Tuple of (X_test, y_test)
            
        Returns:
        --------
        results : dict
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Organize predictions by output name
        pred_dict = {}
        for i, name in enumerate(self.model.output_names):
            pred_dict[name] = predictions[i]
        
        # Calculate metrics
        results = {}
        
        # Regression metrics
        for name, y_true in y_test.items():
            if name.startswith('regression_'):
                y_pred = pred_dict[name].flatten()
                
                # Mean Absolute Error
                mae = mean_absolute_error(y_true, y_pred)
                results[f'{name}_mae'] = mae
                
                # Root Mean Squared Error
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                results[f'{name}_rmse'] = rmse
        
        # Classification metrics
        for name, y_true in y_test.items():
            if name.startswith('classification_'):
                y_pred_prob = pred_dict[name].flatten()
                y_pred_class = (y_pred_prob >= 0.5).astype(int)
                
                # AUC
                auc = roc_auc_score(y_true, y_pred_prob)
                results[f'{name}_auc'] = auc
                
                # Accuracy
                accuracy = accuracy_score(y_true, y_pred_class)
                results[f'{name}_accuracy'] = accuracy
                
                # Sensitivity (Recall)
                sensitivity = recall_score(y_true, y_pred_class)
                results[f'{name}_sensitivity'] = sensitivity
                
                # Precision
                precision = precision_score(y_true, y_pred_class)
                results[f'{name}_precision'] = precision
                
                # Specificity
                tn = np.sum((y_true == 0) & (y_pred_class == 0))
                fp = np.sum((y_true == 0) & (y_pred_class == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                results[f'{name}_specificity'] = specificity
        
        return results
    
    def plot_training_history(self, history, output_dir='../figures'):
        """
        Plot training history.
        
        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        output_dir : str
            Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot overall loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
        plt.close()
        
        # Plot regression losses
        regression_losses = [key for key in history.history.keys() 
                            if key.startswith('regression_') and not key.startswith('val_')]
        
        if regression_losses:
            plt.figure(figsize=(12, 8))
            for loss in regression_losses:
                plt.plot(history.history[loss], label=loss)
                plt.plot(history.history[f'val_{loss}'], label=f'val_{loss}', linestyle='--')
            
            plt.title('Regression Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'regression_losses.png'), dpi=300)
            plt.close()
        
        # Plot classification losses
        classification_losses = [key for key in history.history.keys() 
                                if key.startswith('classification_') and not key.startswith('val_')]
        
        if classification_losses:
            plt.figure(figsize=(12, 8))
            for loss in classification_losses[:3]:  # Plot only first 3 to avoid overcrowding
                plt.plot(history.history[loss], label=loss)
                plt.plot(history.history[f'val_{loss}'], label=f'val_{loss}', linestyle='--')
            
            plt.title('Classification Losses (Sample)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'classification_losses.png'), dpi=300)
            plt.close()
    
    def plot_predictions(self, test_data, output_dir='../figures'):
        """
        Plot model predictions against actual values.
        
        Parameters:
        -----------
        test_data : tuple
            Tuple of (X_test, y_test)
        output_dir : str
            Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Organize predictions by output name
        pred_dict = {}
        for i, name in enumerate(self.model.output_names):
            pred_dict[name] = predictions[i]
        
        # Plot regression predictions
        regression_targets = [name for name in y_test.keys() if name.startswith('regression_')]
        
        for target in regression_targets:
            y_true = y_test[target]
            y_pred = pred_dict[target].flatten()
            
            plt.figure(figsize=(10, 10))
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add error margins
            plt.fill_between([min_val, max_val], 
                            [min_val - 5, max_val - 5], 
                            [min_val + 5, max_val + 5], 
                            color='gray', alpha=0.2)
            
            plt.title(f'Predicted vs Actual {target.replace("regression_cobb_angle_", "")}')
            plt.xlabel('Actual Cobb Angle (degrees)')
            plt.ylabel('Predicted Cobb Angle (degrees)')
            plt.grid(alpha=0.3)
            
            # Add MAE to plot
            mae = mean_absolute_error(y_true, y_pred)
            plt.text(min_val + 0.1 * (max_val - min_val), 
                    max_val - 0.1 * (max_val - min_val), 
                    f'MAE: {mae:.2f}°')
            
            plt.savefig(os.path.join(output_dir, f'{target}_predictions.png'), dpi=300)
            plt.close()
        
        # Plot ROC curves for classification targets
        classification_targets = [name for name in y_test.keys() if name.startswith('classification_')]
        
        # Group by time point
        time_points = ['6m', '12m', '24m']
        target_types = ['progression', 'bracing_threshold', 'surgical_threshold']
        
        for target_type in target_types:
            plt.figure(figsize=(10, 8))
            
            for time_point in time_points:
                target = f'classification_{target_type}_{time_point}'
                if target in y_test:
                    y_true = y_test[target]
                    y_pred = pred_dict[target].flatten()
                    
                    # Calculate ROC curve
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_pred)
                    
                    plt.plot(fpr, tpr, label=f'{time_point} (AUC = {auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {target_type.replace("_", " ").title()}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{target_type}_roc_curves.png'), dpi=300)
            plt.close()

def main():
    """
    Main function to run the model training and evaluation.
    """
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('../data/train_dataset.csv')
    val_df = pd.read_csv('../data/val_dataset.csv')
    test_df = pd.read_csv('../data/test_dataset.csv')
    
    # Initialize model
    print("Initializing model...")
    model = AISProgressionModel()
    
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
    model.plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate((X_test, y_test))
    
    # Print evaluation results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    print("Plotting predictions...")
    model.plot_predictions((X_test, y_test))
    
    print("Done!")

if __name__ == "__main__":
    main()

