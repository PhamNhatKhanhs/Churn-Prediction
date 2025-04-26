# src/models/evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict # Added Dict typing
import sys
import mlflow # Import MLflow

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import save_metrics, save_report # Import utility functions

# Ensure the function name is evaluate_model_and_log
def evaluate_model_and_log(model: Any, scaler: StandardScaler, X_test: pd.DataFrame, y_test: pd.Series, config: dict) -> Dict[str, float]:
    """
    Evaluates the model on the test set, saves metrics/reports locally,
    AND logs metrics to MLflow.

    Args:
        model: The trained machine learning model.
        scaler: The fitted StandardScaler object.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        config: Configuration dictionary.

    Returns:
        A dictionary containing the calculated evaluation metrics.
    """
    print("Starting model evaluation...")

    # --- Scale Test Data ---
    numerical_cols = config['features']['numerical_cols_to_scale']
    X_test_scaled = X_test.copy()
    # Ensure only existing columns are scaled
    cols_to_scale_in_test = [col for col in numerical_cols if col in X_test_scaled.columns]
    if cols_to_scale_in_test:
         X_test_scaled[cols_to_scale_in_test] = scaler.transform(X_test_scaled[cols_to_scale_in_test]) # Use the fitted scaler
         print(f"Test data scaled for columns: {cols_to_scale_in_test}.")
    else:
         print("No numerical columns specified in config found in test data to scale.")


    # --- Predictions ---
    print("Making predictions on test data...")
    y_pred = model.predict(X_test_scaled)
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1 (Churn)
    except AttributeError:
        print("Warning: Model does not support predict_proba. ROC AUC cannot be calculated.")
        y_proba = None # Handle models without predict_proba

    print("Predictions made.")

    # --- Calculate Metrics ---
    print("Calculating evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0) # Handle zero division
    recall = recall_score(y_test, y_pred, zero_division=0)      # Handle zero division
    f1 = f1_score(y_test, y_pred, zero_division=0)              # Handle zero division
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc if roc_auc is not None else 'N/A' # Store N/A if not applicable
    }
    print("Calculated evaluation metrics:")
    for key, value in metrics.items():
        # Format differently if value is 'N/A'
        if isinstance(value, (int, float)):
             print(f"  {key}: {value:.4f}")
        else:
             print(f"  {key}: {value}")


    # --- Log metrics to MLflow ---
    # This relies on being called within an active MLflow run started in main.py
    try:
        # Log only numeric metrics
        mlflow.log_metric("test_accuracy", metrics['accuracy'])
        mlflow.log_metric("test_precision", metrics['precision'])
        mlflow.log_metric("test_recall", metrics['recall'])
        mlflow.log_metric("test_f1_score", metrics['f1_score'])
        if metrics['roc_auc'] != 'N/A':
            mlflow.log_metric("test_roc_auc", metrics['roc_auc'])
        print("Metrics logged to MLflow.")
    except Exception as mlflow_error:
        # Might fail if not called within mlflow.start_run() context
        print(f"Warning: Could not log metrics to MLflow. Ensure this is run within 'mlflow.start_run()'. Error: {mlflow_error}")
    # -----------------------------

    # --- Generate Reports ---
    print("\nGenerating classification report...")
    report = classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'], zero_division=0)
    print("Classification Report generated.")
    print(report)

    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix generated.")
    print(cm)

    # --- Save Artifacts Locally ---
    print("\nSaving evaluation artifacts locally...")
    metrics_save_path = Path(config['evaluation']['metrics_save_path'])
    report_save_path = Path(config['evaluation']['report_save_path'])
    cm_save_path = Path(config['evaluation']['confusion_matrix_save_path'])
    feature_importance_save_path = Path(config['evaluation'].get('feature_importance_save_path', '')) # Use .get for safety

    save_metrics(metrics, metrics_save_path)
    save_report(report, report_save_path)

    # Save Confusion Matrix plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        # Ensure directory exists
        cm_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_cm.savefig(cm_save_path, bbox_inches='tight')
        plt.close(fig_cm) # Close plot to avoid displaying it if running script
        print(f"Confusion matrix plot saved to {cm_save_path}")
    except Exception as plot_error:
        print(f"Error saving confusion matrix plot: {plot_error}")


    # Save Feature Importance plot (if applicable and path provided)
    if feature_importance_save_path and hasattr(model, 'feature_importances_'):
        print("Generating and saving feature importance plot...")
        try:
            importances = model.feature_importances_
            feature_names = X_test.columns # Assuming X_test has original feature names
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
            # Use hue=y for newer seaborn versions if needed, but direct assignment works here
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), ax=ax_fi, palette='viridis') # Top 20
            ax_fi.set_title('Top 20 Feature Importances')
            # Ensure directory exists
            feature_importance_save_path.parent.mkdir(parents=True, exist_ok=True)
            fig_fi.savefig(feature_importance_save_path, bbox_inches='tight')
            plt.close(fig_fi)
            print(f"Feature importance plot saved to {feature_importance_save_path}")
        except Exception as fi_plot_error:
            print(f"Error saving feature importance plot: {fi_plot_error}")
    elif not feature_importance_save_path:
         print("Feature importance save path not specified in config, skipping plot.")
    elif not hasattr(model, 'feature_importances_'):
         print("Model does not have feature_importances_ attribute, skipping plot.")


    print("Evaluation artifacts saved locally.")
    return metrics # Return metrics dictionary
