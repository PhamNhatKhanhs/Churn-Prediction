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
from typing import Any
import sys

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import save_metrics, save_report # Import utility functions

def evaluate_model(model: Any, scaler: StandardScaler, X_test: pd.DataFrame, y_test: pd.Series, config: dict):
    """Evaluates the model on the test set and saves metrics/reports."""
    print("Starting model evaluation...")

    # --- Scale Test Data ---
    numerical_cols = config['features']['numerical_cols_to_scale']
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols]) # Use the fitted scaler
    print("Test data scaled.")

    # --- Predictions ---
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1 (Churn)
    print("Predictions made on test data.")

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    print("Calculated evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # --- Generate Reports ---
    report = classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report generated.")
    print(report)
    print("\nConfusion Matrix generated.")
    print(cm)

    # --- Save Artifacts ---
    metrics_save_path = Path(config['evaluation']['metrics_save_path'])
    report_save_path = Path(config['evaluation']['report_save_path'])
    cm_save_path = Path(config['evaluation']['confusion_matrix_save_path'])
    feature_importance_save_path = Path(config['evaluation']['feature_importance_save_path'])

    save_metrics(metrics, metrics_save_path)
    save_report(report, report_save_path)

    # Save Confusion Matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    fig_cm.savefig(cm_save_path, bbox_inches='tight')
    plt.close(fig_cm) # Close plot to avoid displaying it if running script
    print(f"Confusion matrix plot saved to {cm_save_path}")

    # Save Feature Importance plot (if applicable)
    model_name = config['model_selection']['name']
    if hasattr(model, 'feature_importances_') and model_name != "LogisticRegression":
        importances = model.feature_importances_
        feature_names = X_test.columns # Assuming X_test has original feature names
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), ax=ax_fi, palette='viridis') # Top 20
        ax_fi.set_title('Top 20 Feature Importances')
        fig_fi.savefig(feature_importance_save_path, bbox_inches='tight')
        plt.close(fig_fi)
        print(f"Feature importance plot saved to {feature_importance_save_path}")

    print("Evaluation artifacts saved.")