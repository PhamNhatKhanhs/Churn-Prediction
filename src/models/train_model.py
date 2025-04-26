# src/models/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import Tuple, Any
import sys

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import load_joblib, save_joblib # Import utility functions

def load_processed_data(config: dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Loads processed train and test data."""
    train_path = Path(config['data']['processed_train_path'])
    test_path = Path(config['data']['processed_test_path'])

    print(f"Loading processed data from {train_path} and {test_path}")
    train_data = load_joblib(train_path)
    test_data = load_joblib(test_path)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print("Processed data loaded successfully.")
    return X_train, y_train, X_test, y_test # Returning test set as well for potential immediate evaluation

def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> Tuple[Any, StandardScaler]:
    """Scales data, trains the specified model, and saves model and scaler."""
    print("Starting model training process...")

    # --- Scaling ---
    scaler = StandardScaler()
    numerical_cols = config['features']['numerical_cols_to_scale']
    print(f"Scaling numerical features: {numerical_cols}")

    # Fit scaler ONLY on training data numerical columns
    scaler.fit(X_train[numerical_cols])

    # Transform training data numerical columns
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
    print("Training data scaled.")

    # --- Model Selection and Training ---
    model_name = config['model_selection']['name']
    model_params = config['model_selection'][model_name]['params']
    random_state = config['random_state']

    print(f"Training model: {model_name}")
    if model_name == "LogisticRegression":
        model = LogisticRegression(random_state=random_state, **model_params)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- Save Artifacts ---
    model_save_path = Path(config['artifacts']['model_save_path'])
    scaler_save_path = Path(config['artifacts']['scaler_save_path'])

    save_joblib(model, model_save_path)
    save_joblib(scaler, scaler_save_path)

    return model, scaler