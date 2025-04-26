# src/models/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import make_scorer, recall_score # Import scorers
from pathlib import Path
from typing import Tuple, Any
import sys
import time # Import time for tracking duration

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
    # Returning test set as well because main.py loads it here for evaluation later
    return X_train, y_train, X_test, y_test

# Function with Hyperparameter Tuning using GridSearchCV
def train_model_with_tuning(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> Tuple[Any, StandardScaler, dict, float]:
    """
    Scales data, tunes hyperparameters using GridSearchCV for the specified model,
    trains the best model on the full training data, and saves the best model and scaler.

    Args:
        X_train: Training features.
        y_train: Training target.
        config: Configuration dictionary.

    Returns:
        A tuple containing:
            - best_model: The best estimator found by GridSearchCV, refit on the whole train set.
            - scaler: The fitted StandardScaler object.
            - best_params: The dictionary of best hyperparameters found.
            - best_score: The best cross-validation score achieved (based on the specified scoring metric).
    """
    print("Starting model training and hyperparameter tuning process...")
    start_time = time.time()

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

    # --- Hyperparameter Tuning Setup ---
    model_name = config['model_selection']['name']
    random_state = config['random_state']

    # Define parameter grid (Example for RandomForestClassifier)
    # You can move this grid definition to config.yaml for more flexibility
    if model_name == "RandomForestClassifier":
        print("Setting up parameter grid for RandomForestClassifier...")
        param_grid = {
            'n_estimators': [100, 150],             # Number of trees
            'max_depth': [10, 20, None],            # Max depth of trees (None means unlimited)
            'min_samples_split': [2, 5, 10],        # Min samples to split a node
            'min_samples_leaf': [1, 3, 5],          # Min samples in a leaf node
            'class_weight': ['balanced', None]      # Option to handle imbalance
            # Add other parameters like 'criterion': ['gini', 'entropy'] if needed
        }
        estimator = RandomForestClassifier(random_state=random_state)

    elif model_name == "LogisticRegression":
        print("Setting up parameter grid for LogisticRegression...")
        param_grid = {
            'C': [0.1, 1.0, 10.0],                  # Inverse of regularization strength
            'solver': ['liblinear', 'saga'],        # Solvers that support L1/L2
            'penalty': ['l1', 'l2'],                # Regularization type
            'class_weight': ['balanced', None]      # Option to handle imbalance
        }
        # Increase max_iter if solver='saga' might need more iterations
        estimator = LogisticRegression(random_state=random_state, max_iter=config['model_selection']['LogisticRegression']['params'].get('max_iter', 1000))

    else:
        raise ValueError(f"Unsupported model name for tuning: {model_name}")

    # Define the scoring metric for GridSearchCV
    # Prioritizing Recall for the positive class (Churn=1)
    # Other options: 'accuracy', 'precision', 'f1', 'roc_auc'
    # Note: Use pos_label=1 because Churn 'Yes' was mapped to 1
    scoring_metric = make_scorer(recall_score, pos_label=1)
    scoring_name = 'recall_pos_label_1' # Name for logging
    print(f"Using scoring metric for GridSearchCV: {scoring_name}")

    # Initialize GridSearchCV
    # cv=5 means 5-fold cross-validation
    # n_jobs=-1 uses all available CPU cores
    # verbose=2 provides more detailed output during tuning
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=5,
        n_jobs=-1,
        verbose=2 # Increase verbosity
    )

    print("Starting GridSearchCV... This might take a while.")
    grid_search_start_time = time.time()
    # Fit GridSearchCV on the scaled training data
    grid_search.fit(X_train_scaled, y_train)
    grid_search_end_time = time.time()
    print(f"GridSearchCV finished in {grid_search_end_time - grid_search_start_time:.2f} seconds.")

    # Get the best results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_ # Mean cross-validated score of the best_estimator
    best_model = grid_search.best_estimator_ # Estimator that was chosen by the search, fitted on the whole dataset

    print(f"\nBest parameters found by GridSearchCV:")
    print(best_params)
    print(f"Best cross-validation score ({scoring_name}): {best_score:.4f}")

    # --- Save Artifacts ---
    model_save_path = Path(config['artifacts']['model_save_path'])
    scaler_save_path = Path(config['artifacts']['scaler_save_path'])

    print(f"\nSaving best model to {model_save_path}")
    save_joblib(best_model, model_save_path) # Save the best model found
    print(f"Saving scaler to {scaler_save_path}")
    save_joblib(scaler, scaler_save_path)

    end_time = time.time()
    print(f"Training and tuning process completed in {end_time - start_time:.2f} seconds.")

    # Return the necessary objects for evaluation and logging
    return best_model, scaler, best_params, best_score

# Keep the original train_model function for compatibility or remove if not needed
# def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> Tuple[Any, StandardScaler]:
#     """ (Original function without tuning) """
#     # ... (original code) ...
#     pass
