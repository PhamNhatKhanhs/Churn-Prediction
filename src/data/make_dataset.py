# src/data/make_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
import sys

# Add src directory to sys.path to allow importing utils
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import save_joblib # Import utility function

def load_raw_data(config: dict) -> pd.DataFrame:
    """Loads raw data from the specified path in config."""
    raw_path = Path(config['data']['raw_path'])
    print(f"Loading raw data from: {raw_path}")
    try:
        df = pd.read_csv(raw_path)
        print("Raw data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_path}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles specific missing values for the Telco Churn dataset."""
    print("Handling missing values...")
    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Impute NaN in TotalCharges (assuming they are for tenure=0) with 0
    initial_nan_count = df['TotalCharges'].isnull().sum()
    if initial_nan_count > 0:
        df['TotalCharges'].fillna(0, inplace=True)
        print(f"Filled {initial_nan_count} NaN values in TotalCharges with 0.")
    # Check for other NaNs if necessary
    # ...
    return df

def encode_categorical(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encodes categorical features using One-Hot Encoding."""
    print("Encoding categorical features...")
    target = config['data']['target_column']
    categorical_cols = config['features']['categorical_cols_ohe']

    # Auto-detect if list is empty
    if not categorical_cols:
        categorical_cols = df.select_dtypes(include='object').drop(columns=[target], errors='ignore').columns.tolist()
        print(f"Auto-detected categorical columns for OHE: {categorical_cols}")

    # Encode target variable
    if target in df.columns:
        df[target] = df[target].replace({'No': 0, 'Yes': 1})
        print(f"Target variable '{target}' encoded to 0/1.")

    # One-Hot Encode features
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print("Applied One-Hot Encoding.")

    return df


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Applies preprocessing steps: dropping columns, handling missing values, encoding."""
    print("Starting data preprocessing...")
    # Drop specified columns
    cols_to_drop = config['features']['cols_to_drop']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped columns: {cols_to_drop}")

    df = handle_missing_values(df.copy())
    df = encode_categorical(df.copy(), config)

    print("Preprocessing finished.")
    return df

def split_data(df: pd.DataFrame, config: dict):
    """Splits data into train and test sets and saves them."""
    print("Splitting data into train and test sets...")
    target_column = config['data']['target_column']
    test_size = config['data']['test_size']
    random_state = config['random_state']
    train_path = Path(config['data']['processed_train_path'])
    test_path = Path(config['data']['processed_test_path'])

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Save train and test sets (saving as dict with features and target)
    train_data = {'X': X_train, 'y': y_train}
    test_data = {'X': X_test, 'y': y_test}

    save_joblib(train_data, train_path)
    save_joblib(test_data, test_path)

    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")