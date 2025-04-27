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

    # Check if fillna is needed AFTER converting to numeric
    initial_nan_count = df['TotalCharges'].isnull().sum()
    if initial_nan_count > 0:
        # --- SỬA DÒNG NÀY ---
        # Gán trực tiếp thay vì dùng inplace=True
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        # --- KẾT THÚC SỬA ---
        print(f"Filled {initial_nan_count} NaN values in TotalCharges with 0.")
    else:
        print("No NaN found in TotalCharges after numeric conversion.")

    # Check for other NaNs if necessary
    # ...
    return df

def encode_categorical(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encodes categorical features using One-Hot Encoding and maps the target."""
    print("Encoding categorical features...")
    target = config['data']['target_column']
    categorical_cols = config['features']['categorical_cols_ohe']

    # Auto-detect if list is empty
    if not categorical_cols:
        categorical_cols = df.select_dtypes(include='object').drop(columns=[target], errors='ignore').columns.tolist()
        print(f"Auto-detected categorical columns for OHE: {categorical_cols}")

    # Encode target variable using .map()
    if target in df.columns:
        # --- SỬA DÒNG NÀY ---
        # Sử dụng .map() thay vì .replace() cho việc ánh xạ đơn giản
        target_map = {'No': 0, 'Yes': 1}
        df[target] = df[target].map(target_map)
        # --- KẾT THÚC SỬA ---
        print(f"Target variable '{target}' mapped to 0/1.")
        # Kiểm tra xem có giá trị nào không map được không (ví dụ: nếu có giá trị khác No/Yes)
        if df[target].isnull().any():
             print(f"Warning: Found NaN values in target column '{target}' after mapping. Check original values.")


    # One-Hot Encode features
    if categorical_cols:
        # Chỉ mã hóa các cột thực sự tồn tại trong DataFrame
        cols_to_encode_present = [col for col in categorical_cols if col in df.columns]
        if cols_to_encode_present:
            df = pd.get_dummies(df, columns=cols_to_encode_present, drop_first=True)
            print(f"Applied One-Hot Encoding to: {cols_to_encode_present}")
        else:
            print("No specified categorical columns found in the DataFrame to encode.")

    return df


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Applies preprocessing steps: dropping columns, handling missing values, encoding."""
    print("Starting data preprocessing...")
    # Drop specified columns
    cols_to_drop = config['features'].get('cols_to_drop', []) # Use .get for safety
    if cols_to_drop:
        # Only drop columns that actually exist
        cols_to_drop_present = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop_present:
            df = df.drop(columns=cols_to_drop_present)
            print(f"Dropped columns: {cols_to_drop_present}")
        else:
            print("No specified columns to drop were found in the DataFrame.")


    # Sử dụng copy để tránh SettingWithCopyWarning tiềm ẩn ở các bước sau
    df_copy = df.copy()
    df_copy = handle_missing_values(df_copy)
    df_copy = encode_categorical(df_copy, config)

    print("Preprocessing finished.")
    return df_copy

def split_data(df: pd.DataFrame, config: dict):
    """Splits data into train and test sets and saves them."""
    print("Splitting data into train and test sets...")
    target_column = config['data']['target_column']

    # Kiểm tra xem cột target có tồn tại không sau tiền xử lý
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the preprocessed DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError(f"Target column '{target_column}' missing after preprocessing.")

    test_size = config['data']['test_size']
    random_state = config['random_state']
    train_path = Path(config['data']['processed_train_path'])
    test_path = Path(config['data']['processed_test_path'])

    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Kiểm tra NaN trong target trước khi stratify
        if y.isnull().any():
            print(f"Warning: Target column '{target_column}' contains NaN values before splitting. Rows with NaN target will be dropped by train_test_split if stratify is used.")
            # Optionally handle NaNs here, e.g., df.dropna(subset=[target_column], inplace=True) before splitting X and y

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

    except KeyError as e:
        print(f"Error during splitting data: {e}. This might happen if the target column was dropped unexpectedly.")
        raise
    except ValueError as e:
         print(f"Error during train_test_split: {e}. This might be due to issues with the target variable (e.g., all NaNs after processing).")
         raise
    except Exception as e:
        print(f"An unexpected error occurred during data splitting: {e}")
        raise

