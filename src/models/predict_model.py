# src/models/predict_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Any, Tuple, Union, List, Dict # Thêm List, Dict
import sys
import joblib # Sử dụng joblib để tải mô hình/scaler đã lưu

# Add src directory to sys.path for utils import
# This needs to be robust to different execution contexts
try:
    # Standard path when running from project root via main.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils import load_config, load_joblib
except ImportError:
    # Fallback if running script directly or structure differs
    sys.path.append('.')
    try:
        from src.utils import load_config, load_joblib
    except ImportError:
        print("Error: Could not import utility functions. Ensure you are running from the project root"
              " or the PYTHONPATH is set correctly.")
        sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


def load_artifacts(config: dict) -> Tuple[Any, StandardScaler, List[str]]:
    """
    Loads the trained model, scaler, and training column names.

    Args:
        config: Configuration dictionary.

    Returns:
        A tuple containing (loaded_model, loaded_scaler, training_columns).
    """
    model_path = Path(config['artifacts']['model_save_path'])
    scaler_path = Path(config['artifacts']['scaler_save_path'])
    # Path to the processed training data to get column names
    train_data_path = Path(config['data']['processed_train_path'])

    print(f"Loading model from: {model_path}")
    model = load_joblib(model_path)

    print(f"Loading scaler from: {scaler_path}")
    scaler = load_joblib(scaler_path)

    # Load training columns - CRUCIAL for consistent preprocessing
    try:
        print(f"Loading training data columns from: {train_data_path}")
        train_data = load_joblib(train_data_path)
        # Ensure 'X' exists and has columns attribute
        if 'X' in train_data and hasattr(train_data['X'], 'columns'):
            train_columns = train_data['X'].columns.tolist()
            print(f"Successfully loaded {len(train_columns)} training column names.")
        else:
             raise ValueError("Processed training data file is missing 'X' DataFrame or columns.")

    except Exception as e:
        print(f"Error loading training columns from {train_data_path}: {e}")
        print("Cannot proceed with prediction without knowing the training columns.")
        raise # Re-raise the exception to stop execution

    print("Model, scaler, and training columns loaded successfully.")
    return model, scaler, train_columns

def preprocess_new_data(
    new_data: pd.DataFrame,
    config: dict,
    scaler: StandardScaler,
    train_columns: List[str] # Required list of columns from training
    ) -> Union[pd.DataFrame, None]: # Return None on failure
    """
    Preprocesses new customer data EXACTLY like the training data.
    Handles missing values, one-hot encodes categoricals, ensures column
    consistency, and scales numerical features.

    Args:
        new_data: DataFrame containing new customer data. Must have columns
                  similar to the original raw data (before dropping ID/target).
        config: Project configuration dictionary.
        scaler: The StandardScaler object fitted on the training data.
        train_columns: List of feature column names (after OHE) from the
                       training data. Used to ensure consistency.

    Returns:
        DataFrame containing the preprocessed new data, ready for prediction.
        Returns None if preprocessing fails critically.
    """
    print("Starting preprocessing for new data...")
    if not isinstance(new_data, pd.DataFrame):
        print("Error: new_data must be a pandas DataFrame.")
        return None

    df_processed = new_data.copy()

    # 1. Handle Missing Values (specifically TotalCharges for this dataset)
    print("Handling missing values (TotalCharges)...")
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        # Check if fillna is needed AFTER converting to numeric
        if df_processed['TotalCharges'].isnull().any():
             df_processed['TotalCharges'].fillna(0, inplace=True)
             print("Filled NaN in TotalCharges with 0.")
        else:
             print("No NaN found in TotalCharges after numeric conversion.")
    else:
        print("Warning: 'TotalCharges' column not found in new data.")

    # 2. One-Hot Encode Categorical Features
    print("Encoding categorical features...")
    categorical_cols_config = config['features'].get('categorical_cols_ohe', []) # Use .get for safety

    # Determine columns to encode: Use config list or detect object columns if config is empty
    if not categorical_cols_config:
        cols_to_encode_in_new = df_processed.select_dtypes(include='object').columns.tolist()
        print(f"Auto-detected categorical columns for OHE in new data: {cols_to_encode_in_new}")
    else:
        # Use columns from config, but only those present in the new data
        cols_to_encode_in_new = [col for col in categorical_cols_config if col in df_processed.columns]
        print(f"Using categorical columns from config for OHE (if present): {cols_to_encode_in_new}")

    if cols_to_encode_in_new:
        try:
            df_processed = pd.get_dummies(df_processed, columns=cols_to_encode_in_new, drop_first=True)
            print(f"Applied One-Hot Encoding to: {cols_to_encode_in_new}")
        except Exception as e:
            print(f"Error during One-Hot Encoding: {e}")
            return None # Stop if OHE fails
    else:
        print("No categorical columns found or specified for One-Hot Encoding in new data.")

    # 3. Ensure Consistent Column Structure with Training Data
    print("Ensuring consistent column structure...")
    current_cols = df_processed.columns.tolist()

    # Add missing columns (present in train, not in new) with value 0
    missing_cols = set(train_columns) - set(current_cols)
    if missing_cols:
        print(f"Adding {len(missing_cols)} missing columns found in training data:")
        for c in missing_cols:
            df_processed[c] = 0
            # print(f"  Added column: {c}") # Can be verbose
    else:
        print("No missing columns compared to training data.")

    # Remove extra columns (present in new, not in train)
    extra_cols = set(current_cols) - set(train_columns)
    if extra_cols:
        print(f"Removing {len(extra_cols)} extra columns not present in training data:")
        # print(f"  Columns to remove: {list(extra_cols)}") # Can be verbose
        df_processed = df_processed.drop(columns=list(extra_cols))
    else:
        print("No extra columns found compared to training data.")

    # Reorder columns to match the training data order
    try:
        df_processed = df_processed[train_columns]
        print(f"Columns reordered to match training data ({len(train_columns)} columns).")
    except KeyError as e:
        print(f"Error reordering columns: A required training column is missing even after adding defaults: {e}")
        print(f"Columns in processed data: {df_processed.columns.tolist()}")
        print(f"Expected training columns: {train_columns}")
        return None # Stop if reordering fails

    # 4. Scale Numerical Features
    numerical_cols_config = config['features'].get('numerical_cols_to_scale', [])
    # Identify numerical columns that actually exist after OHE and alignment
    cols_to_scale_in_processed = [col for col in numerical_cols_config if col in df_processed.columns]

    if cols_to_scale_in_processed:
        print(f"Scaling numerical features: {cols_to_scale_in_processed}")
        try:
            # Use the loaded scaler to transform the new data
            df_processed[cols_to_scale_in_processed] = scaler.transform(df_processed[cols_to_scale_in_processed])
            print("Numerical features scaled successfully.")
        except Exception as e:
            print(f"Error during scaling: {e}")
            return None # Stop if scaling fails
    else:
        print("No numerical columns specified in config found in the final processed data to scale.")

    print("Preprocessing of new data finished.")
    return df_processed

def make_prediction(
    input_data: Union[pd.DataFrame, List[Dict], str], # Allow path to CSV
    config_path: str = 'config/config.yaml'
    ) -> Union[pd.DataFrame, None]:
    """
    Performs churn prediction on new input data.

    Args:
        input_data: New customer data, which can be:
                    - A pandas DataFrame.
                    - A list of dictionaries (each dict is a customer).
                    - A string path to a CSV file.
        config_path: Path to the project's YAML configuration file.

    Returns:
        A pandas DataFrame containing the original input data (or data read
        from CSV) with an added 'ChurnProbability' column.
        Returns None if prediction fails at any critical step.
    """
    print("--- Starting Churn Prediction ---")
    # 1. Load Configuration
    try:
        config = load_config(Path(config_path))
    except Exception as e:
        print(f"Failed to load configuration from {config_path}: {e}")
        return None

    # 2. Load Model, Scaler, and Training Columns
    try:
        model, scaler, train_columns = load_artifacts(config)
    except Exception as e:
        print(f"Failed to load necessary artifacts: {e}")
        return None

    # 3. Prepare Input DataFrame
    print("Preparing input data...")
    if isinstance(input_data, str): # If input is a file path
        try:
            input_path = Path(input_data)
            input_df = pd.read_csv(input_path)
            print(f"Loaded data from CSV: {input_path}")
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_data}")
            return None
        except Exception as e:
            print(f"Error reading input CSV file {input_data}: {e}")
            return None
    elif isinstance(input_data, list): # If input is list of dicts
        try:
            input_df = pd.DataFrame(input_data)
            print("Converted list of dictionaries to DataFrame.")
        except Exception as e:
            print(f"Error converting list of dictionaries to DataFrame: {e}")
            return None
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy() # Work on a copy
        print("Using provided DataFrame as input.")
    else:
        print("Error: Invalid input_data format. Must be DataFrame, List[Dict], or CSV path string.")
        return None

    # Keep the original data to merge results later
    original_input_df = input_df.copy()
    # Drop ID column if present, as it wasn't used in training
    # Use .get on config['features'] for safety, provide default empty list
    id_cols_to_drop = config['features'].get('cols_to_drop', [])
    if id_cols_to_drop:
        # Only drop columns that actually exist in the input
        cols_present = [col for col in id_cols_to_drop if col in input_df.columns]
        if cols_present:
             input_df = input_df.drop(columns=cols_present)
             print(f"Dropped potential ID columns {cols_present} for preprocessing.")


    # 4. Preprocess the New Data
    processed_input_df = preprocess_new_data(input_df, config, scaler, train_columns)

    if processed_input_df is None:
        print("Prediction aborted due to preprocessing failure.")
        return None # Stop if preprocessing failed

    # 5. Make Probability Predictions
    print("Making churn probability predictions...")
    try:
        # Ensure columns match exactly what the model expects (handled by preprocess)
        probabilities = model.predict_proba(processed_input_df)[:, 1] # Get probability of class 1 (Churn)
        print("Prediction successful.")
    except ValueError as ve:
        print(f"Error during prediction: {ve}")
        print("This usually indicates a mismatch between processed input features and features seen during training.")
        # Debugging info:
        print(f"Columns in final preprocessed data ({len(processed_input_df.columns)}): {processed_input_df.columns.tolist()}")
        if hasattr(model, 'n_features_in_'):
             print(f"Features expected by model: {model.n_features_in_}")
        if hasattr(model, 'feature_names_in_'):
             print(f"Feature names expected by model: {model.feature_names_in_}")
        return None # Stop if prediction fails
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return None


    # 6. Add Probabilities to Original Data and Return
    print("Adding prediction results to original data...")
    # Use index of original_input_df to ensure correct alignment if rows were dropped/filtered
    # If original_input_df might have different index, consider merging carefully
    # Assuming index remains consistent for this example:
    original_input_df['ChurnProbability'] = probabilities
    print("--- Churn Prediction Finished ---")
    return original_input_df

# --- Example Usage ---
if __name__ == '__main__':
    # This block allows running the script directly for testing
    print("\n--- Running Prediction Script in Test Mode ---")

    # Option 1: Create sample data as a list of dictionaries
    # Make sure these dictionaries contain all necessary columns expected
    # in the raw data BEFORE preprocessing (except target/ID if dropped).
    sample_data_list = [
        { # Customer 1: Likely to churn (short tenure, month-to-month, fiber, high charges)
            'customerID': 'TestCust001', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.70, 'TotalCharges': '151.65' # Note: TotalCharges as string initially
        },
        { # Customer 2: Less likely to churn (long tenure, two year contract, lower charges)
            'customerID': 'TestCust002', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
            'tenure': 68, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
            'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
            'Contract': 'Two year', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 84.80, 'TotalCharges': '5790.8'
        }
    ]
    print("\n--- Predicting using List of Dictionaries ---")
    # Call the prediction function with the list of dictionaries
    predictions_from_list = make_prediction(input_data=sample_data_list)

    # Print the results
    if predictions_from_list is not None:
        print("\nPrediction Results (from List):")
        print(predictions_from_list[['customerID', 'ChurnProbability']].round(4)) # Show ID and probability
        # print(predictions_from_list) # Uncomment to print the full DataFrame
    else:
        print("\nPrediction failed for the list input.")


    # Option 2: Create a sample CSV file for testing (Optional)
    # You would create a file named 'sample_input.csv' in the project root
    # with content similar to the dictionaries above (comma-separated).
    # Example: Create 'sample_input.csv' with header and the two rows above.
    sample_csv_path = 'sample_input.csv'
    try:
        # Create a DataFrame from the list and save to CSV for the example
        pd.DataFrame(sample_data_list).to_csv(sample_csv_path, index=False)
        print(f"\n--- Predicting using CSV file ({sample_csv_path}) ---")

        # Call the prediction function with the CSV file path
        predictions_from_csv = make_prediction(input_data=sample_csv_path)

        # Print the results
        if predictions_from_csv is not None:
            print("\nPrediction Results (from CSV):")
            print(predictions_from_csv[['customerID', 'ChurnProbability']].round(4))
            # print(predictions_from_csv) # Uncomment to print the full DataFrame
        else:
            print("\nPrediction failed for the CSV input.")

        # Clean up the sample CSV file (optional)
        # import os
        # os.remove(sample_csv_path)

    except Exception as e:
        print(f"\nCould not run CSV prediction example (maybe failed to write file?): {e}")


    print("\n--- Prediction Script Test Mode Finished ---")

       