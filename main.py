# main.py
import argparse
from pathlib import Path
import yaml # PyYAML library needed
import sys
import mlflow # Import MLflow
import mlflow.sklearn # Import MLflow scikit-learn integration

# Add src directory to sys.path BEFORE importing from src
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from utils import load_config
from data import make_dataset
# Import the specific functions needed, including the new ones
from models.train_model import load_processed_data, train_model_with_tuning
from models.evaluate_model import evaluate_model_and_log

def main():
    parser = argparse.ArgumentParser(description="Churn Prediction Project Pipeline")
    parser.add_argument(
        'stage',
        choices=['preprocess', 'train', 'evaluate', 'full'], # Added 'full' pipeline
        help='Pipeline stage to run (preprocess, train, evaluate, or full)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to the configuration file (default: config/config.yaml)'
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(Path(args.config))
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1) # Exit if config fails to load

    # --- Execute pipeline stages ---
    if args.stage == 'preprocess' or args.stage == 'full':
        print("\n>>> Running Preprocessing Stage <<<")
        try:
            raw_df = make_dataset.load_raw_data(config)
            processed_df = make_dataset.preprocess_data(raw_df, config)
            make_dataset.split_data(processed_df, config)
            print(">>> Preprocessing Stage Completed <<<\n")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            sys.exit(1)

    if args.stage == 'train' or args.stage == 'full':
        print("\n>>> Running Training Stage with MLflow Tracking <<<")
        try:
            # Set MLflow experiment name (will create if it doesn't exist)
            mlflow_experiment_name = config.get('mlflow_experiment_name', 'Churn Prediction Experiments')
            mlflow.set_experiment(mlflow_experiment_name)
            print(f"MLflow experiment set to: '{mlflow_experiment_name}'")

            # Start an MLflow run
            with mlflow.start_run():
                run_id = mlflow.active_run().info.run_id
                print(f"Started MLflow run with ID: {run_id}")

                # Log key configuration parameters
                print("Logging parameters to MLflow...")
                mlflow.log_param("model_name", config['model_selection']['name'])
                mlflow.log_param("test_size", config['data']['test_size'])
                mlflow.log_param("random_state", config['random_state'])
                # Log the scoring metric used for tuning if available in config or known
                # mlflow.log_param("tuning_scoring", "recall_pos_label_1") # Example

                # Load data
                print("Loading processed data...")
                X_train, y_train, X_test, y_test = load_processed_data(config)

                # Train model using the function with tuning
                print("Starting training and hyperparameter tuning...")
                trained_model, fitted_scaler, best_params, cv_score = train_model_with_tuning(X_train, y_train, config)
                print(">>> Training and Tuning Stage Completed <<<\n")

                # Log best parameters and CV score found by GridSearchCV
                print("Logging best parameters and CV score...")
                mlflow.log_params(best_params)
                # Make sure the scoring name matches what GridSearchCV used
                # Adjust "best_cv_recall_score" if you used a different scorer
                mlflow.log_metric("best_cv_recall_score", cv_score)

                # Evaluate on test set using the function that logs to MLflow
                print("\n>>> Running Evaluation Stage (Post-Training) <<<")
                # evaluate_model_and_log will log test metrics to MLflow
                metrics = evaluate_model_and_log(trained_model, fitted_scaler, X_test, y_test, config)
                print(">>> Evaluation Stage Completed <<<\n")

                # Log model and scaler as MLflow artifacts
                print("Logging model and scaler as MLflow artifacts...")
                mlflow.sklearn.log_model(trained_model, "model")
                # Saving scaler as a generic artifact might be more robust if needed outside sklearn context
                # Or use log_model if you always load it as a scikit-learn object
                mlflow.log_artifact(config['artifacts']['scaler_save_path'], artifact_path="scaler")
                # Alternatively, log the scaler object directly if always used with sklearn:
                # mlflow.sklearn.log_model(fitted_scaler, "scaler")

                # Log other artifacts (reports, plots)
                print("Logging evaluation artifacts...")
                try:
                    mlflow.log_artifact(config['evaluation']['metrics_save_path'])
                    mlflow.log_artifact(config['evaluation']['report_save_path'])
                    mlflow.log_artifact(config['evaluation']['confusion_matrix_save_path'])
                    # Check if feature importance path exists before logging
                    fi_path = Path(config['evaluation'].get('feature_importance_save_path', ''))
                    if fi_path.exists():
                         mlflow.log_artifact(str(fi_path))
                    else:
                         print(f"Feature importance plot not found at {fi_path}, skipping MLflow logging.")

                except Exception as artifact_error:
                    print(f"Warning: Could not log one or more evaluation artifacts: {artifact_error}")

                print(f"MLflow Run {run_id} finished successfully.")

        except Exception as e:
            print(f"Error during training/evaluation with MLflow: {e}")
            # Optionally log the error to the MLflow run before exiting
            # mlflow.log_param("run_status", "failed")
            # mlflow.log_param("error_message", str(e))
            # if mlflow.active_run(): # Ensure run is active before setting status
            #     mlflow.end_run(status="FAILED")
            sys.exit(1)

    # Note: An explicit 'evaluate' stage might load the *saved* model/scaler
    # from a specific MLflow run ID for more controlled evaluation.
    elif args.stage == 'evaluate':
         print("\n>>> Running Standalone Evaluation Stage <<<")
         print("Standalone evaluation stage needs implementation (loading artifacts from MLflow or local path).")
         # Example: Load model from MLflow run ID (replace RUN_ID)
         # logged_model = 'runs:/RUN_ID/model'
         # loaded_model = mlflow.pyfunc.load_model(logged_model)
         # ... load scaler, test data ...
         # evaluate_model.evaluate_model_and_log(...) # Use the logging version


if __name__ == '__main__':
    main()
