# main.py
import argparse
from pathlib import Path
import yaml # PyYAML library needed
import sys
import mlflow # Import MLflow
import mlflow.sklearn # Import MLflow scikit-learn integration

# Add src directory to sys.path BEFORE importing from src
# This needs to be robust
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from utils import load_config
from data import make_dataset
# Import the specific functions needed, including the new ones
# Import training and evaluation functions only when needed in their stages
# from models.train_model import load_processed_data, train_model_with_tuning
# from models.evaluate_model import evaluate_model_and_log

def main():
    parser = argparse.ArgumentParser(description="Churn Prediction Project Pipeline")
    parser.add_argument(
        'stage',
        # Add 'predict' to the choices
        choices=['preprocess', 'train', 'evaluate', 'full', 'predict'],
        help='Pipeline stage to run (preprocess, train, evaluate, full, or predict)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to the configuration file (default: config/config.yaml)'
    )
    # Add arguments specific to the predict stage
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to the input CSV file for prediction (required for predict stage)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='predictions.csv', # Default output filename
        help='Path to save the prediction results CSV file (default: predictions.csv)'
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
        # Import training/evaluation functions only when needed
        from models.train_model import load_processed_data, train_model_with_tuning
        from models.evaluate_model import evaluate_model_and_log

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
                # Saving scaler as a generic artifact might be more robust
                mlflow.log_artifact(config['artifacts']['scaler_save_path'], artifact_path="scaler")

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
            # if mlflow.active_run():
            #     mlflow.end_run(status="FAILED")
            sys.exit(1)

    # Standalone evaluate stage (implementation still needed for loading from MLflow/local)
    elif args.stage == 'evaluate':
         print("\n>>> Running Standalone Evaluation Stage <<<")
         print("Standalone evaluation stage needs implementation (loading artifacts from MLflow or local path).")
         # Example: Load model from MLflow run ID (replace RUN_ID)
         # logged_model = 'runs:/RUN_ID/model'
         # loaded_model = mlflow.pyfunc.load_model(logged_model)
         # ... load scaler, test data ...
         # evaluate_model.evaluate_model_and_log(...) # Use the logging version

    # New Predict Stage
    elif args.stage == 'predict':
        print("\n>>> Running Prediction Stage <<<")
        # Check if input file argument was provided
        if not args.input_file:
            print("Error: --input-file argument is required for the 'predict' stage.")
            parser.print_help() # Show help message
            sys.exit(1)

        try:
            # Import the prediction function only when needed
            from models.predict_model import make_prediction

            print(f"Attempting to make predictions on input file: {args.input_file}")
            # Make predictions using the specified input file
            prediction_results = make_prediction(
                input_data=args.input_file, # Pass the input file path
                config_path=args.config     # Pass the config file path
            )

            if prediction_results is not None:
                # Save results to the specified output file
                output_path = Path(args.output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                prediction_results.to_csv(output_path, index=False)
                print(f"Prediction results saved successfully to: {output_path}")
                print(">>> Prediction Stage Completed <<<\n")
            else:
                # make_prediction function should have printed error details
                print("Prediction process failed. Check logs above for details.")
                print(">>> Prediction Stage Failed <<<\n")
                sys.exit(1)

        except ImportError:
             print("Error: Could not import 'make_prediction' from models.predict_model.")
             print("Please check the file structure and ensure the function exists.")
             sys.exit(1)
        except FileNotFoundError:
             print(f"Error: Input file not found at {args.input_file}")
             sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during the prediction stage: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
    