# main.py
import argparse
from pathlib import Path
import yaml # PyYAML library needed
import sys

# Add src directory to sys.path BEFORE importing from src
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from utils import load_config
from data import make_dataset
from models import train_model, evaluate_model

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
        print("\n>>> Running Training Stage <<<")
        try:
            # Load processed data (train_model loads both train/test needed for consistency)
            X_train, y_train, X_test, y_test = train_model.load_processed_data(config)
            # Train model and save artifacts
            trained_model, fitted_scaler = train_model.train_model(X_train, y_train, config)
            print(">>> Training Stage Completed <<<\n")

            # Optionally run evaluation immediately after training in 'train' or 'full' stage
            print("\n>>> Running Evaluation Stage (Post-Training) <<<")
            evaluate_model.evaluate_model(trained_model, fitted_scaler, X_test, y_test, config)
            print(">>> Evaluation Stage Completed <<<\n")

        except Exception as e:
            print(f"Error during training/evaluation: {e}")
            sys.exit(1)

    # Note: An explicit 'evaluate' stage might load the *saved* model/scaler
    # instead of using the ones returned by train_model, useful for testing loading.
    # This implementation runs evaluation right after training for simplicity here.
    elif args.stage == 'evaluate':
         print("\n>>> Running Standalone Evaluation Stage <<<")
         # This would typically load saved model, scaler, and test data
         print("Standalone evaluation stage needs implementation (loading saved artifacts).")
         # Example (needs load_joblib from utils):
         # try:
         #     model = utils.load_joblib(Path(config['artifacts']['model_save_path']))
         #     scaler = utils.load_joblib(Path(config['artifacts']['scaler_save_path']))
         #     test_data = utils.load_joblib(Path(config['data']['processed_test_path']))
         #     X_test = test_data['X']
         #     y_test = test_data['y']
         #     evaluate_model.evaluate_model(model, scaler, X_test, y_test, config)
         #     print(">>> Standalone Evaluation Stage Completed <<<\n")
         # except Exception as e:
         #     print(f"Error during standalone evaluation: {e}")
         #     sys.exit(1)


if __name__ == '__main__':
    main()