# src/utils.py
# Placeholder for utility functions if needed later
# For now, standard libraries like pathlib, joblib, yaml handle main tasks.

import yaml
from pathlib import Path
import joblib
import json

def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def save_joblib(obj: object, file_path: Path):
    """Saves an object using joblib."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        print(f"Object saved to {file_path}")
    except Exception as e:
        print(f"Error saving object to {file_path}: {e}")
        raise

def load_joblib(file_path: Path) -> object:
    """Loads an object using joblib."""
    try:
        obj = joblib.load(file_path)
        print(f"Object loaded from {file_path}")
        return obj
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading object from {file_path}: {e}")
        raise

def save_metrics(metrics: dict, file_path: Path):
    """Saves metrics dictionary to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {file_path}")
    except Exception as e:
        print(f"Error saving metrics to {file_path}: {e}")
        raise

def save_report(report: str, file_path: Path):
    """Saves a text report (like classification report) to a file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {file_path}")
    except Exception as e:
        print(f"Error saving report to {file_path}: {e}")
        raise

# Add more utility functions as needed, e.g., logging setup