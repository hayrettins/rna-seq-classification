import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from mrmr import mrmr_classif
import joblib

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification script for RNA-seq data.")
    parser.add_argument('--processed_data', type=str, required=True, help="Path to the processed data CSV file.")
    parser.add_argument('--metadata', type=str, required=True, help="Path to the metadata output file from preprocessing.")
    parser.add_argument('--models', type=str, required=True, help="Path to the recommended models file.")
    parser.add_argument('--hyperparameters', type=str, required=True, help="Path to the best hyperparameters JSON file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output files.")
    parser.add_argument('--num_features', type=int, default=5, help="Number of top features to select.")
    return parser.parse_args()

def setup_output_directories(output_dir):
    reports_dir = os.path.join(output_dir, 'reports')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Output directories set up at {output_dir}")
    return reports_dir, plots_dir

def read_metadata(metadata_file):
    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    return metadata

def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully with shape {data.shape}")
    return data

def preprocess_data(data):
    X = data.drop('label', axis=1)
    y = data['label'].astype(str).str.lower().str.strip()  # Normalize labels

    # Generate label mapping dynamically
    unique_labels = y.unique()
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    logging.info(f"Generated label mapping: {label_mapping}")

    # Map labels to numerical values
    y_mapped = y.map(label_mapping)

    return X, y_mapped, label_mapping

def select_features(X, y, num_features):
    logging.info(f"Selecting top {num_features} features using mRMR")
    try:
        
        selected_features = mrmr_classif(X=X, y=y, K=num_features)
        logging.info(f"Selected features: {selected_features}")
    except Exception as e:
        logging.error(f"Feature selection failed: {e}")
        raise
    X_selected = X[selected_features]
    return X_selected, selected_features

def split_data(X, y):
    logging.info("Splitting data into train and test sets")
    return train_test_split(X, y, test_size=0.25, random_state=42)

def load_recommended_models(models_file):
    with open(models_file, 'r') as f:
        models = [line.strip() for line in f.readlines()]
    return models

def load_best_hyperparameters(hyperparameters_file):
    with open(hyperparameters_file, 'r') as f:
        best_params = json.load(f)
    return best_params

    logging.info("Training models")
    models = {}

    for model_name in models_to_train:
        logging.info(f"Training {model_name}")
        params = best_params.get(model_name, {})
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, **params)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**params)
        elif model_name == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
        else:
            logging.warning(f"Model {model_name} not recognized. Skipping.")
            continue

        model.fit(X_train, y_train)
        models[model_name] = model
        
        model_filename = os.path.join(output_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_filename)
        logging.info(f"Trained model saved to {model_filename}")

    return models

def train_models(X_train, y_train, models_to_train, best_params, output_dir):
    logging.info("Training models")
    models = {}

    for model_name in models_to_train:
        logging.info(f"Training {model_name}")
        params = best_params.get(model_name, {})
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, **params)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**params)
        elif model_name == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
        else:
            logging.warning(f"Model {model_name} not recognized. Skipping.")
            continue

        model.fit(X_train, y_train)
        models[model_name] = model
        model_filename = os.path.join(output_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_filename)
        logging.info(f"Trained model saved to {model_filename}")

    return models

def save_label_mapping(label_mapping, reports_dir):
    mapping_file = os.path.join(reports_dir, 'label_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(label_mapping, f)
    logging.info(f"Label mapping saved to {mapping_file}")

def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        metadata = read_metadata(args.metadata)

        data = load_data(args.processed_data)
        X, y, label_mapping = preprocess_data(data)

        save_label_mapping(label_mapping, args.output_dir)

        X_selected, selected_features = select_features(X, y, num_features=args.num_features)

        X_train, X_test, y_train, y_test = split_data(X_selected, y)

        test_data_file = os.path.join(args.output_dir, 'test_data.npz')
        np.savez(test_data_file, X_test=X_test, y_test=y_test)
        logging.info(f"Test data saved to {test_data_file}")

        models_to_train = load_recommended_models(args.models)
        best_params = load_best_hyperparameters(args.hyperparameters)

        train_models(X_train, y_train, models_to_train, best_params, args.output_dir)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()