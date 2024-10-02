import pandas as pd
import numpy as np
import argparse
import logging
import sys
import json
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization script for RNA-seq data.")
    parser.add_argument('--processed_data', type=str, required=True, help="Path to the processed data CSV file.")
    parser.add_argument('--metadata', type=str, required=True, help="Path to the metadata output file.")
    parser.add_argument('--models', type=str, required=True, help="Path to the recommended models file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the best hyperparameters.")
    return parser.parse_args()

def load_data(processed_data_file):
    data = pd.read_csv(processed_data_file)
    X = data.drop('label', axis=1)
    y = data['label'].astype(str).str.lower().str.strip()
    return X, y

def load_recommended_models(models_file):
    with open(models_file, 'r') as f:
        models = [line.strip() for line in f.readlines()]
    return models

def optimize_hyperparameters(X, y, models):
    best_params = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for model_name in models:
        logging.info(f"Optimizing hyperparameters for {model_name}")
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['lbfgs', 'saga'],
            }
        elif model_name == 'RandomForest':
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
            }
        elif model_name == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
            }
        else:
            logging.warning(f"Model {model_name} not recognized. Skipping.")
            continue

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_params[model_name] = grid_search.best_params_
        logging.info(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    return best_params

def save_best_hyperparameters(best_params, output_file):
    with open(output_file, 'w') as f:
        json.dump(best_params, f)
    logging.info(f"Best hyperparameters saved to {output_file}")

def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:

        X, y = load_data(args.processed_data)

        models = load_recommended_models(args.models)

        best_params = optimize_hyperparameters(X, y, models)

        save_best_hyperparameters(best_params, args.output_file)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()
