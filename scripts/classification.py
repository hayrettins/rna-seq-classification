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

def train_models(X_train, y_train, models_to_train, best_params):
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

    return models

def evaluate_models(models, X_test, y_test, label_mapping, reports_dir, plots_dir):
    labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    for model_name, model in models.items():
        logging.info(f"Evaluating model: {model_name}")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{model_name} Accuracy: {accuracy}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info(f"{model_name} Confusion Matrix:\n{conf_matrix}")

        class_report = classification_report(y_test, y_pred)
        logging.info(f"{model_name} Classification Report:\n{class_report}")

        save_results(model_name, accuracy, conf_matrix, class_report, reports_dir)

        plot_confusion_matrix(conf_matrix, labels, model_name, normalize=False, output_dir=plots_dir)


def save_results(model_name, accuracy, conf_matrix, class_report, reports_dir):
    
    results_file = os.path.join(reports_dir, f'{model_name}_classification_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    logging.info(f"{model_name} classification results saved to {results_file}")

  

def save_label_mapping(label_mapping, reports_dir):
    mapping_file = os.path.join(reports_dir, 'label_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(label_mapping, f)
    logging.info(f"Label mapping saved to {mapping_file}")

def plot_confusion_matrix(cm, classes, model_name, normalize=False, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory for plots at {output_dir}")

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info(f"Plotting normalized confusion matrix for {model_name}")
        cm_to_plot = cm_normalized
        title = f'Confusion Matrix (Normalized) - {model_name}'
        filename = f'{model_name}_confusion_matrix_normalized.png'
    else:
        logging.info(f"Plotting confusion matrix for {model_name}")
        cm_to_plot = cm
        title = f'Confusion Matrix - {model_name}'
        filename = f'{model_name}_confusion_matrix.png'

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_to_plot, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {plot_path}")

def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    reports_dir, plots_dir = setup_output_directories(args.output_dir)

    metadata = read_metadata(args.metadata)
    num_classes_meta = metadata.get('num_classes', None)
    if num_classes_meta is not None:
        num_classes_meta = int(num_classes_meta)
        logging.info(f"Number of classes from metadata: {num_classes_meta}")
    else:
        logging.warning("Number of classes not specified in metadata.")

    data = load_data(args.processed_data)
    X, y, label_mapping = preprocess_data(data)

    num_classes_data = len(label_mapping)
    if num_classes_meta is not None and num_classes_meta != num_classes_data:
        logging.warning(f"Number of classes in metadata ({num_classes_meta}) does not match number of classes in data ({num_classes_data}).")

    save_label_mapping(label_mapping, reports_dir)

    X_selected, selected_features = select_features(X, y, num_features=args.num_features)

    X_train, X_test, y_train, y_test = split_data(X_selected, y)

    models_to_train = load_recommended_models(args.models)
    best_params = load_best_hyperparameters(args.hyperparameters)

    models = train_models(X_train, y_train, models_to_train, best_params)

    evaluate_models(models, X_test, y_test, label_mapping, reports_dir, plots_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)
