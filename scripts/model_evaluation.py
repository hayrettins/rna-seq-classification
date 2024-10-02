import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model evaluation script for RNA-seq data.")
    parser.add_argument('--models_dir', type=str, required=True, help="Directory containing the trained model files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the evaluation reports and plots.")
    return parser.parse_args()

def load_test_data(models_dir):
    test_data_file = os.path.join(models_dir, 'test_data.npz')
    data = np.load(test_data_file)
    X_test = data['X_test']
    y_test = data['y_test']
    logging.info(f"Test data loaded from {test_data_file}")
    return X_test, y_test

def load_label_mapping(models_dir):
    mapping_file = os.path.join(models_dir, 'label_mapping.json')
    with open(mapping_file, 'r') as f:
        label_mapping = json.load(f)
    logging.info(f"Label mapping loaded from {mapping_file}")
    return label_mapping

def load_trained_models(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    models = {}
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '')
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        models[model_name] = model
        logging.info(f"Loaded model {model_name} from {model_path}")
    return models

def evaluate_models(models, X_test, y_test, label_mapping, output_dir):
    labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, 'reports')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

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

    try:

        X_test, y_test = load_test_data(args.models_dir)
        label_mapping = load_label_mapping(args.models_dir)
        models = load_trained_models(args.models_dir)
        evaluate_models(models, X_test, y_test, label_mapping, args.output_dir)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()
