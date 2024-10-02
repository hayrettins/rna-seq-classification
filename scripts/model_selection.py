import pandas as pd
import argparse
import logging
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model selection script for RNA-seq data.")
    parser.add_argument('--processed_data', type=str, required=True, help="Path to the processed data CSV file.")
    parser.add_argument('--metadata', type=str, required=True, help="Path to the metadata output file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the recommended models.")
    return parser.parse_args()

def read_metadata(metadata_file):
    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    return metadata

def analyze_data(data, metadata):
    
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1  
    num_classes = int(metadata.get('num_classes', 2))

    logging.info(f"Number of samples: {num_samples}")
    logging.info(f"Number of features: {num_features}")
    logging.info(f"Number of classes: {num_classes}")

    recommended_models = []


    if num_samples < 100:
        recommended_models.append('LogisticRegression')
        recommended_models.append('RandomForest')
    else:
        recommended_models.append('RandomForest')
        recommended_models.append('XGBoost')

    if num_features > 1000:
        logging.info("High dimensionality detected. Considering models that handle high-dimensional data well.")
   
    return recommended_models

def save_recommended_models(models, output_file):
    with open(output_file, 'w') as f:
        for model in models:
            f.write(f"{model}\n")
    logging.info(f"Recommended models saved to {output_file}")

def main():
    
    args = parse_arguments()

    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:

        data = pd.read_csv(args.processed_data)

        metadata = read_metadata(args.metadata)

        recommended_models = analyze_data(data, metadata)

        save_recommended_models(recommended_models, args.output_file)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()

