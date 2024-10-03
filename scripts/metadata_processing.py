#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
import pandas as pd  

def parse_arguments():
    parser = argparse.ArgumentParser(description="Metadata processing script using OpenAI LLM.")
    parser.add_argument('--raw_metadata', type=str, required=True, help="Path to the raw metadata file.")
    parser.add_argument('--rnaseq', type=str, required=True, help="Path to the RNA-seq data file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the structured metadata JSON file.")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def read_raw_metadata(raw_metadata_path):
    if not os.path.exists(raw_metadata_path):
        logging.error(f"Raw metadata file does not exist: {raw_metadata_path}")
        sys.exit(1)
    with open(raw_metadata_path, 'r') as file:
        content = file.read()
    logging.info("Raw metadata read successfully.")
    return content

def read_rnaseq_file(rnaseq_path):
    if not os.path.exists(rnaseq_path):
        logging.error(f"RNA-seq data file does not exist: {rnaseq_path}")
        sys.exit(1)
    try:
        rnaseq_data = pd.read_csv(rnaseq_path, sep="\t", header=0, index_col=0, dtype=str)
        num_features = rnaseq_data.shape[0]
        logging.info(f"Number of features (genes) in RNA-seq data: {num_features}")
        return num_features
    except Exception as e:
        logging.error(f"Failed to read RNA-seq data file: {e}")
        sys.exit(1)

def generate_prompt(raw_metadata_content, num_features):
    prompt = f"""
    You are a data processing assistant. Extract the following information from the given metadata:

    1. Number of samples
    2. Number of classes
    3. Class names

    Additionally, include the number of features provided separately.

    Provide the information in the following JSON format:

    {{
        "number_of_samples": <int>,
        "number_of_features": {num_features},
        "number_of_classes": <int>,
        "class_names": [<string>, ...]
    }}

    Metadata:
    {raw_metadata_content}
    """
    return prompt

def call_openai_api(prompt):
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        sys.exit(1)

def parse_response(response_text):
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        metadata = json.loads(json_str)
        # Validate required fields
        required_fields = ["number_of_samples", "number_of_features", "number_of_classes", "class_names"]
        for field in required_fields:
            if field not in metadata:
                logging.error(f"Missing field '{field}' in structured metadata.")
                sys.exit(1)
        return metadata
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        logging.error("Response text received from OpenAI:")
        logging.error(response_text)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to parse OpenAI response: {e}")
        sys.exit(1)

def save_structured_metadata(metadata, output_file):
    try:
        with open(output_file, 'w') as file:
            json.dump(metadata, file, indent=4)
        logging.info(f"Structured metadata saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save structured metadata: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    setup_logging()

    if 'OPENAI_API_KEY' not in os.environ:
        logging.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    raw_metadata_content = read_raw_metadata(args.raw_metadata)
    num_features = read_rnaseq_file(args.rnaseq)
    prompt = generate_prompt(raw_metadata_content, num_features)
    response_text = call_openai_api(prompt)
    structured_metadata = parse_response(response_text)
    save_structured_metadata(structured_metadata, args.output_file)

if __name__ == "__main__":
    main()
