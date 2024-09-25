#!/usr/bin/env nextflow

// Process 1: Data Preprocessing
process preprocess_data {
    publishDir 'results', mode: 'copy'

    input:
        path rnaseq_file
        path metadata_file
        path main_script
        path preprocessing_script

    output:
        path 'processed_data.csv'
        path 'metadata_output.txt'

    script:
    """
    Rscript ${main_script} \
        --rnaseq ${rnaseq_file} \
        --metadata ${metadata_file} \
        --output_csv processed_data.csv \
        --output_metadata metadata_output.txt
    """
}


process classification {
    publishDir 'results', mode: 'copy'

    input:
        path processed_data_csv
        path metadata_output_txt
        path classification_script

    output:
        path 'reports/*'
        path 'plots/*'
       

    script:
    """
    python3 ${classification_script} \
        --processed_data ${processed_data_csv} \
        --metadata ${metadata_output_txt} \
        --output_dir . \
        --num_features 5
    """
}


// Workflow Definition
workflow {
    // Input definitions
    def rnaseq_file = Channel.fromPath('data/GSE252145_Normalized_DESEQ.txt')
    def metadata_file = Channel.fromPath('data/metadata.txt')
    def main_script = Channel.fromPath('scripts/main.R')
    def preprocessing_script = Channel.fromPath('scripts/preprocessing.R')
    def classification_script = Channel.fromPath('scripts/classification.py')

    // Run preprocess_data process
    preprocess_data(rnaseq_file, metadata_file, main_script, preprocessing_script)

    // Access the outputs
    def processed_data_csv = preprocess_data.out[0]
    def metadata_output_txt = preprocess_data.out[1]
    // Run classification process, passing output from preprocess_data
    classification(processed_data_csv, metadata_output_txt, classification_script)
}
