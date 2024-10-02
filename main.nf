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

process model_selection {
    publishDir 'results', mode: 'copy'

    input:
        path processed_data_csv
        path metadata_output_txt
        path model_selection_script

    output:
        path 'recommended_models.txt'

    script:
    """
    python3 ${model_selection_script} \
        --processed_data ${processed_data_csv} \
        --metadata ${metadata_output_txt} \
        --output_file recommended_models.txt
    """
}


process hyperparameter_optimization {
    publishDir 'results', mode: 'copy'

    input:
        path processed_data_csv
        path metadata_output_txt
        path recommended_models_txt
        path hyperparameter_optimization_script

    output:
        path 'best_hyperparameters.json'

    script:
    """
    python3 ${hyperparameter_optimization_script} \
        --processed_data ${processed_data_csv} \
        --metadata ${metadata_output_txt} \
        --models ${recommended_models_txt} \
        --output_file best_hyperparameters.json
    """
}

process classification {
    publishDir 'results', mode: 'copy'

    input:
        path processed_data_csv
        path metadata_output_txt
        path recommended_models_txt
        path best_hyperparameters_json
        path classification_script

    output:
        path 'reports/*'
        path 'plots/*'

    script:
    """
    python3 ${classification_script} \
        --processed_data ${processed_data_csv} \
        --metadata ${metadata_output_txt} \
        --models ${recommended_models_txt} \
        --hyperparameters ${best_hyperparameters_json} \
        --output_dir . \
        --num_features ${params.num_features}
    """
}


workflow {

  
    Channel.fromPath(params.rnaseq_file).set { rnaseq_file }
    Channel.fromPath(params.metadata_file).set { metadata_file }
    Channel.fromPath(params.main_script).set { main_script }
    Channel.fromPath(params.preprocessing_script).set { preprocessing_script }
    Channel.fromPath(params.model_selection_script).set { model_selection_script }
    Channel.fromPath(params.hyperparameter_optimization_script).set { hyperparameter_optimization_script }
    Channel.fromPath(params.classification_script).set { classification_script }

  
    preprocess_data(
        rnaseq_file,
        metadata_file,
        main_script,
        preprocessing_script
    )
    def processed_data_csv = preprocess_data.out[0]
    def metadata_output_txt = preprocess_data.out[1]


   
    model_selection(
        processed_data_csv,
        metadata_output_txt,
        model_selection_script
    )

    
    def recommended_models_txt = model_selection.out[0]

    
    hyperparameter_optimization(
        processed_data_csv,
        metadata_output_txt,
        recommended_models_txt,
        hyperparameter_optimization_script
    )

    def best_hyperparameters_json = hyperparameter_optimization.out[0]

    
    classification(
        processed_data_csv,
        metadata_output_txt,
        recommended_models_txt,
        best_hyperparameters_json,
        classification_script
    )
}
