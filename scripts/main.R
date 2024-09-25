library(optparse)
source("preprocessing.R")

# CLI 
option_list <- list(
  make_option(c("-r", "--rnaseq"), type = "character", help = "Path to RNA-seq data file", metavar = "FILE"),
  make_option(c("-m", "--metadata"), type = "character", default = NULL, help = "Path to metadata file (optional)", metavar = "FILE"),
  make_option(c("-o", "--output_csv"), type = "character", default = "processed_data.csv", help = "Output CSV file path", metavar = "FILE"),
  make_option(c("-t", "--output_metadata"), type = "character", default = "metadata_output.txt", help = "Output metadata TXT file path", metavar = "FILE")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

process_data_pipeline <- function(rnaseq_file, metadata_file = NULL, output_csv, output_txt) {
  ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  processed_data <- process_rnaseq_data(rnaseq_file, ensembl)
  cat("Processed data dimensions:", dim(processed_data), "\n")
  
  if (!is.null(metadata_file)) {
    metadata <- read_metadata(metadata_file)
    sample_names <- metadata$sample_names
    labels <- metadata$labels
    cat("Number of labels:", length(labels), "\n")
    cat("Labels:\n")
    print(labels)
    
    if (nrow(processed_data) != length(labels)) {
      stop("The number of samples in the RNA-seq data does not match the number of labels.")
    }
    
   
    combined_data <- cbind(processed_data, label = labels)
    cat("Combined data dimensions:", dim(combined_data), "\n")
    cat("Unique labels in combined data:\n")
    print(unique(combined_data$label))
  } else {
    warning("No metadata provided. Proceeding without labels.")
    combined_data <- processed_data
  }
  
  
  row.names(combined_data) <- NULL
  
  write_output_files(combined_data, output_csv, output_txt)
  
  cat("Data processing complete. Output files generated:\n")
  cat(" - Data CSV:", output_csv, "\n")
  cat(" - Metadata TXT:", output_txt, "\n")
}

process_data_pipeline(
  rnaseq_file = opt$rnaseq,
  metadata_file = opt$metadata,
  output_csv = opt$output_csv,
  output_txt = opt$output_metadata
)
