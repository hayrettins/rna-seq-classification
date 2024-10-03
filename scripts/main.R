library(jsonlite)  
library(dplyr)     
library(logging)   

basicConfig()
addHandler(writeToFile, file = "preprocess_data.log", level = 'DEBUG')


option_list <- list(
  make_option(c("-r", "--rnaseq"), type = "character", help = "Path to RNA-seq data file", metavar = "FILE"),
  make_option(c("-m", "--metadata"), type = "character", help = "Path to structured metadata JSON file", metavar = "FILE"),
  make_option(c("-o", "--output_csv"), type = "character", default = "processed_data.csv", help = "Output CSV file path", metavar = "FILE"),
  make_option(c("-t", "--output_metadata"), type = "character", default = "metadata_output.txt", help = "Output metadata TXT file path", metavar = "FILE")
)


opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)


process_data_pipeline <- function(rnaseq_file, metadata_file, output_csv, output_txt) {
  
  loginfo("Starting data processing pipeline.")
  

  metadata <- tryCatch({
    fromJSON(metadata_file)
  }, error = function(e) {
    logerror("Failed to parse metadata JSON: %s", e$message)
    stop("Exiting due to JSON parsing error.")
  })

  logdebug("Metadata Content:")
  logdebug(metadata)
  

  num_samples <- metadata$number_of_samples
  num_features <- metadata$number_of_features
  num_classes <- metadata$number_of_classes
  class_names <- metadata$class_names
  

  logdebug(paste("Number of samples:", num_samples))
  logdebug(paste("Number of features:", num_features))
  logdebug(paste("Number of classes:", num_classes))
  logdebug(paste("Class names:", paste(class_names, collapse = ", ")))
  

  if (is.null(num_samples) | is.null(num_features) | is.null(num_classes) | is.null(class_names)) {
    logerror("One or more metadata fields are missing.")
    stop("Exiting due to missing metadata fields.")
  }
  

  rnaseq_data <- tryCatch({
    read.table(rnaseq_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
  }, error = function(e) {
    logerror("Failed to read RNA-seq data file: %s", e$message)
    stop("Exiting due to RNA-seq data read error.")
  })
  
  loginfo("Initial RNA-seq data dimensions: %d x %d", nrow(rnaseq_data), ncol(rnaseq_data))
  

  processed_data <- tryCatch({
    t(rnaseq_data)
  }, error = function(e) {
    logerror("Failed to process RNA-seq data: %s", e$message)
    stop("Exiting due to data processing error.")
  })
  
  loginfo("Processed data dimensions: %d x %d", nrow(processed_data), ncol(processed_data))
  

  if (nrow(processed_data) != num_samples) {
    logerror("Number of samples mismatch: RNA-seq data has %d samples, metadata specifies %d samples.", nrow(processed_data), num_samples)
    stop("The number of samples in the RNA-seq data does not match the number of samples in the metadata.")
  }

  tryCatch({
    write.csv(processed_data, file = output_csv, row.names = TRUE)
    loginfo("Processed data saved to %s", output_csv)
  }, error = function(e) {
    logerror("Failed to write processed data to CSV: %s", e$message)
    stop("Exiting due to write error.")
  })
  

  tryCatch({
    writeLines(c(
      paste("Number of samples:", num_samples),
      paste("Number of features:", num_features),
      paste("Number of classes:", num_classes),
      paste("Class names:", paste(class_names, collapse = ", "))
    ), con = output_txt)
    loginfo("Metadata output saved to %s", output_txt)
  }, error = function(e) {
    logerror("Failed to write metadata output: %s", e$message)
    stop("Exiting due to write error.")
  })
  
  loginfo("Data processing complete. Output files generated.")
}

process_data_pipeline(
  rnaseq_file = opt$rnaseq,
  metadata_file = opt$metadata,
  output_csv = opt$output_csv,
  output_txt = opt$output_metadata
)
