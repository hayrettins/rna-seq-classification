library(biomaRt)
library(dplyr)


process_rnaseq_data <- function(rnaseq_file, mart) {

  data <- read.csv(rnaseq_file, sep = '\t', header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
  cat("Initial RNA-seq data dimensions:", dim(data), "\n")
  
 
  gene_ids <- data[[1]]
  cat("Sample gene IDs before cleaning:\n")
  print(head(gene_ids))
  

  gene_ids <- sub("\\..*", "", gene_ids)
  cat("Sample gene IDs after cleaning:\n")
  print(head(gene_ids))
 
  data[[1]] <- gene_ids
  colnames(data)[1] <- "gene_id"
  

  id_type <- detect_gene_id_type(gene_ids, mart)
  cat("Detected gene ID type:", id_type, "\n")
  
  new_genes <- get_gene_mappings(gene_ids, id_type, mart)
  cat("Number of gene mappings retrieved from BioMart:", nrow(new_genes), "\n")
  
  if (nrow(new_genes) == 0) {
    stop("No gene mappings were retrieved from BioMart. Check if gene IDs match.")
  }
  
 
  data <- merge(new_genes, data, by.x = id_type, by.y = "gene_id", all.y = FALSE)
  cat("Data dimensions after merging with gene mappings:", dim(data), "\n")
  

  data <- data[!duplicated(data$ensembl_gene_id), ]
  cat("Data dimensions after removing duplicate genes:", dim(data), "\n")
  
  
  data <- data[!is.na(data$ensembl_gene_id), ]
  cat("Data dimensions after removing NAs in 'ensembl_gene_id':", dim(data), "\n")
  
 
  row.names(data) <- data$ensembl_gene_id
  data <- data[, !(names(data) %in% c(id_type, "ensembl_gene_id"))]
  
  num_rows <- nrow(data)
  num_cols <- ncol(data)
  
  if (num_cols > num_rows) {
    cat("Detected samples as columns and genes as rows after mapping.\n")
    samples_are_columns <- TRUE
  } else {
    cat("Detected samples as rows and genes as columns after mapping.\n")
    samples_are_columns <- FALSE
    data <- t(data)
    cat("Data dimensions after transposing:", dim(data), "\n")
  }
  
  
  processed_data <- as.data.frame(data)
  row.names(processed_data) <- NULL  
  
  cat("Final processed data dimensions:", dim(processed_data), "\n")
  
  return(processed_data)
}

read_metadata <- function(metadata_file) {
 
  metadata <- read.table(metadata_file, header = FALSE, sep = "", stringsAsFactors = FALSE, fill = TRUE)
  sample_names <- metadata[[1]]
  
  # Extract treatment status from the last two columns
  num_cols <- ncol(metadata)
  labels_raw <- metadata[, (num_cols - 1):num_cols]  # Get the last two columns
  labels <- apply(labels_raw, 1, function(row) paste(row[!is.na(row)], collapse = " "))
  
  # Normalize labels
  labels <- tolower(labels)  # Convert to lowercase
  labels <- gsub("-", " ", labels)  # Replace hyphens with spaces
  labels <- trimws(labels)  # Trim whitespace
  labels <- ifelse(grepl("pre", labels), "pre treatment", labels)
  labels <- ifelse(grepl("post", labels), "post treatment", labels)
  

  print("Extracted and normalized labels:")
  print(labels)
  
  return(list(sample_names = sample_names, labels = labels))
}


write_output_files <- function(combined_data, output_csv, output_txt) {

  write.csv(combined_data, output_csv, row.names = FALSE)
  
  # Calculate metadata
  num_samples <- nrow(combined_data)
  if ("label" %in% colnames(combined_data)) {
    num_features <- ncol(combined_data) - 1  # Subtract 1 for the label column
    num_classes <- length(unique(combined_data$label))
  } else {
    num_features <- ncol(combined_data)
    num_classes <- NA  # No class labels provided
  }
  
  metadata <- list(
    num_samples = num_samples,
    num_features = num_features,
    num_classes = num_classes
  )
  

  metadata_lines <- c(
    paste("num_samples:", metadata$num_samples),
    paste("num_features:", metadata$num_features),
    paste("num_classes:", ifelse(is.na(metadata$num_classes), "N/A", metadata$num_classes))
  )
  
  writeLines(metadata_lines, con = output_txt)
}

detect_gene_id_type <- function(gene_ids, mart) {
  id_types <- c("ensembl_gene_id", "entrezgene_id", "external_gene_name", "hgnc_symbol")
  id_counts <- c()
  for (id_type in id_types) {
    cat("Trying ID type:", id_type, "\n")
    mappings <- getBM(attributes = c(id_type), filters = id_type, values = unique(gene_ids), mart = mart)
    count <- nrow(mappings)
    cat("Number of matches for", id_type, ":", count, "\n")
    id_counts <- c(id_counts, count)
  }
  max_count <- max(id_counts)
  if (max_count == 0) {
    stop("Could not determine the gene ID type. No matches found in BioMart.")
  }
  detected_id_type <- id_types[which.max(id_counts)]
  return(detected_id_type)
}


get_gene_mappings <- function(gene_ids, id_type, mart) {
  attributes <- c(id_type, "ensembl_gene_id")
  mappings <- getBM(attributes = attributes, filters = id_type, values = gene_ids, mart = mart)
  mappings <- unique(mappings)
  return(mappings)
}
