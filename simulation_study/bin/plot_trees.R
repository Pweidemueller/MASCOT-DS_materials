#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript plot_trees.R <tree_file> <type>\n  type: 'groundtruth', 'datastreams', or 'original'")
}

tree_file <- args[1]
tree_type <- args[2]

# Validate tree type
valid_types <- c("groundtruth", "datastreams", "original")
if (!tree_type %in% valid_types) {
  stop(paste("Error: type must be one of:", paste(valid_types, collapse = ", ")))
}

# Check if tree file exists
if (!file.exists(tree_file)) {
  stop(paste("Error: Tree file not found:", tree_file))
}

# Load required libraries
library(tidyverse)
library(ggtree)
library(treeio)

# Read tree
tree <- read.beast(tree_file)

# Function to analyze deme switches
analyze_deme_switches <- function(tree, tree_type) {
  # Get the phylo object
  phylo <- tree@phylo
  
  # Get deme information based on tree type
  if (tree_type == "groundtruth") {
    # Ground truth uses 'type' field
    if (!"type" %in% names(tree@data)) {
      stop("Error: 'type' field not found in tree data")
    }
    deme_data <- tree@data$type
    names(deme_data) <- tree@data$node
  } else if (tree_type %in% c("original", "datastreams")) {
    # Original and datastreams use 'max' field
    if (!"max" %in% names(tree@data)) {
      stop("Error: 'max' field not found in tree data")
    }
    deme_data <- tree@data$max
    names(deme_data) <- tree@data$node
  }
  print(head(tree@data))
  
  # Get edge matrix (parent-child relationships)
  edges <- phylo$edge
  colnames(edges) <- c("parent", "child")
  
  # Get deme for each node
  parent_demes <- deme_data[as.character(edges[, "parent"])]
  child_demes <- deme_data[as.character(edges[, "child"])]
  
  # Determine if each edge is a switch (where parent and child have different demes)
  valid_edges <- !is.na(parent_demes) & !is.na(child_demes)
  is_switch <- parent_demes != child_demes & valid_edges
  
  # Create results data frame with all edges
  results <- data.frame(
    child_node = edges[valid_edges, "child"],
    parent_node = edges[valid_edges, "parent"],
    Deme_parent = as.integer(str_extract(parent_demes[valid_edges], "\\d+")),
    Deme_child = as.integer(str_extract(child_demes[valid_edges], "\\d+")),
    switch = ifelse(is_switch[valid_edges], "yes", "no"),
    stringsAsFactors = FALSE
  )
  
  return(results)
}

# Analyze deme switches
deme_results <- analyze_deme_switches(tree, tree_type)

# Print summary
cat("\n=== Deme Switch Analysis ===\n")
cat(paste("Total number of parent-child pairs:", nrow(deme_results), "\n"))
n_switches <- sum(deme_results$switch == "yes")
cat(paste("Number of deme switches:", n_switches, "\n"))
cat(paste("Number of same-deme pairs (no switch):", nrow(deme_results) - n_switches, "\n\n"))
# Summary by switch type
cat("\nSummary by deme transition:\n")
switch_summary <- deme_results %>%
  count(Deme_parent, Deme_child, switch, name = "count") %>%
  arrange(desc(count))
print(switch_summary, row.names = FALSE)
cat("\n")

# Generate output filenames based on input filename
base_name <- gsub("\\.(trees|tree)$", "", basename(tree_file))
csv_file <- paste0(base_name, "_deme_switches_", tree_type, ".csv")
output_file <- paste0(base_name, "_tree_", tree_type, ".png")

# Save results table as CSV
write.csv(switch_summary, file = csv_file, row.names = FALSE)
cat(paste("Results table saved to:", csv_file, "\n\n"))

# Plot based on tree type
if (tree_type == "groundtruth") {
  # Ground truth tree uses 'type' field with I{0 and I{1
  p <- ggtree(tree, aes(color = type), linewidth = 1) +
    theme_classic(base_size = 16) +
    geom_tippoint(show.legend = FALSE) +
    labs(x = "Time", color = 'Deme') +
    scale_color_manual(
      values = c("I{0" = "#56b3e9", "I{1" = "#e69d00"),
      labels = c("I{0" = "0", "I{1" = "1")
    ) +
    theme(
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank()
    ) +
    ggtitle('true simulated tree')
  
} else if (tree_type %in% c("original", "datastreams")) {
  # Original and datastreams trees use 'max' field with I0 and I1
  title_text <- ifelse(tree_type == "original", "MASCOT", "MASCOT datastreams")
  
  p <- ggtree(tree, aes(color = max), linewidth = 1) +
    theme_classic(base_size = 16) +
    geom_tippoint(show.legend = FALSE) +
    labs(x = "Time", color = 'Deme') +
    scale_color_manual(
      values = c("I0" = "#56b3e9", "I1" = "#e69d00"),
      labels = c("I0" = "0", "I1" = "1")
    ) +
    theme(
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank()
    ) +
    ggtitle(title_text)
}

# Save plot
# Set height dynamically: base height of 2, plus 0.15 per leaf, max with min height 3
n_leaves <- length(tree@phylo$tip.label)
height <- max(3, 2 + 0.04 * n_leaves)
ggsave(output_file, plot = p, dpi = 300, width = 5, height = height)
cat(paste("Plot saved to:", output_file, "\n"))

