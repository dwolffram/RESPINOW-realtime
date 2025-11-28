# Apply the KIT-hhh4 model with default settings (see below)
# Author: Johannes Bracher, johannes.bracher@kit.edu

######################################################
# set language to English
Sys.setlocale("LC_ALL", "C")

library(here)
here::i_am("r/hhh4/hhh4_default.R")

# Settings specific to this file:
label <- "hhh4-coupling"
# not excluding the COVID period in this file
exclusion_period <- as.Date(NULL)
# forecasts for different nowcast paths are aggregated by linear pool
aggregation_paths <- "linear pool"
# nowcast quantiles are ordered rather than shuffled
shuffle_paths <- FALSE
# last observation is not omitted
skip_last = FALSE


# define data sources and disease:
if (interactive()) {
  # Running in RStudio → pick manually
  disease <- "are"    # <- change here when testing
} else {
  # Running via Rscript → use CLI argument or default (e.g. Rscript nowcasting.R are)
  args <- commandArgs(trailingOnly = TRUE) # Read command line arguments
  disease <- ifelse(length(args) >= 1, args[1], "sari")
}

# map disease → data_source
data_source <- switch(
  disease,
  "sari" = "icosari",
  "are"  = "agi",
  stop("Unknown disease: ", disease)
)

message("Disease: ", disease)
message("Data source: ", data_source)

######################################################
# get packages and functions:
library(surveillance)
library(hhh4addon)
source(here("r", "hhh4", "functions_hhh4.R"))

######################################################
# get global setup shared between all versions of hhh4:
source(here("r", "hhh4", "setup_hhh4.R"))

######################################################
# run core code that can generate hhh4 forecasts for most settings
source(here("r", "hhh4", "core_hhh4.R"))
