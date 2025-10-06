# Apply the KIT-hhh4 model skipping the last available (most incomplete)
# observation. Otherwise default settings.
# Author: Johannes Bracher, johannes.bracher@kit.edu

######################################################
# Settings specific to this file:
label <- "hhh4-discard"
# not excluding the COVID period in this file
exclusion_period <- NULL
# forecasts for different nowcast paths are aggregated by linear pool
aggregation_paths <- "linear pool"
# nowcast quantiles are ordered rather than shuffled
shuffle_paths <- FALSE
# last observation is not omitted
skip_last = TRUE # this is the interesting part here

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
