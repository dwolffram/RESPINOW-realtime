# Apply an extended version of the model suggested by Agosto et al
# using the tscount package. We replace the Poisson assumption by a negative
# binomial and add seasonality.
# Author: Johannes Bracher, johannes.bracher@kit.edu
# NOTE: this takes *much* longer than the hhh4 implementations as prediction is
# based on samples rather than analytical expressions for moments.

# set language to English
Sys.setlocale("LC_ALL", "C")


######################################################
# Settings specific to this file:
label <- "tscount-extended"
# Model (implemented: hhh4 and tscount following Agosto et al)
# Not excluding the COVID period in this file
exclusion_period <- as.Date(NULL)
# seq(from = as.Date("2019-06-30"), to = as.Date("2023-07-03"), by = 1)
aggregation_paths <- "linear pool"
# "quantile average"
# Not shuffling nowcast paths
shuffle_paths <- FALSE
# not skipping last observation
skip_last = FALSE
# tscount settings:
model <- list(past_obs = 1, past_mean = 1)
distr <- "nbinom" # different to simple
link <- "log"
include_seasonality <- TRUE # different to simple
######################################################

# get libraries and functions
library(tscount)
source(here("r", "hhh4", "functions_hhh4.R")) # also uses some functions from the hhh4 implementation

###############################################################
# get global setup from hhh4:
library(surveillance)
source(here("r", "hhh4", "setup_hhh4.R"))

######################################################
# run core code that generates tscount forecasts
source(here("r", "tscount", "core_tscount.R"))
