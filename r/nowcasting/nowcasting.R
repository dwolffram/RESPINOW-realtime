# Apply the KIT-simple_nowcast baseline model to age-stratified SARI data.
# Author: Johannes Bracher, johannes.bracher@kit.edu

# set language to English
Sys.setlocale("LC_ALL", "C")


library(here)
here::i_am("r/nowcasting/nowcasting.R")

# read in functions for the nowcasting method
source(here("r", "nowcasting", "functions_nowcasting.R"))

# global settings:
# note: these are handled using lists so code can be adapted to other data sources

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

# the type of nowcast correction which is necessary:
type <- "additions"
# how borrowing of delays is done:
borrow_delays <- TRUE
# how borrowing of dispersion parameters is done
borrow_dispersion <- TRUE
# this just serves to suppress messages on subsetting which are not relevant here
messages <- FALSE

# set the sizes of training data sets
# limited by number of observations (in the early part, not relevant anymore)
n_history_dispersion <- 15
n_history_expectations <- 15
max_delay <- 4
max_horizon <- 3
quantile_levels <- seq(2.5, 97.5, 2.5) / 100

# dates for which to produce nowcasts (stored centrally):
# forecast_dates <- as.Date(
#   read.csv(here("r", "auxiliary", "forecast_dates.csv"))$forecast_date
# )

# Select most recent Thursday as forecast_date:
forecast_dates0 <- Sys.Date() - 0:6
forecast_dates <- forecast_dates0[weekdays(forecast_dates0) == "Thursday"]

# read in reporting triangle:
triangle <- read.csv(
  here(
    "data",
    paste0("reporting_triangle-", data_source, "-", disease, ".csv")
  ),
  colClasses = c("date" = "Date"),
  check.names = FALSE
)
# note: we load the raw RT, preprocessing takes place inside compute_nowcast
# this ensures data are used as they were available at a given point in time

# run over forecast_dates for which to generate nowcasts:
for (i in seq_along(forecast_dates)) {
  # select forecast_date from vector
  forecast_date <- forecast_dates[i]
  cat(as.character(forecast_dates[i]), "\n")

  # a place holder, to be overwritten by a data frame to store nowcasts
  all_nc <- NULL

  # identify age groups:
  ags <- sort(unique(triangle$age_group))

  # run through age groups to generate nowcasts:
  for (ag in ags) {
    # compute nowcast.
    # note: pre-processing and subsetting of RT takes place inside
    nc <- compute_nowcast(
      observed = triangle,
      location = "DE",
      age_group = ag,
      forecast_date = forecast_date,
      observed2 = triangle,
      location2 = "DE",
      age_group2 = "00+",
      type = type,
      borrow_delays = borrow_delays,
      borrow_dispersion = borrow_dispersion,
      n_history_expectations = n_history_expectations,
      n_history_dispersion = n_history_dispersion,
      max_delay = max_delay,
      quantile_levels = quantile_levels,
      messages = messages
    )
    nc <- nc$result

    # store in all_nc:
    if (is.null(all_nc)) {
      # create if does not exist yet
      all_nc <- nc
    } else {
      all_nc <- rbind(all_nc, nc)
    }
  }

  # write out:
  model_name <- "simple_nowcast"
  outdir <- here("nowcasts", model_name)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  filename <- paste0(
    forecast_date,
    "-",
    data_source,
    "-",
    disease,
    "-",
    model_name,
    ".csv"
  )

  write.csv(
    all_nc,
    file = here(
      "nowcasts",
      model_name,
      filename
    ),
    row.names = FALSE
  )
}
