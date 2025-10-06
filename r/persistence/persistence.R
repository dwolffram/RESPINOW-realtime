# Apply the persistence baseline model to age-stratified data.
# Author: Johannes Bracher, johannes.bracher@kit.edu

# set language to English
Sys.setlocale("LC_ALL", "C")

# read in functions for the nowcasting method
source(here("r", "nowcasting", "functions_nowcasting.R"))
source(here("r", "persistence", "functions_persistence.R"))

# global settings:
# note: these are handled using lists so code can be adapted to other data sources

# define data sources and disease:
data_source <- "icosari"
disease <- "sari"
# the type of nowcast correction which is necessary:
type <- "additions"
# how borrowing of delays is done:
borrow_delays <- TRUE
# how borrowing of dispersion parameters is done
borrow_dispersion <- TRUE
# this just serves to suppress messages on subsetting which ar not relevant here
messages <- FALSE

# set the sizes of training data sets
# limited by number of observations (in the early part, not relevant anymore)
n_history_dispersion <- 15
n_history_expectations <- 15
max_delay <- 4
max_horizon <- 4
quantile_levels <- seq(2.5, 97.5, 2.5) / 100

# dates for which to produce nowcasts (stored centrally):
forecast_dates <- as.Date(
  read.csv(here("r", "auxiliary", "forecast_dates.csv"))$forecast_date
)

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

# run over forecast dates to generate nowcasts:
for (i in seq_along(forecast_dates)) {
  forecast_date <- forecast_dates[i]
  cat(as.character(forecast_dates[i]), "\n")

  # a place holder for a data frame in which nowcasts will be stored
  all_fc <- NULL

  # generate nowcasts for age groups
  # identify age groups:
  ags <- sort(unique(triangle$age_group))

  for (ag in ags) {
    # compute forecasts
    # note: pre-processing and subsetting of RT takes place inside
    fc <- compute_forecast(
      observed = triangle,
      location = "DE",
      age_group = ag,
      forecast_date = forecast_date,
      observed2 = triangle,
      location2 = "DE",
      age_group2 = "00+",
      max_horizon = max_horizon,
      type = type,
      borrow_delays = borrow_delays,
      borrow_dispersion = borrow_dispersion,
      # note using n_history_expectations_, n_history_dispersion_,
      # which may be reduced to fit shorter triangle.
      n_history_expectations = n_history_expectations,
      n_history_dispersion = n_history_dispersion,
      max_delay = max_delay,
      messages = messages
    )
    fc <- fc$result

    # store in all_fc:
    if (is.null(all_fc)) {
      all_fc <- fc
    } else {
      all_fc <- rbind(all_fc, fc)
    }
  }

  # write out:
  model_name <- "persistence"
  outdir <- here("forecasts", model_name)
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
    all_fc,
    file = here(
      "forecasts",
      model_name,
      filename
    ),
    row.names = FALSE
  )
}
