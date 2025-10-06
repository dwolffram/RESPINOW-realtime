# Shared setup for all hhh4 and also tscount models
# Contains global settings and reading in of data.
# Author: Johannes Bracher, johannes.bracher@kit.edu

###############################################################
# Global settings:

# define data sources and disease:
data_source <- "icosari"
disease <- "sari"

# dates for which to produce nowcasts (stored centrally):
forecast_dates <- as.Date(
  read.csv(here("r", "auxiliary", "forecast_dates.csv"))$forecast_date
)

# which quantile levels are contained in the nowcast?
quantile_levels_nowcast <- seq(2.5, 97.5, 2.5) / 100
# which quantile levels shall be contained in the output?
quantile_levels <- c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

# maximum delay in nowcasting:
max_delay <- 4

# maximum horizon in forecasting:
max_horizon <- 4

# define the control lists:
# for pooled data, i.e., "00+": includes end component
ctrl <- list(
  end = list(f = addSeason2formula(~1, period = 52.25)),
  ar = list(f = addSeason2formula(~1), period = 52.25),
  family = "NegBin1",
  par_lag = 0.5
)
# for age strata: exclude end component (see paper)
ctrl_strat <- list(
  end = list(f = ~0),
  ar = list(f = addSeason2formula(~1), period = 52.25),
  family = "NegBin1",
  par_lag = 0.5
)

###############################################################
# Read in data:

# construct time series:
# load most recent available data release. This also contains reports with more than 4 weeks delay.
# reaches back to 2023
latest_data <- read.csv(
  here(
    "data",
    paste0("latest_data-", data_source, "-", disease, ".csv")
  ),
  colClasses = c("date" = "Date"),
  check.names = FALSE
)

# load target time series. This only contains reports with max 4 weeks of delay.
# reaches back to 2014
target <- read.csv(
  here(
    "data",
    paste0("target-", data_source, "-", disease, ".csv")
  ),
  colClasses = c("date" = "Date"),
  check.names = FALSE
)

# identify age groups:
ags <- (unique(latest_data$age_group))

# set up time series data (long format for different age groups):
timeseries <- NULL
for (ag in ags) {
  # we use target where we can (from 2023 on) and latest data where we have to (2014-2022)
  min_date_target <- min(target$date[target$age_group == ag])
  latest_data_to_keep <- subset(
    latest_data,
    age_group == ag & date < min_date_target
  )
  target_to_keep <- subset(target, age_group == ag & date >= min_date_target)
  timeseries <- rbind(timeseries, latest_data_to_keep, target_to_keep)
}
