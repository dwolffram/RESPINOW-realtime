# Apply the KIT-hhh4 model to uncorrected data ("naive")
# Otherwise default settings.
# Author: Johannes Bracher, johannes.bracher@kit.edu

# As the entire nowcasting block is removed and vintage data are needed instead,
# a separate file was set up.

# set language to English
Sys.setlocale("LC_ALL", "C")


######################################################
# Settings specific to this file:
label <- "hhh4-naive"
# Not excluding the COVID period in this file
exclusion_period <- as.Date(NULL)
# there is no nowcasting happening here, thus no specification of aggregate_paths and shuffle_paths
######################################################

######################################################
# get packages and functions:
library(surveillance)
library(hhh4addon)
source(here("r", "hhh4", "functions_hhh4.R"))
source(here("r", "nowcasting", "functions_nowcasting.R")) # also requires nowcasting functions

###############################################################
# get global setup shared between all versions of hhh4:
source(here("r", "hhh4", "setup_hhh4.R"))


##############################################################
# read in reporting triangle
triangle <- read.csv(
  here(
    "data",
    paste0("reporting_triangle-", data_source, "-", disease, ".csv")
  ),
  colClasses = c("date" = "Date"),
  check.names = FALSE
)
# this is needed here as unlike in the other versions (where nowcasts generated
# separately are read in) we actually use vintage data here.

###############################################################
# Generate predictions

# create directory if needed:
dir <- here("forecasts", label)
folder_exists <- dir.exists(dir)
if (!folder_exists) {
  dir.create(dir)
}

# run over forecast dates to generate nowcasts:
for (i in seq_along(forecast_dates)) {
  forecast_date <- forecast_dates[i]
  cat(as.character(forecast_dates[i]), "\n")

  # a place holder for a data frame in which forecasts will be stored
  all_fc <- NULL

  # run through age groups:
  for (ag in ags) {
    # restrict time series
    ts <- subset(timeseries, age_group == ag & date <= forecast_date)
    # remove exclusion period
    ts$value[ts$date %in% exclusion_period] <- NA

    # plug in most recent incomplete values:
    # truth data as of forecast_date, subset to relevant stratum
    reporting_triangle_back_in_time <- data_as_of(
      dat_truth = triangle,
      date = forecast_date,
      location = "DE",
      age_group = ag,
      max_lag = max_delay
    )

    # which dates does incomplete part correspond to?
    dates_to_replace <- tail(reporting_triangle_back_in_time$date, max_delay)
    # also re-order values in nowcast trajectory.
    values_to_replace <- replacement <- tail(
      rowSums(
        reporting_triangle_back_in_time[, grepl(
          "value_",
          colnames(reporting_triangle_back_in_time)
        )],
        na.rm = TRUE
      ),
      max_delay
    )

    # plug incomplete values into the time series
    ts_realtime <- ts
    ts_realtime$value[
      ts_realtime$date %in% dates_to_replace
    ] <- values_to_replace

    # now run forecasting
    # set up data for model fits
    sts <- sts(
      c(round(ts_realtime$value), rep(NA, max_horizon)),
      start = c(ts_realtime$year[1], ts_realtime$week[1])
    )
    # choose control
    ctrl_temp <- if (ag == "00+") ctrl else ctrl_strat
    ctrl_temp$subset <- 6:nrow(ts_realtime)
    # fit model
    fit_temp <- hhh4_lag(sts, control = ctrl_temp)
    # get moment-based predictions
    forecast_temp <- predictive_moments(
      fit_temp,
      t_condition = nrow(ts_realtime),
      lgt = max_horizon
    )

    # predictive means and quantiles can be obtained directly here as no nowcast paths
    mu <- forecast_temp$mu_matrix[, 1]
    sigma2 <- forecast_temp$var_matrix[, 1]
    size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)

    pred_means <- mu
    quantile_matrix <- matrix(
      nrow = max_horizon,
      ncol = length(quantile_levels)
    )
    fc <- NULL
    for (h in 1:4) {
      # compute quantiles:
      quantile_matrix[h, ] <- qnbinom(
        quantile_levels,
        mu = mu[h],
        size = size[h]
      )
    }

    # identify target end dates:
    target_end_dates <- max(ts_realtime$date) + 7 * (1:max_horizon)

    # format everything:
    fc <- format_forecasts(
      quantile_matrix = quantile_matrix,
      means = pred_means,
      forecast_date = forecast_date,
      target_end_dates = target_end_dates,
      location = "DE",
      age_group = ag,
      horizons = 1:max_horizon,
      quantile_levels = quantile_levels
    )

    # add to data.frame containing forecasts for all age groups:
    all_fc <- rbind(all_fc, fc)
  }

  # write out:
  filename <- paste0(
    forecast_date,
    "-",
    data_source,
    "-",
    disease,
    "-",
    label,
    ".csv"
  )

  write.csv(
    all_fc,
    file = here(
      "forecasts",
      label,
      filename
    ),
    row.names = FALSE
  )
}
