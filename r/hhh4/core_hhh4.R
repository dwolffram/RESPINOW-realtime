# Core code to apply the KIT-hhh4 model
# Author: Johannes Bracher, johannes.bracher@kit.edu

# This code is sourced with different settings in the following files:
# - hhh4_default
# - hhh4_exclude_covid
# - hhh4_shuffle
# - hhh4_skip
# - hhh4_vincentization

# set language to English
Sys.setlocale("LC_ALL", "C")

###############################################################
#  if last observation is skipped: need to predict one additional week.
if (skip_last) {
  max_horizon <- max_horizon + 1
}

###############################################################
# Generate predictions

# create directory for outputs if needed:
dir <- here("forecasts", label)
folder_exists <- dir.exists(dir)
if (!folder_exists) {
  dir.create(dir)
}

# run over forecast dates to generate forecastsss:
for (i in seq_along(forecast_dates)) {
  forecast_date <- forecast_dates[i]
  cat(as.character(forecast_dates[i]), "\n")

  # read in nowcast:
  nc <- read.csv(here(
    "nowcasts",
    "KIT-simple_nowcast",
    paste0(
      forecast_date,
      "-icosari-sari-KIT-simple_nowcast.csv"
    )
  ))
  # shuffle quantiles per age group, horizon and location if desired:
  if (shuffle_paths) {
    nc <- shuffle_quantiles(nc)
  }

  # a place holder for a data frame in which forecasts will be stored
  all_fc <- NULL

  # run through age groups:
  for (ag in ags) {
    # restrict time series
    ts <- subset(timeseries, age_group == ag & date <= forecast_date)
    # remove exclusion period
    ts$value[ts$date %in% exclusion_period] <- NA

    # now run forecasting for different nowcasting trajectories,
    # replacing last weeks with nowcasts

    # initialize matrices to store mu and var (predictive means and variances):
    mu_matrix <- var_matrix <- matrix(
      nrow = max_horizon,
      ncol = length(quantile_levels_nowcast)
    )

    # run through nowcasting trajectories / quantile levels
    for (j in seq_along(quantile_levels_nowcast)) {
      # get nowcast trajectory
      nc_temp <- subset(
        nc,
        quantile == quantile_levels_nowcast[j] & age_group == ag
      )
      # which dates does nowcast trajectory refer to? Order them even though they should already be ordered.
      dates_to_replace <- sort(nc_temp$target_end_date)
      # also re-order values in nowcast trajectory.
      values_to_replace <- nc_temp$value[order(nc_temp$target_end_date)]
      # plug nowcast trajectory into the time series
      ts_temp <- ts
      ts_temp$value[ts_temp$date %in% dates_to_replace] <- values_to_replace
      # remove last entry if skip_last:
      if (skip_last) {
        ts_temp <- ts_temp[-nrow(ts_temp), ]
      }

      # set up data for model fits
      # if skip_last, an additional week needs to be predicted, already covered in max_horizon
      sts_temp <- sts(
        c(round(ts_temp$value), rep(NA, max_horizon)),
        start = c(ts_temp$year[1], ts_temp$week[1])
      )
      # choose control
      ctrl_temp <- if (ag == "00+") ctrl else ctrl_strat
      ctrl_temp$subset <- 6:nrow(ts_temp)
      # fit model
      fit_temp <- hhh4_lag(sts_temp, control = ctrl_temp)

      # get moment-based predictions
      forecast_temp <- predictive_moments(
        fit_temp,
        t_condition = nrow(ts_temp),
        lgt = max_horizon
      )
      # store:
      mu_matrix[, j] <- forecast_temp$mu_matrix[, 1]
      var_matrix[, j] <- forecast_temp$var_matrix[, 1]
      # note: contains an aditional row for horizon 0 if skip_last == TRUE
    }

    # aggregate to predictive means and quantiles:
    pred_means <- rowMeans(mu_matrix)
    quantile_matrix <- aggregate_forecasts_to_quantiles(
      mu_matrix,
      var_matrix,
      quantile_levels_nowcast,
      quantile_levels,
      type = aggregation_paths
    )

    # identify target end dates:
    target_end_dates <- max(ts_temp$date) + 7 * (1:max_horizon) # also works for skip_last == TRUE

    # format everything:
    horizons <- (1:max_horizon) - skip_last # shift horizons if last observation skipped
    fc <- format_forecasts(
      quantile_matrix = quantile_matrix,
      means = pred_means,
      forecast_date = forecast_date,
      target_end_dates = target_end_dates,
      location = "DE",
      age_group = ag,
      horizons = horizons,
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
