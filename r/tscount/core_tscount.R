# Core code to apply the KIT-tscount model
# Author: Johannes Bracher, johannes.bracher@kit.edu

# This code is sourced with different settings in the following files:
# - hhh4_default
# - hhh4_exclude_covid
# - hhh4_shuffle
# - hhh4_skip
# - hhh4_vincentization

###############################################################
# Auxiliary things:
interval_level_1sd <- pnorm(0.5) - pnorm(-0.5) # a constant needed below to compute predictive variances
#  if last observation is skipped: need to predict one additional week.
if (skip_last) {
  max_horizon <- max_horizon + 1
}

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
  cat("Starting", as.character(forecast_dates[i]), "\n")

  # read in nowcast:
  nc <- read.csv(here(
    "nowcasts",
    "KIT-simple_nowcast",
    paste0(forecast_date, "-icosari-sari-KIT-simple_nowcast.csv")
  ))
  # shuffle quantiles per age group, horizon and location if desired:
  if (shuffle_paths) {
    nc <- shuffle_quantiles(nc)
  }

  # a place holder for a data frame in which forecasts will be stored
  all_fc <- NULL

  # run through age groups:
  for (ag in ags) {
    cat(ag, "...")

    # restrict time series
    ts <- subset(timeseries, age_group == ag & date <= forecast_date)
    # remove exclusion period
    ts$value[ts$date %in% exclusion_period] <- NA

    # now run forecasting for different nowcasting trajectories,
    # replacing last weeks with nowcasts and generate forecasts

    # initialize matrices to store mu and var:
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

      # generate covariates (seasonality if desired)
      xreg_past <- xreg_new <- NULL
      if (include_seasonality) {
        xreg <- matrix(ncol = 2, nrow = nrow(ts_temp) + max_horizon)
        colnames(xreg) <- c("sin", "cos")
        xreg[, 1] <- sin((2 * pi * 1:nrow(xreg) / 52.25))
        xreg[, 2] <- cos((2 * pi * 1:nrow(xreg) / 52.25))
        # split:
        xreg_past <- xreg[1:nrow(ts_temp), ] # for fitting
        xreg_new <- xreg[nrow(ts_temp) + 1:max_horizon, ] # for forecasting
      }

      # fit model:
      fit_temp <- tsglm(
        ts_temp$value,
        model = model,
        distr = distr,
        link = link,
        xreg = xreg_past
      )
      # generate forecast:
      forecast_temp <- predict(
        fit_temp,
        n.ahead = max_horizon,
        level = interval_level_1sd,
        newxreg = xreg_new
      )

      # extract predictive moments:
      mu_matrix[, j] <- forecast_temp$pred
      # variance can be computed from intervals:
      var_matrix[, j] <- (forecast_temp$interval[, 2] -
        forecast_temp$interval[, 1])^2
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
    target_end_dates <- max(ts_temp$date) + 7 * (1:max_horizon)
    # shift if skip_last == TRUE
    if (skip_last) {
      target_end_dates <- target_end_dates + 7
    }

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

    cat(" done.\n")
  }

  # write out:
  write.csv(
    all_fc,
    file = here(
      "forecasts",
      label,
      paste0(
        forecast_date,
        "-",
        data_source,
        "-",
        disease,
        "-",
        label,
        ".csv"
      )
    ),
    row.names = FALSE
  )
}
