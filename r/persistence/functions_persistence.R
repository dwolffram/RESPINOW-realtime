# Functions implementing the persistence forecasting method (which features a nowcasting step).
# Depends on functions_nowcasting.R
# Author: Johannes Bracher, johannes.bracher@kit.edu

#' Function to generate the persistence forecast
#' @param observed the observations / reporting triangle data.frame
#' @param location the location for which to generate nowcasts
#' @param age_group the age group for which to generate nowcasts
#' @param forecast_date the date when the nowcast is issued. The function automatically restricts
#' the reportng triangle if an earlier forecast_date is provided to imitate real-time nowcasting.
#' @param type type of nowcasting task: either "additions" for delayed reporting or "revise_average"
#' if an estimate gests refined (and can move either way)
#' @param max_horizon maximum horizon for which to generate forecasts
#' @param borrow_delays if borrow_delays == TRUE, a separate reporting triangle observed2 is used to
#' estimate the delay distribution. Typically used to share strength across strata (then observed2,
#' location2, age_group2 coresponds to a pooled reporting triangle).
#' @param borrow_dispersion if borrow_delays == TRUE, the separate reporting triangle is also used to
#' estimate the nowcast dispersion.
#' @param observed2: the observations / reporting triangle matrix pooled across strata.
#' This can be used to estimate delay distributions more reliably if they are similar across strata.
#' @param location2 the location used to estimate the delay distribution (and possibly dispersion).
#' @param age_group2 the age group used to estimate the delay distribution (and possibly dispersion).
#' @param weekday_end_of_week the weekday when data updates happen (needed to know which data were available when).
#' @param max_delay the maximum delay considered.
#' @param n_history_expectations the number of observations used to estimate the delay distribution
#' @param n_history_dispersion the number of re-computed nowcasts used to estimate the error variance
#' @param quantile_levels the predictive quantile levels to return
compute_forecast <- function(
  observed,
  location = "DE",
  age_group = "00+",
  forecast_date,
  type = "additions",
  max_horizon = 4,
  borrow_delays = FALSE,
  borrow_dispersion = FALSE,
  observed2 = NULL,
  location2 = NULL,
  age_group2 = NULL,
  weekday_data_updates = "Thursday",
  max_delay = 4,
  n_history_expectations = 15,
  n_history_dispersion = 15,
  quantile_levels = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
  messages = TRUE
) {
  # check reporting triangle and forecast_date:
  if (forecast_date > max(observed$date) + 7) {
    stop(
      "Reporting triangle not up to date. forecast_date can be at most max(observed$date) + 7."
    )
  }

  if (any(observed$date >= forecast_date) & messages) {
    message(
      "Reporting triangle contains dates later than forecast_date. ",
      " Note that data will be subsetted to those available on forecast_date (if applicable, negative delays are respected)."
    )
  }

  if (weekdays(forecast_date) != weekday_data_updates) {
    warning(
      "forecast_date is a different weekday than weekday_data_updates. This may be unintended."
    )
  }

  # Can only use observed2 for dispersion if also used for delay distribution:
  if (borrow_dispersion & !borrow_delays) {
    stop("borrow_dispersion == TRUE is only allowed of borrow_delays == TRUE")
  }

  # Check type is allowed
  if (!type %in% c("additions", "revise_average")) {
    stop("type needs to be either 'additions' or 'revise_average'.")
  }

  # pre-process reporting triangle.
  # note: only done for type == "additions" where nowcasting methods assume positive increments.
  if (type == "additions") {
    observed <- preprocess_reporting_triangle(observed)
    if (!is.null(observed2)) {
      observed2 <- preprocess_reporting_triangle(observed2)
    }
  }

  # which horizons need to be considered?
  horizons <- 1:max_horizon

  # bring to state as of forecast_date, subset to location and age group:
  observed_as_of <- data_as_of(
    observed,
    age_group = age_group,
    location = location,
    date = forecast_date,
    weekday_data_updates = weekday_data_updates,
    max_lag = max_delay,
    return_matrix = TRUE
  )
  # pad with NAs if needed
  observed_as_of <- pad_matr(
    observed_as_of,
    n_history_expectations + n_history_dispersion
  )
  # if sharing desired: same for observed2_as_of
  if (borrow_delays) {
    observed2_as_of <- data_as_of(
      observed2,
      age_group = age_group2,
      location = location2,
      date = forecast_date,
      weekday_data_updates = weekday_data_updates,
      max_lag = max_delay,
      return_matrix = TRUE
    )
    # no padding here as observed2_as_of needs to have the desired size
  } else {
    # if no sharing desired: set all _2 variables to their regular counterparts
    if (!is.null(observed2) | !is.null(location2) | !is.null(age_group2)) {
      warning(
        "observed2, location2 or age_group2 were provided despite borrow_delays == FALSE. These will be ignored."
      )
    }
    observed2_as_of <- observed_as_of
    location2 <- location
    age_group2 <- age_group
  }

  # generate point nowcast:
  point_forecast <- compute_expectations(
    observed = observed_as_of,
    observed2 = observed2_as_of,
    n_history = n_history_expectations,
    borrow_delays = borrow_delays,
    remove_observed = FALSE
  )
  mu <- rep(sum(point_forecast[nrow(point_forecast), ]), max_horizon)
  # estimate size parameters for negative binomial:
  disp_params <- fit_dispersion_forecast(
    observed = if (borrow_dispersion) observed2 else observed,
    location = if (borrow_dispersion) location2 else location,
    age_group = if (borrow_dispersion) age_group2 else age_group,
    observed2 = observed2,
    location2 = location2,
    age_group2 = age_group2,
    type = switch(type, "additions" = "size", "revise_average" = "sd"),
    forecast_date = forecast_date,
    max_delay = max_delay,
    max_horizon = max_horizon,
    borrow_delays = borrow_delays,
    borrow_dispersion = borrow_dispersion,
    n_history_expectations = n_history_expectations,
    n_history_dispersion = n_history_dispersion,
    weekday_data_updates = weekday_data_updates
  )

  # set up data frame to store:
  df_all <- NULL

  # run through horizons:
  for (d in 1:max_horizon) {
    # get numeric horizon - only needed in creation of data.frame
    h <- horizons[d]

    # by how much do we need to shift quantiles upwards? Note that this needs to use index d
    already_observed <- sum(
      observed_as_of[nrow(observed_as_of) - d + 1, ],
      na.rm = TRUE
    )

    # data frame for expecations:
    weekday_data_updates_numeric <- weekday_as_number(weekday_data_updates)
    df_mean <- data.frame(
      location = location,
      age_group = age_group,
      forecast_date = forecast_date,
      target_end_date = forecast_date - weekday_data_updates_numeric + 7 * h,
      horizon = h,
      type = "mean",
      quantile = NA,
      value = round(mu[d] + already_observed)
    )

    # obtain quantiles:
    if (type == "additions") {
      qtls <- qnbinom(quantile_levels, size = disp_params[d], mu = mu[d])
    } else {
      qtls <- qnorm(quantile_levels, sd = disp_params[d], mean = mu[d])
    }

    # data.frame for quantiles:
    df_qtls <- data.frame(
      location = location,
      age_group = age_group,
      forecast_date = forecast_date,
      target_end_date = forecast_date - weekday_data_updates_numeric + 7 * h,
      horizon = h,
      type = "quantile",
      quantile = quantile_levels,
      value = qtls
    )

    # join:
    df <- rbind(df_mean, df_qtls)

    # add to results from other horizons
    if (is.null(df_all)) {
      df_all <- df
    } else {
      df_all <- rbind(df_all, df)
    }
  }

  # return
  return(list(result = df_all, mu = mu, size_params = disp_params)) # ,
  # expectation_to_add_already_observed = expectation_to_add_already_observed,
  # to_add_already_observed = to_add_already_observed))
}

#' Function to fit the overdispersion parameter used in compute_forecast
#' arguments as in compute_forecast
fit_dispersion_forecast <- function(
  observed,
  location,
  age_group,
  observed2 = NULL,
  location2 = NULL,
  age_group2 = NULL,
  type = "size",
  forecast_date,
  max_delay,
  max_horizon,
  borrow_delays = FALSE,
  borrow_dispersion = FALSE,
  n_history_expectations,
  n_history_dispersion,
  weekday_data_updates = "Thursday"
) {
  if (borrow_delays) {
    # catch missing observed2
    if (is.null(observed2)) stop("observed2 is needed if borrow_delays == TRUE")
  } else {
    if (!is.null(observed2)) {
      warning("observed2 is not used as borrow_delays == FALSE")
    }
    observed2 <- observed
  }

  if (!type %in% c("size", "sd")) {
    stop("type needs to be either size or sd.")
  }

  # if no observed2 is provided: use observed
  # having an object observed2 in any case makes the code simpler in the following
  if (is.null(observed2)) {
    observed2 <- observed
  }

  # bring reporting triangles to state as of forecast date:
  matr_observed <- data_as_of(
    observed,
    age_group = age_group,
    location = location,
    date = forecast_date,
    weekday_data_updates = weekday_data_updates,
    max_lag = max_delay,
    return_matrix = FALSE
  )
  # pad matr_observed with NAs if too short:
  matr_observed <- pad_matr(
    matr_observed,
    n_history_expectations + n_history_dispersion
  )
  matr_observed2 <- data_as_of(
    observed2,
    age_group = age_group2,
    location = location2,
    date = forecast_date,
    weekday_data_updates = weekday_data_updates,
    max_lag = max_delay,
    return_matrix = FALSE
  )
  # no padding here as matr_observed2 cannot be too short.

  all_forecast_dates <- seq(
    from = forecast_date - 7 * (n_history_dispersion + max_delay + max_horizon),
    by = 7,
    to = forecast_date - 7 * (max_delay + max_horizon)
  ) # exclude actual forecast date
  weekday_data_updates_numeric <- weekday_as_number(weekday_data_updates)

  all_target_dates0 <- all_forecast_dates - weekday_data_updates_numeric

  cols_value <- paste0("value_", 0:max_delay, "w")
  observed_total <- rowSums(matr_observed[
    which(matr_observed$date %in% all_target_dates0),
    cols_value,
    drop = FALSE
  ])
  point_nowcasts <- numeric(length(all_forecast_dates))

  # run through forecast dates to generate point nowcasts and corresponding observations:
  for (t in seq_along(all_forecast_dates)) {
    # identify date for which to compute retrospective nowcast
    forecast_date_temp <- all_forecast_dates[t]
    # bring to state of forecast_date_temp, subset to location and age group:
    matr_observed_temp <- data_as_of(
      observed,
      age_group = age_group,
      location = location,
      date = forecast_date_temp,
      weekday_data_updates = weekday_data_updates,
      max_lag = max_delay,
      return_matrix = TRUE
    )
    matr_observed2_temp <- data_as_of(
      observed2,
      age_group = age_group2,
      location = location2,
      date = forecast_date_temp,
      weekday_data_updates = weekday_data_updates,
      max_lag = max_delay,
      return_matrix = TRUE
    )

    # pad and catch case where matr_observed does not contain any data
    if (nrow(matr_observed_temp) > 0) {
      matr_observed_temp <- pad_matr(matr_observed_temp, n_history_expectations)
    } else {
      matr_observed_temp <- NA * matr_observed2_temp
    }

    # get same subset of the reporting triangle, but filled as far as possible at forecast_date:
    matr_observed_temp_full <- matr_observed[
      which(
        rownames(matr_observed) %in%
          tail(rownames(matr_observed_temp), n_history_expectations)
      ),
    ]
    # this is needed to estimate dispersion parameters below

    # generate retrospective point nowcast:
    point_forecasts_temp <- compute_expectations(
      observed = matr_observed_temp,
      observed2 = matr_observed2_temp,
      n_history = n_history_expectations,
      borrow_delays = borrow_delays,
      remove_observed = FALSE
    )
    point_nowcasts[t] <- rowSums(point_forecasts_temp[
      nrow(point_forecasts_temp),
    ])
  }

  # estimate dispersion
  disp_params <- numeric(max_horizon)
  lgt <- length(observed_total)

  # run through horizons
  for (i in 1:max_horizon) {
    obs_temp <- observed_total[(i + 1):lgt]
    mu_temp <- point_nowcasts[1:(lgt - i)]

    # remove entries with zero initial reports (Christmas etc)
    obs_temp <- obs_temp[mu_temp >= 1]
    mu_temp <- mu_temp[mu_temp >= 1]

    if (type == "size") {
      mu_temp <- mu_temp + 0.1
      # plus 0.1 to avoid ill-defined negative binomial
      disp_params[i] <- fit_nb(x = obs_temp, mu = mu_temp)
    } else {
      disp_params[i] <- sd(obs_temp - mu_temp)
    }
  }

  return(disp_params)
}
