# Helper functions to apply the hhh4 model
# Author: Johannes Bracher, johannes.bracher@kit.edu

#' Function to compute (approximate) quantiles from matrices of means and variances for different nowcast paths
#' wraps around linear_pool and quantile_average
#' @param mu_matrix The matrix of means (paths per nowcast quantile level in columns, horizons in rows)
#' @param var_matrix The matrix of variances (paths per nowcast quantile level in columns), horizons in rows)
#' @param quantile_levels_nowcast The nowcast quantile levels corresponding to the columns
#' @param quantile_levels The forecast quantile levels to be generated
#' @param type The type of aggregation to be performed: "linear pool" or "quantile average"
aggregate_forecasts_to_quantiles <- function(
  mu_matrix,
  var_matrix,
  quantile_levels_nowcast,
  quantile_levels,
  type = "linear pool"
) {
  if (type == "linear pool") {
    return(linear_pool(
      mu_matrix = mu_matrix,
      var_matrix = var_matrix,
      quantile_levels_nowcast = quantile_levels_nowcast,
      quantile_levels = quantile_levels
    ))
  }

  if (type == "quantile average") {
    return(quantile_average(
      mu_matrix = mu_matrix,
      var_matrix = var_matrix,
      quantile_levels_nowcast = quantile_levels_nowcast,
      quantile_levels = quantile_levels
    ))
  }

  if (!type %in% c("linear pool", "quantile_average")) {
    stop("type needs to be 'linear pool' or 'quantile average'")
  }
}

#' Function to compute (approximate) quantiles of a linear pool from matrices of
#'  means and variances for different nowcast paths
#'  Arguments as in aggregate_forecasts_to_quantiles
linear_pool <- function(
  mu_matrix,
  var_matrix,
  quantile_levels_nowcast,
  quantile_levels
) {
  # get quantiles via a negative binomial approximation based on moments, format and store:
  support <- seq(from = 0, to = 3 * max(mu_matrix), by = 10)
  horizons <- 1:nrow(mu_matrix)
  quantile_matrix <- matrix(
    nrow = length(horizons),
    ncol = length(quantile_levels)
  )

  for (h in horizons) {
    # a matrix to store CDFs
    cdf_matrix <- matrix(
      NA,
      ncol = length(quantile_levels_nowcast),
      nrow = length(support)
    )
    for (k in 1:ncol(cdf_matrix)) {
      # extract moments from matrices where they are stored
      mu <- mu_matrix[h, k]
      sigma2 <- var_matrix[h, k]
      size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)
      # fill in CDF:
      cdf_matrix[, k] <- pnbinom(support, mu = mu, size = size)
    }
    # aggregate CDFs:
    cdf <- rowMeans(cdf_matrix)

    # extract quantiles:
    for (k in seq_along(quantile_levels)) {
      quantile_matrix[h, k] <- max(support[cdf <= quantile_levels[k]])
    }
  }
  return(quantile_matrix)
}

#' Function to compute (approximate) quantile averages from matrices of
#'  means and variances for different nowcast paths
#'  Arguments as in aggregate_forecasts_to_quantiles
quantile_average <- function(
  mu_matrix,
  var_matrix,
  quantile_levels_nowcast,
  quantile_levels
) {
  horizons <- 1:nrow(mu_matrix)
  quantile_matrix <- matrix(
    nrow = length(horizons),
    ncol = length(quantile_levels)
  )

  # get CDFs
  for (h in horizons) {
    quantile_matrix_h <- matrix(
      NA,
      ncol = length(quantile_levels_nowcast),
      nrow = length(quantile_levels)
    )
    for (k in 1:ncol(quantile_matrix_h)) {
      # compute quantiles via NegBin approximation:
      mu <- mu_matrix[h, k]
      sigma2 <- var_matrix[h, k]
      size <- pmin(abs(mu / (sigma2 / mu - 1)), 10000)
      quantile_matrix_h[, k] <- qnbinom(quantile_levels, mu = mu, size = size)
      # aggregate quantiles:
      quantile_matrix[h, ] <- rowMeans(quantile_matrix_h)
    }
  }

  return(quantile_matrix)
}


#' function to shuffle quantiles in nowcast paths (sensitivity analysis required by reviewer)
#' @param nc the nowcast in standard data.frame format
shuffle_quantiles <- function(nc) {
  # determine horizons, age_groups and locations:
  horizons <- unique(nc$horizon)
  age_groups <- unique(nc$age_group)
  locations <- unique(nc$location)

  # run through all combinations and shuffle the respective quantiles:
  for (h in horizons) {
    for (ag in age_groups) {
      for (loc in locations) {
        # identify relevant rows:
        inds <- which(
          nc$horizon == h &
            nc$location == loc &
            nc$age_group == ag &
            nc$type == "quantile"
        )
        # permute them:
        inds_resampled <- sample(inds, size = length(inds), replace = FALSE)
        nc$value[inds] <- nc$value[inds_resampled]
      }
    }
  }
  return(nc)
}

#' Function to bring forecasts into the standard output format
#' @param quantile_matrix a quantile_matrix as returned by aggregate_forecasts_to_quantiles
#' @param means a vector of predictive means
#' @param forecast_date the forecast date (note: just one; function to be used inside loop)
#' @param target_end_dates vector of target end dates (corresponding to rows in quantile_matrix)
#' @param the location
#' @param age_group the age group
#' @param horizons the horizons (corresponding to rows in quantile_matrix)
#' @param quantile_levels the quantile levels (corresponding to columns in quantile_matrix)
format_forecasts <- function(
  quantile_matrix,
  means,
  forecast_date,
  target_end_dates,
  location,
  age_group,
  horizons,
  quantile_levels
) {
  # format quantiles:
  n_quantile_levels <- length(quantile_levels)

  # place holder to join results across horizons:
  all_formatted <- NULL

  for (i in seq_along(horizons)) {
    h <- horizons[i]
    target_end_date <- target_end_dates[i]
    # fill data.frame with quantiles:
    quantiles_formatted_h <- data.frame(
      location = rep(location, n_quantile_levels),
      age_group = rep(age_group, n_quantile_levels),
      forecast_date = rep(forecast_date, n_quantile_levels),
      target_end_date = rep(target_end_dates[i], n_quantile_levels),
      horizon = rep(h, each = n_quantile_levels),
      type = "quantile",
      quantile = quantile_levels,
      value = quantile_matrix[i, ]
    )

    # add mean:
    mean_formatted_h <- quantiles_formatted_h[1, ]
    mean_formatted_h$type <- "mean"
    mean_formatted_h$quantile <- NA
    mean_formatted_h$value <- means[i]

    # append:
    all_formatted <- rbind(
      all_formatted,
      quantiles_formatted_h,
      mean_formatted_h
    )
  }

  return(all_formatted)
}
