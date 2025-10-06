# Purpose:
# - When opening the R subproject directly (e.g. sari-forecasting.Rproj in RStudio),
#   make sure the renv environment in this folder is activated automatically.
#
# - At the same time, anchor the `here` package to the *repository root* (../)
#   so that paths like here("data", ...) always resolve to ./data, ./figures, etc.,
#   regardless of whether you open the repo root or the r/ subproject.
#
# Together, this ensures that both Python and R can share the same data/figures/forecasts/nowcasts
# folders at the repo root without having to adjust relative paths.

source("renv/activate.R")

library(here)
here::i_am("r/.Rprofile")
