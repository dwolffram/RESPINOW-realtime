# Purpose:
# - When opening the whole repository in Positron (or RStudio),
#   make sure the R environment from ./r (managed by renv) is activated automatically.
#   Without this, renv would only activate if you opened r/sari-forecasting.Rproj directly.
#
# - At the same time, anchor the `here` package to the *repository root*
#   so that paths like here("data", ...) always resolve to ./data, ./figures, etc.,
#   regardless of whether you open the repo root or the r/ subproject.
#
# Together, this ensures that both Python and R can share the same data/figures/forecasts/nowcasts
# folders at the repo root without having to adjust relative paths.

message("Activating renv project at ./r")
renv::load("r")

library(here)
here::i_am(".Rprofile")
