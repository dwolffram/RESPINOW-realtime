## Files and their purpose

### `/auxiliary`

Auxilary files:

-   `forecast_dates.csv` A csv file containing the dates for which nowcasts and forecasts are to be generated.

### `/hhh4`

Files to run different versions of the hhh4 model. Shared functionality is implemented in the following files:

-   `functions_hhh4.R`: Helper functions for hhh4 models.
-   `setup_hhh4.R`: Code shared by all hhh4 versions, defining global settings and reading in data.
-   `core_hhh4.R` Generic code which is sourced with different settings by the following files. Note: these all read in nowcasts generated in `nowcasting.R`.
    -   `hhh4_default.R`: Model with default settings.
    -   `hhh4_exclude_covid.R`: Model fitted to data without CVOID period, otherwise default settings.
    -   `hhh4_shuffle`: Using shuffled rather than ordered nowcast paths, otherwise default settings.
    -   `hhh4-skip`: Removing the last observation, otherwise default settings.
    -   `hhh4-vincentization`: Using vincentization / quantile average rather than linear pool to aggregate forecasts from different nowast paths.
-   The following versions of the model do not fit the framework from `core_hhh4.R` and have their own standalone files:
    -   `hhh4_naive`: Model fitted naively to data as available in real time, wthout any correction.
    -   `hhh4_oracle.R`: Model fitted to final data up to a certain time point (which is not possible in real time).

### `/nowcasting`

Files to run nowcasting:

-   `functions_nowcasting.R`: Functions implementing nowcasting and a few helper functions.
-   `nowcasting.R`: Generate nowcasts.

### `/outputs`

Results are stored in this folder (within subfolders per model).

### `/persistence`

Files to run the persistence forecast.

-   `functions_persistence.R`: Functions implementing the persistence forecast and a few helper functions. Depends on `functions_nowcasting.R`.
-   `persistence.R`: Generate persistence forecasts.

### `/tscount`

Files to generate forecasts inspired by Agosto et al, using the tscount package.

-   `core_hhh4.R` Generic code which is sourced with different settings by the following files. Note: these all read in nowcasts generated in `nowcasting.R`.
    -   `tscount_simple`: Simple Poisson model as suggested by Agosto et al.
    -   `tscount_extended`: Extension of the model by Agosto et al., with negative binomial assumption and seasonality.