import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.utils.ts_utils import retain_period_common_to_all

from config import ROOT, SOURCE_DICT


def load_latest_series(indicator="sari"):
    source = SOURCE_DICT[indicator]

    ts = pd.read_csv(ROOT / f"data/latest_data-{source}-{indicator}.csv")

    ts = ts[ts.location == "DE"]

    ts = TimeSeries.from_group_dataframe(
        ts,
        group_cols=["age_group"],
        time_col="date",
        value_cols="value",
        freq="7D",
        fillna_value=0,
    )
    ts = concatenate(ts, axis=1)
    ts = ts.with_columns_renamed(
        ts.static_covariates.age_group.index,
        f"{source}-{indicator}-" + ts.static_covariates.age_group,
    )
    ts = ts.with_columns_renamed(f"{source}-{indicator}-00+", f"{source}-{indicator}-DE")

    return ts


def load_target_series(indicator="sari", as_of=None, age_group=None):
    source = SOURCE_DICT[indicator]

    if as_of is None:
        target = pd.read_csv(ROOT / f"data/target-{source}-{indicator}.csv")
    else:
        rt = load_rt(indicator)
        target = target_as_of(rt, as_of)

    target = target[target.location == "DE"]

    if age_group is not None:
        target = target[target.age_group == age_group]

    ts_target = TimeSeries.from_group_dataframe(
        target,
        group_cols=["age_group"],
        time_col="date",
        value_cols="value",
        freq="7D",
        fillna_value=0,
    )
    ts_target = concatenate(
        retain_period_common_to_all(ts_target), axis=1
    )  # all components start at the same time (SARI!)
    ts_target = ts_target.with_columns_renamed(
        ts_target.static_covariates.age_group.index,
        f"{source}-{indicator}-" + ts_target.static_covariates.age_group,
    )

    if age_group is None or age_group == "00+":
        ts_target = ts_target.with_columns_renamed(f"{source}-{indicator}-00+", f"{source}-{indicator}-DE")

    return ts_target


def load_nowcast(
    forecast_date,
    probabilistic=True,
    indicator="sari",
    local=True,
    model="simple_nowcast",
):
    source = SOURCE_DICT[indicator]

    if local:
        filepath = (
            ROOT
            / f"{f'nowcasts/{model}' if indicator == 'sari' else '../ari/nowcasts'}/{forecast_date}-{source}-{indicator}-{model}.csv"
        )
    else:
        filepath = f"https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/refs/heads/main/submissions/{source}/{indicator}/KIT-{model}/{forecast_date}-{source}-{indicator}-KIT-{model}.csv"
    df = pd.read_csv(filepath)
    df = df[(df.location == "DE") & (df.type == "quantile") & (df.horizon >= -3)]
    df = df.rename(columns={"target_end_date": "date"})
    df = df.sort_values(["location", "age_group"], ignore_index=True)

    if not probabilistic:
        df = df[df["quantile"] == 0.5]

    all_nowcasts = []
    for age in df.age_group.unique():
        # print(age)
        df_temp = df[df.age_group == age]

        # transform nowcast into a TimeSeries object
        nowcast_age = TimeSeries.from_group_dataframe(
            df_temp,
            group_cols=["age_group", "quantile"],
            time_col="date",
            value_cols="value",
            freq="7D",
            fillna_value=0,
        )

        nowcast_age = concatenate(nowcast_age, axis="sample")
        nowcast_age.static_covariates.drop(columns=["quantile"], inplace=True, errors="ignore")
        nowcast_age = nowcast_age.with_columns_renamed(nowcast_age.components, [f"{source}-{indicator}-" + age])

        all_nowcasts.append(nowcast_age)

    all_nowcasts = concatenate(all_nowcasts, axis="component")
    all_nowcasts = all_nowcasts.with_columns_renamed(f"{source}-{indicator}-00+", f"{source}-{indicator}-DE")

    return all_nowcasts


def make_target_paths(target_series, nowcast):
    """Cut known truth series and append nowcasted values."""

    # Only cut if nowcast.start_time is within the target_series
    if nowcast.start_time() <= target_series.end_time():
        target_temp = target_series.drop_after(nowcast.start_time())
    else:
        target_temp = target_series

    # every entry is a multivariate timeseries (one sample path for each age group)
    # there is one entry per quantile level
    target_list = [
        concatenate(
            [target_temp[age].append_values(nowcast[age].univariate_values(sample=i)) for age in nowcast.components],
            axis="component",
        )
        for i in range(nowcast.n_samples)
    ]

    return target_list


def load_rt(indicator="sari", preprocessed=False):
    """Load reporting triangle for a given indicator."""
    source = SOURCE_DICT[indicator]
    rt = pd.read_csv(
        ROOT / f"data/reporting_triangle-{source}-{indicator}{'-preprocessed' if preprocessed else ''}.csv",
        parse_dates=["date"],
    )

    return rt.loc[:, :"value_4w"]


def set_last_n_values_to_nan(group):
    for i in [1, 2, 3, 4]:  # Loop for value_1w, value_2w, ..., value_4w
        group.loc[group.index[-i:], f"value_{i}w"] = np.nan
    return group


def target_as_of(rt, date):
    """Return the target time series as it would have been known on the specified date."""
    date = pd.Timestamp(date)
    rt_temp = rt[rt.date <= date]

    # in column 'value_1w' the last entry is set to nan, in column 'value_2w' the last two entries, etc.
    rt_temp = (
        rt_temp.groupby(["location", "age_group"]).apply(set_last_n_values_to_nan, include_groups=False).reset_index()
    )
    rt_temp["value"] = rt_temp[["value_0w", "value_1w", "value_2w", "value_3w", "value_4w"]].sum(axis=1).astype(int)

    return rt_temp[["location", "age_group", "year", "week", "date", "value"]]


def get_preceding_thursday(date):
    """Returns the date of the preceding Thursday. If 'date' is itself a Thursday, 'date' is returned."""
    date = pd.Timestamp(date)  # to also accept dates given as strings
    return date - pd.Timedelta(days=(date.weekday() - 3) % 7)  # weekday of Thursday is 3


def load_realtime_training_data(as_of=None, drop_incomplete=True):
    # load sari data
    target_sari = load_target_series("sari", as_of)
    latest_sari = load_latest_series("sari")

    ts_sari = concatenate([latest_sari.drop_after(target_sari.start_time()), target_sari])

    # load are data
    target_are = load_target_series("are", as_of)
    latest_are = load_latest_series("are")

    ts_are = concatenate([latest_are.drop_after(target_are.start_time()), target_are])

    if drop_incomplete:
        return ts_sari[:-4], ts_are[:-4]  # only use complete data points

    else:
        return ts_sari, ts_are
