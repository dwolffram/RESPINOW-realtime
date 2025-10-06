from pathlib import Path

import pandas as pd
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from config import MODEL_NAMES, QUANTILES, ROOT

# Explicit mapping from Darts rounded quantile labels to our desired values
Q_MAP = {
    "q0.03": "0.025",
    "q0.10": "0.1",
    "q0.25": "0.25",
    "q0.50": "0.5",
    "q0.75": "0.75",
    "q0.90": "0.9",
    "q0.97": "0.975",
}


def extract_info(row):
    """
    Splits the info of the 'strata' column into 'location' and 'age_group'.
    """
    if any(char.isdigit() for char in row["strata"]):
        location = "DE"
        age_group = row["strata"]
    else:
        location = row["strata"]
        age_group = "00+"
    return pd.Series([location, age_group])


def reshape_forecast(ts_forecast, nowcast=False, deterministic=False):
    """
    Transforms a forecast from TimeSeries format to Hub format.
    """

    if deterministic:
        df_temp = ts_forecast.pd_dataframe().reset_index().melt(id_vars="date")
    else:
        df_temp = ts_forecast.quantile(q=QUANTILES).to_dataframe()
        df_temp = df_temp.reset_index().melt(id_vars="date", var_name="component")
        df_temp["quantile"] = df_temp["component"].str.rsplit("_", n=1).str[-1].map(Q_MAP)

    source = ts_forecast.components[0].split("-")[0]
    indicator = ts_forecast.components[0].split("-")[1]

    df_temp["strata"] = df_temp.component.apply(lambda x: x.replace(f"{source}-{indicator}-", "").split("_")[0])
    df_temp[["location", "age_group"]] = df_temp.apply(extract_info, axis=1)

    df_temp["horizon"] = df_temp.date.rank(method="dense").astype(int)
    if nowcast:
        df_temp.horizon = df_temp.horizon - df_temp.horizon.max()
        df_temp["forecast_date"] = df_temp.date.max() + pd.Timedelta(days=4)
    else:
        df_temp["forecast_date"] = df_temp.date.min() - pd.Timedelta(days=3)

    if deterministic:
        df_temp = df_temp.loc[df_temp.index.repeat(len(QUANTILES))].reset_index(drop=True)
        df_temp["quantile"] = pd.Series(QUANTILES * len(df_temp)).astype("str")

    df_temp["type"] = "quantile"
    df_temp = df_temp.rename(columns={"date": "target_end_date"})

    return df_temp[
        [
            "location",
            "age_group",
            "forecast_date",
            "target_end_date",
            "horizon",
            "type",
            "quantile",
            "value",
        ]
    ]


def filter_by_level(df, level):
    if level == "national":
        df = df[(df["location"] == "DE") & (df["age_group"] == "00+")]
    elif level == "states":
        df = df[(df["location"] != "DE")]
    elif level == "age":
        df = df[(df["location"] == "DE") & (df["age_group"] != "00+")]
    return df


def add_median(df):
    df_median = df[df["quantile"] == 0.5].copy()
    df_median["type"] = "median"
    return pd.concat([df, df_median], ignore_index=True)


def add_truth(df, source="icosari", disease="sari", target=False):
    if target:
        df_truth = pd.read_csv(ROOT / f"data/target-{source}-{disease}.csv")
    else:
        df_truth = pd.read_csv(ROOT / f"data/latest_data-{source}-{disease}.csv")

    df_truth = df_truth.rename(columns={"value": "truth"})

    df = df.merge(
        df_truth,
        how="left",
        left_on=["location", "age_group", "target_end_date"],
        right_on=["location", "age_group", "date"],
    )
    return df


def load_predictions(
    models=None,
    start="2023-11-16",
    end="2024-09-12",
    exclude_christmas=True,
    include_median=True,
    include_truth=True,
    target=True,
):
    path_forecasts = ROOT / "forecasts"
    files = [f for f in path_forecasts.rglob("*.csv") if "checkpoint" not in f.name]

    df = pd.concat(
        (pd.read_csv(f).assign(model=f.stem.split("-", 5)[-1]) for f in files),
        ignore_index=True,
    )

    if include_median:
        df = add_median(df)
    if include_truth:
        df = add_truth(df, source="icosari", disease="sari", target=target)
    if exclude_christmas:
        df = df[df.forecast_date != "2023-12-28"]

    df.model = df.model.replace(MODEL_NAMES)

    if models is None:
        models = MODEL_NAMES.values()
    df = df[df.model.isin(models)]

    return df[df.forecast_date.between(start, end)].reset_index(drop=True)


def load_nowcasts(
    start="2023-11-16",
    end="2024-09-12",
    include_truth=True,
    exclude_christmas=True,
    quantiles=None,
):
    path_nowcasts = Path.cwd().parent / "nowcasts" / "simple_nowcast"
    files = list(path_nowcasts.rglob("*.csv"))

    df = pd.concat(
        (pd.read_csv(f).assign(model="Nowcast") for f in files),
        ignore_index=True,
    )

    if exclude_christmas:
        df = df[df.forecast_date != "2023-12-28"]

    df = df[df.forecast_date.between(start, end)].reset_index(drop=True)

    if include_truth:
        df = add_truth(df, target=True)

    if quantiles is not None:
        df = df[df["quantile"].isin(quantiles)]

    return df


def encode_static_covariates(ts, ordinal=False):
    ts.static_covariates.drop(columns=["source", "target", "location"], inplace=True, errors="ignore")

    # Use OneHotEncoder per default, use OrdinalEncoder if 'ordinal' is True
    scaler = StaticCovariatesTransformer(transformer_cat=OrdinalEncoder() if ordinal else OneHotEncoder())
    ts = scaler.fit_transform(ts)
    return ts
