import pandas as pd

from src.load_data import extract_info


def reshape_truth(y):
    """
    Reformat timeseries so prediciton bands can start at the last known value at each forecast date.
    """
    source = y.components[0].split("-")[0]
    indicator = y.components[0].split("-")[1]

    y = y.to_dataframe()
    y = y.reset_index().melt(id_vars="date", var_name="component")

    # y['strata']   = y.component.apply(lambda x: x.split('-', 2)[-1].split('_')[0])
    y["strata"] = y.component.apply(lambda x: x.replace(f"{source}-{indicator}-", "").split("_")[0])
    y[["location", "age_group"]] = y.apply(extract_info, axis=1)

    for q in [
        "quantile_0.025",
        "quantile_0.25",
        "quantile_0.5",
        "quantile_0.75",
        "quantile_0.975",
    ]:
        y[q] = y.value

    y["type"] = "truth"
    y["horizon"] = 0
    y = y.rename(columns={"date": "target_end_date"})
    y["forecast_date"] = y.target_end_date + pd.Timedelta(days=4)
    y = y.drop(columns=["component", "strata", "value"])

    return y


def prepare_plot_data(df, y):
    """
    Transform dataframe to wide format and add truth data for plotting.
    """
    df_plot = df.pivot(
        index=[
            "location",
            "age_group",
            "forecast_date",
            "target_end_date",
            "type",
            "horizon",
        ],
        columns="quantile",
        values="value",
    )

    df_plot.columns = ["quantile_" + str(q) for q in df_plot.columns]
    df_plot = df_plot.reset_index()

    y = reshape_truth(y)

    return pd.concat([df_plot, y], ignore_index=True)


def get_sundays(start_date, end_date):
    sundays = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
    return sundays
