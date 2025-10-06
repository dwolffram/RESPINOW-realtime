from src.load_data import filter_by_level


# Quantile score function
def quantile_score(q, y, alpha):
    return 2 * (int(y < q) - alpha) * (q - y)


# Compute squared error, absolute error or quantile score based on "type"
def score(prediction, observation, score_type, quantile):
    if score_type == "mean":
        return (prediction - observation) ** 2
    elif score_type == "median":
        return abs(prediction - observation)
    elif score_type == "quantile":
        return quantile_score(prediction, observation, quantile)
    else:
        raise ValueError("Invalid type specified")


def compute_row_score(row):
    return round(score(row["value"], row["truth"], row["type"], row["quantile"]), 5)


# Compute scores for each row in a dataframe
def compute_scores(df):
    df["score"] = df.apply(compute_row_score, axis=1)
    return df.drop(columns=["value", "truth"])


# Compute WIS decomposition
def compute_wis(df):
    # Filter rows where 'quantile' is 0.5, rename 'value' to 'med', and drop unnecessary columns
    df_median = df[df["quantile"] == 0.5].copy()
    df_median = df_median.rename(columns={"value": "med"}).drop(
        columns=["quantile", "pathogen", "retrospective", "truth"], errors="ignore"
    )

    # Filter rows where 'type' is 'quantile' and merge with df_median
    df_quantile = df[df["type"] == "quantile"].copy()
    df = df_quantile.merge(df_median, how="left")

    # Compute scores and other metrics row-wise
    df["wis"] = df.apply(
        lambda row: score(row["value"], row["truth"], row["type"], row["quantile"]),
        axis=1,
    )
    df["spread"] = df.apply(
        lambda row: score(row["value"], row["med"], row["type"], row["quantile"]),
        axis=1,
    )
    df["overprediction"] = df.apply(
        lambda row: row["wis"] - row["spread"] if row["med"] > row["truth"] else 0,
        axis=1,
    )
    df["underprediction"] = df.apply(
        lambda row: row["wis"] - row["spread"] if row["med"] < row["truth"] else 0,
        axis=1,
    )

    # Group by 'model' and compute the mean for each metric
    result_df = (
        df.groupby("model")
        .agg(
            {
                "spread": "mean",
                "overprediction": "mean",
                "underprediction": "mean",
                "wis": "mean",
            }
        )
        .reset_index()
    )

    return result_df


def compute_coverage(df):
    df_wide = df[df.type == "quantile"].pivot(
        index=[
            "location",
            "age_group",
            "forecast_date",
            "target_end_date",
            "horizon",
            "type",
            "model",
            "date",
            "year",
            "week",
            "truth",
        ],
        columns="quantile",
        values="value",
    )

    df_wide.columns = [f"quantile_{col}" for col in df_wide.columns]

    df_wide = df_wide.reset_index()

    df_wide["c50"] = (df_wide["truth"] >= df_wide["quantile_0.25"]) & (df_wide["truth"] <= df_wide["quantile_0.75"])
    df_wide["c95"] = (df_wide["truth"] >= df_wide["quantile_0.025"]) & (df_wide["truth"] <= df_wide["quantile_0.975"])

    coverage_df = df_wide.groupby("model").agg(c50=("c50", "mean"), c95=("c95", "mean")).reset_index()

    return coverage_df


def compute_ae(df):
    df_ae = df[df["quantile"] == 0.5].copy()
    df_ae["ae"] = abs(df_ae.value - df_ae.truth)
    return df_ae.groupby("model").agg({"ae": "mean"}).reset_index()


def evaluate_models(df, level="national", by_horizon=False, by_age=False):
    df_temp = filter_by_level(df, level)
    if by_horizon:
        wis_temp = df_temp.groupby("horizon")[df_temp.columns].apply(compute_wis).reset_index().drop(columns="level_1")
        ae_temp = df_temp.groupby("horizon")[df_temp.columns].apply(compute_ae).reset_index().drop(columns="level_1")
        coverage_temp = (
            df_temp.groupby("horizon")[df_temp.columns].apply(compute_coverage).reset_index().drop(columns="level_1")
        )
        results = (
            wis_temp.merge(ae_temp, on=["model", "horizon"])
            .merge(coverage_temp, on=["model", "horizon"])
            .sort_values(["horizon", "wis"], ignore_index=True)
        )
    elif by_age:
        wis_temp = (
            df_temp.groupby("age_group")[df_temp.columns].apply(compute_wis).reset_index().drop(columns="level_1")
        )
        ae_temp = df_temp.groupby("age_group")[df_temp.columns].apply(compute_ae).reset_index().drop(columns="level_1")
        coverage_temp = (
            df_temp.groupby("age_group")[df_temp.columns].apply(compute_coverage).reset_index().drop(columns="level_1")
        )
        results = (
            wis_temp.merge(ae_temp, on=["model", "age_group"])
            .merge(coverage_temp, on=["model", "age_group"])
            .sort_values(["age_group", "wis"], ignore_index=True)
        )
    else:
        wis_temp = compute_wis(df_temp)
        ae_temp = compute_ae(df_temp)
        coverage_temp = compute_coverage(df_temp)
        results = (
            wis_temp.merge(ae_temp, on="model").merge(coverage_temp, on="model").sort_values("wis", ignore_index=True)
        )
    return results
