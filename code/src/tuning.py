import ast

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import TSMixerModel
from epiweeks import Week

from config import ALLOWED_MODELS, OPTIMIZER_DICT, ROOT, ModelName


def compute_validation_score(
    model,
    targets_train,
    targets_validation,
    covariates,
    horizon,
    num_samples,
    metric,
    metric_kwargs,
    enable_optimization=True,
    sample_weight=None,
):
    # torch model: add dataloader_kwargs
    if isinstance(model, TSMixerModel):
        model.fit(
            targets_train,
            past_covariates=covariates,
            sample_weight=sample_weight,
            dataloader_kwargs={"pin_memory": torch.cuda.is_available()},
        )

    else:
        model.fit(targets_train, past_covariates=covariates, sample_weight=sample_weight)

    if isinstance(targets_train, list):
        validation_start = targets_train[0].end_time() + targets_train[0].freq
    else:
        validation_start = targets_train.end_time() + targets_train.freq

    scores = model.backtest(
        series=targets_validation,
        past_covariates=covariates,
        start=validation_start,
        forecast_horizon=horizon,
        stride=1,
        last_points_only=False,
        retrain=False,
        verbose=False,
        num_samples=num_samples,
        metric=metric,
        metric_kwargs=metric_kwargs,
        enable_optimization=enable_optimization,
    )

    score = np.mean(scores)

    return score if not np.isnan(score) else float("inf")


def get_season_end(start_year):
    return pd.to_datetime(Week(start_year + 1, 39, system="iso").enddate())


def train_validation_split(series, validation_year):
    validation_end = get_season_end(validation_year)
    train_end = get_season_end(validation_year - 1)

    ts_validation = series[:validation_end]
    ts_train = series[:train_end]

    return ts_train, ts_validation


def exclude_covid_weights(targets):
    exclusion_start = pd.Timestamp("2019-06-30")
    exclusion_end = pd.Timestamp("2023-07-03")

    # Linear weights for the entire time range
    total_duration = len(targets.time_index)
    weights = np.linspace(0, 1, total_duration)

    # Adjust for exclusion period: Set weights to 0 during the exclusion period
    weights = np.where(
        (targets.time_index >= exclusion_start) & (targets.time_index <= exclusion_end),
        0,  # Weight is 0 during the exclusion period
        weights,  # Linear increase otherwise
    )

    # Create TimeSeries object for weights
    ts_weights = TimeSeries.from_times_and_values(times=targets.time_index, values=weights)

    return ts_weights


def get_best_parameters(
    model: ModelName,
    use_covariates: bool | None = None,
    sample_weight: str | None = None,
    clean: bool = False,
    return_score: bool = False,
) -> dict | tuple:
    """
    Loads a gridsearch CSV and returns the configuration with the lowest WIS.
    Optionally filters by covariates and sample weight, parses covariate columns,
    and drops error columns.

    Args:
        model (ModelName): Model name used to construct the gridsearch CSV file path.
        use_covariates (bool | None): Filter rows by this value if specified.
        sample_weight (str | None): Filter rows by this value if specified.
        clean (bool): If True, strips helper keys and normalizes params
                      (ready for model init/fit).
        return_score (bool): If True, also return the validation score (WIS). Defaults to False.
    Returns:
        dict | tuple[dict, float]: The best parameter configuration. If
        `return_score` is True, returns `(params, wis)` where `wis` is the
        validation score.
    """
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model: {model}")

    gs = pd.read_csv(ROOT / "results" / "tuning" / f"gridsearch_{model}.csv")

    # Optional filtering
    if use_covariates is not None and "use_covariates" in gs.columns:
        gs = gs[gs["use_covariates"] == use_covariates]
    if sample_weight is not None and "sample_weight" in gs.columns:
        gs = gs[gs["sample_weight"] == sample_weight]

    # Convert string representations of covariate lags back into Python objects
    for col in ["lags_past_covariates", "lags_future_covariates"]:
        if col in gs.columns:
            gs[col] = gs[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Drop error columns if present
    gs = gs.drop(columns=[c for c in ["error_flag", "error_msg"] if c in gs.columns])

    # Find row with lowest WIS
    best_row = gs.loc[gs["WIS"].idxmin()].to_dict()
    wis = best_row.pop("WIS")

    # Remove extra WIS columns if present
    for key in ["WIS_1", "WIS_2", "WIS_3", "WIS_std"]:
        best_row.pop(key, None)

    if clean:
        # Drop lag params if covariates were disabled
        if best_row.get("use_covariates") is False:
            best_row.pop("lags_past_covariates", None)

        # Remove meta flags
        # Some keys aren't present for both model families; pop(..., None) is safe
        for k in ("use_covariates", "sample_weight", "model", "use_features"):
            best_row.pop(k, None)

        # Normalize optimizer fields
        if "optimizer" in best_row:
            optimizer = best_row.pop("optimizer")
            best_row["optimizer_cls"] = OPTIMIZER_DICT[optimizer]
            best_row["optimizer_kwargs"] = {
                "lr": best_row.pop("optimizer_kwargs.lr", None),
                "weight_decay": best_row.pop("optimizer_kwargs.weight_decay", None),
            }

    params = {k: best_row[k] for k in sorted(best_row)}

    return (params, float(wis)) if return_score else params
