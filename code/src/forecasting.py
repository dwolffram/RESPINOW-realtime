from src.silence import silence

silence()

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
from darts import concatenate
from darts.models import LightGBMModel, TSMixerModel
from tqdm import tqdm

from config import (
    ALLOWED_DATA_MODES,
    ALLOWED_MODELS,
    ALLOWED_MODES,
    DATA_MODE_CONFIG,
    ENCODERS,
    FORECAST_DATES,
    HORIZON,
    NUM_SAMPLES,
    QUANTILES,
    RANDOM_SEEDS,
    ROOT,
    SHARED_ARGS,
    DataMode,
    Mode,
    ModelName,
)
from src.load_data import encode_static_covariates, reshape_forecast
from src.realtime_utils import (
    load_nowcast,
    load_realtime_training_data,
    make_target_paths,
)
from src.tuning import exclude_covid_weights, get_best_parameters

MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "tsmixer": TSMixerModel,
}


# core (pure, no I/O)
def compute_forecast(
    model,
    *,
    # preloaded inputs
    targets=None,  # as-of targets (for "naive", "coupling", "discard")
    covariates=None,  # as-of covariates
    ts_nowcast=None,  # preloaded nowcast (for "coupling", "discard")
    complete_targets=None,  # fully corrected truth (only for "oracle")
    # meta
    forecast_date=None,
    mode: Mode = "naive",
    # defaults
    horizon: int = HORIZON,
    num_samples: int = NUM_SAMPLES,
) -> pd.DataFrame:
    """
    Pure forecasting wrapper (no I/O).
      - 'naive': use uncorrected as-of targets/covariates as provided.
      - 'coupling': build sample-path targets from as-of targets + nowcast.
      - 'discard': like 'coupling' but drop the last data point
      - 'oracle': truncate fully corrected targets at forecast_date (no nowcast).
    """
    if mode not in ("naive", "coupling", "discard", "oracle"):
        raise ValueError("mode must be one of {'naive','coupling','discard','oracle'}.")

    if mode == "naive":
        series_for_model = targets
        covs_for_model = covariates

    elif mode in {"coupling", "discard"}:
        if targets is None or ts_nowcast is None:
            raise ValueError("coupling/discard require `targets` (as-of) and `ts_nowcast`.")
        target_list = make_target_paths(targets, ts_nowcast)
        target_list = [encode_static_covariates(t, ordinal=False) for t in target_list]
        if mode == "discard":
            target_list = [t[:-1] for t in target_list]  # discard last data point
        series_for_model = target_list
        covs_for_model = [covariates] * len(target_list) if covariates is not None else None

    else:  # "oracle"
        if complete_targets is None:
            raise ValueError("oracle requires `complete_targets` (fully corrected).")
        ts_cut = complete_targets[: pd.Timestamp(forecast_date)]
        ts_cut = encode_static_covariates(ts_cut, ordinal=False)
        series_for_model = ts_cut
        covs_for_model = covariates

    fct = model.predict(
        n=horizon,
        series=series_for_model,
        past_covariates=covs_for_model,
        num_samples=num_samples,
    )

    ts_forecast = concatenate(fct, axis="sample") if isinstance(fct, list) else fct
    df = reshape_forecast(ts_forecast)

    df["forecast_date"] = pd.Timestamp(forecast_date)
    if mode == "discard":
        df["horizon"] = df["horizon"] - 1
    return df


# helpers
def aggregate_runs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(dfs, ignore_index=True)
        .groupby(
            ["location", "age_group", "forecast_date", "target_end_date", "horizon", "type", "quantile"], as_index=False
        )["value"]
        .mean()
        .sort_values(["location", "age_group", "horizon", "quantile"])
    )


def save_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / filename, index=False)


def fit_model(model, targets, covariates, params, use_covariates, use_encoders, weights, seed):
    if model == "lightgbm":
        mdl = LightGBMModel(
            **params,
            output_chunk_length=HORIZON,
            add_encoders=ENCODERS if use_encoders else None,
            likelihood="quantile",
            quantiles=QUANTILES,
            verbose=-1,
            random_state=seed,
        )
        mdl.fit(
            targets,
            past_covariates=covariates if use_covariates else None,
            sample_weight=weights,
        )

    elif model == "tsmixer":
        mdl = TSMixerModel(
            **params,
            add_encoders=ENCODERS if use_encoders else None,
            **SHARED_ARGS,
            random_state=seed,
        )
        mdl.fit(
            targets,
            past_covariates=covariates if use_covariates else None,
            sample_weight=weights,
            dataloader_kwargs={
                "pin_memory": False,
                "num_workers": 0,
            },
        )
    return mdl


def print_training_config(
    *, model_name, use_covariates, sample_weight, modes, seeds, params, wis, forecast_dates
) -> None:
    print(
        f"\n=== Training config ===\n"
        f"  model          : {model_name}\n"
        f"  use_covariates : {use_covariates}\n"
        f"  sample_weight  : {sample_weight}\n"
        f"  modes          : {list(modes)}\n"
        f"  dates          : {min(forecast_dates)} → {max(forecast_dates)} (n={len(forecast_dates)})\n"
        f"  seeds          : {min(seeds)} → {max(seeds)} (n={len(seeds)})\n"
        f"=======================\n"
        f"  Parameters:"
    )
    for k, v in params.items():
        print(f"    {k}: {v}")
    print(f"\n  Validation score : {wis:.3f}\n=======================\n")


def generate_forecasts(
    model: ModelName,
    forecast_dates: Union[str, Sequence[str]] = FORECAST_DATES,
    *,
    modes: Union[Mode, Sequence[Mode]] = ("naive", "coupling", "discard", "oracle"),
    data_mode: DataMode = "all",
    seeds=RANDOM_SEEDS,
    save_models: bool = False,
) -> None:
    """
    Train and generate forecasts for one or many forecast dates.

    Validates inputs once, computes best hyperparameters once, optionally prints the
    configuration, then for each date:
      - loads training/as-of data,
      - trains a fresh model per seed,
      - runs the selected modes (naive/coupling/discard/oracle),
      - aggregates across seeds per mode,
      - exports one CSV per mode.

    Parameters
    ----------
    model : ModelName
        "lightgbm" or "tsmixer".
    forecast_dates : str or sequence[str], default=FORECAST_DATES
        One date (ISO) or a list of dates.
    modes : Mode or sequence[Mode], default=("naive","coupling","discard","oracle")
        Strategies for handling incomplete recent data.
    data_mode : DataMode, default="all"
        One of {"all", "no_covariates", "no_covid"}.
    seeds : sequence[int], default=RANDOM_SEEDS
        Random seeds for repeated training.
    save_models : bool, default=False
        If True, saves each trained model (clean=True) under ROOT/models/<date>/...
    verbose : bool, default=True
        If True, prints the training configuration once.

    Returns
    -------
    None
    """
    # ---- normalize + validate once
    if isinstance(forecast_dates, str):
        forecast_dates = [forecast_dates]
    modes = [modes] if isinstance(modes, str) else list(modes)

    if model not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model: {model!r}. Allowed values: {sorted(ALLOWED_MODELS)}")
    if any(m not in ALLOWED_MODES for m in modes):
        raise ValueError(f"Invalid mode(s): {modes!r}. Allowed values: {sorted(ALLOWED_MODES)}")
    if data_mode not in ALLOWED_DATA_MODES:
        raise ValueError(f"Invalid data_mode: {data_mode!r}. Allowed values: {sorted(ALLOWED_DATA_MODES)}")

    model_name = model if data_mode == "all" else f"{model}-{data_mode}"
    use_covariates, sample_weight = DATA_MODE_CONFIG[data_mode]

    # ---- compute best params once
    params, wis = get_best_parameters(
        model, use_covariates=use_covariates, sample_weight=sample_weight, clean=True, return_score=True
    )
    use_encoders = params.pop("use_encoders", False)

    print_training_config(
        model_name=model_name,
        use_covariates=use_covariates,
        sample_weight=sample_weight,
        modes=modes,
        seeds=seeds,
        params=params,
        wis=wis,
        forecast_dates=forecast_dates,
    )

    # Load complete targets once if any 'oracle' requested
    complete_targets = None
    if "oracle" in modes:
        complete_targets, _ = load_realtime_training_data()

    failed: List[Tuple[str, str]] = []

    # ---- main loop: per date
    for fd in forecast_dates:
        print(f"→ {fd}")
        try:
            # Training data (complete up to fd) + weights
            targets_train, covars_train = load_realtime_training_data(as_of=fd, drop_incomplete=True)
            weights = exclude_covid_weights(targets_train) if sample_weight == "no-covid" else sample_weight

            # As-of data (may include incomplete values)
            targets_asof, covars_asof = load_realtime_training_data(as_of=fd, drop_incomplete=False)

            # Nowcast per date for coupling/discard
            ts_now = None
            if any(m in ("coupling", "discard") for m in modes):
                ts_now = load_nowcast(forecast_date=fd)

            # Collect per-seed runs for each mode
            per_mode_runs: Dict[Mode, List[pd.DataFrame]] = {m: [] for m in modes}

            for seed in tqdm(seeds, desc=f"{fd}", leave=False):
                mdl = fit_model(model, targets_train, covars_train, params, use_covariates, use_encoders, weights, seed)

                if save_models:
                    model_path = ROOT / "models" / fd / f"{fd}-{model_name}-{seed}.pkl"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    mdl.save(str(model_path), clean=True)

                covars_for_predict = None if not getattr(mdl, "uses_past_covariates", True) else covars_asof

                if "naive" in modes:
                    per_mode_runs["naive"].append(
                        compute_forecast(
                            mdl, targets=targets_asof, covariates=covars_for_predict, forecast_date=fd, mode="naive"
                        )
                    )
                if "coupling" in modes:
                    per_mode_runs["coupling"].append(
                        compute_forecast(
                            mdl,
                            targets=targets_asof,
                            covariates=covars_for_predict,
                            ts_nowcast=ts_now,
                            forecast_date=fd,
                            mode="coupling",
                        )
                    )
                if "discard" in modes:
                    per_mode_runs["discard"].append(
                        compute_forecast(
                            mdl,
                            targets=targets_asof,
                            covariates=covars_for_predict,
                            ts_nowcast=ts_now,
                            forecast_date=fd,
                            mode="discard",
                        )
                    )
                if "oracle" in modes:
                    per_mode_runs["oracle"].append(
                        compute_forecast(
                            mdl,
                            complete_targets=complete_targets,
                            covariates=covars_for_predict,
                            forecast_date=fd,
                            mode="oracle",
                        )
                    )

            # Aggregate & export per mode
            for m in modes:
                df = aggregate_runs(per_mode_runs[m])
                out_dir = ROOT / "forecasts" / f"{model_name}-{m}"
                fname = f"{fd}-icosari-sari-{model_name}-{m}.csv"
                save_csv(df, out_dir, fname)

        except Exception as e:
            failed.append((fd, f"{type(e).__name__}: {e}"))
            print(f"[{fd}] ABORTED — {type(e).__name__}: {e}")

    if failed:
        print("\nCompleted with errors — the following dates failed:")
        for d, reason in failed:
            print(f"  {d}: {reason}")
    else:
        print("\nAll dates completed successfully.")
