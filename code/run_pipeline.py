import argparse
import os
import subprocess

import papermill as pm

from config import ROOT
from src.r_utils import detect_rscript

# Headless plotting for matplotlib inside notebooks
os.environ["MPLBACKEND"] = "Agg"

# Locate Rscript matching the required version
RSCRIPT = detect_rscript()

# Ensure output dirs exist
for p in [
    ROOT / "figures",
    ROOT / "forecasts",
    ROOT / "nowcasts",
    ROOT / "results" / "tuning",
    ROOT / "results" / "scores",
]:
    p.mkdir(parents=True, exist_ok=True)

# Directories for code
CODE_PY = ROOT / "code"
CODE_R = ROOT / "r"

# Ordered pipeline
STAGES = ("exploration", "nowcasts", "tuning", "forecasts", "ensemble", "scores", "evaluation")

# Stage → tasks (relative to CODE_PY or CODE_R)
TASKS = {
    "exploration": [
        "plot_sari.ipynb",
        "plot_ari.ipynb",
        "plot_delays.ipynb",
        "autocorrelation.ipynb",
    ],
    "nowcasts": [
        "nowcasting/compute_nowcasts.R",
    ],
    "tuning": [
        "tuning_lightgbm.ipynb",
        "tuning_tsmixer.ipynb",
    ],
    "forecasts": [
        "compute_forecasts.ipynb",
        "baseline_historical.ipynb",
        "persistence/persistence.R",
        "hhh4/hhh4_default.R",
        "hhh4/hhh4_exclude_covid.R",
        "hhh4/hhh4_naive.R",
        "hhh4/hhh4_oracle.R",
        "hhh4/hhh4_shuffle.R",
        "hhh4/hhh4_skip.R",
        "hhh4/hhh4_vincentization.R",
        "tscount/tscount_extended.R",
        "tscount/tscount_simple.R",
    ],
    "ensemble": [
        "compute_ensemble.R",
    ],
    "scores": [
        "compute_scores.ipynb",
    ],
    "evaluation": [
        "plot_nowcasts.ipynb",
        "plot_forecasts.ipynb",
        "evaluation.ipynb",
        "evaluation_quantiles.ipynb",
        "diebold_mariano.ipynb",
    ],
}


def select_stages(start: str | None = None, end: str | None = None):
    i = STAGES.index(start or STAGES[0])  # default = first stage
    j = STAGES.index(end or STAGES[-1])  # default = last stage
    if j < i:
        raise SystemExit(
            f"Invalid stage range: {start!r} cannot precede {end!r}.\nPipeline order is: {', '.join(STAGES)}."
        )
    return STAGES[i : j + 1]


def run_rscript(script_name: str) -> None:
    print(f"- Executing {script_name} …")
    r_path = CODE_R / script_name

    try:
        subprocess.run(
            [RSCRIPT, r_path],
            cwd=ROOT,  # start R in the repo root so .Rprofile & renv auto-activate
        )
        print("  ✓ Done.")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def run_notebook(nb_name: str) -> None:
    print(f"- {nb_name}")
    nb_path = CODE_PY / nb_name

    try:
        pm.execute_notebook(
            input_path=nb_path,
            output_path=None,
            progress_bar=True,
            cwd=CODE_PY,
            kernel_name="replication-sari",
        )
        print("  ✓ Done.")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def run_task(task: str) -> None:
    if task.endswith(".ipynb"):
        run_notebook(task)
    elif task.endswith(".R"):
        run_rscript(task)
    else:
        raise SystemExit(f"Unsupported task type: {task}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run replication pipeline (Python + R).")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=STAGES, help="Run only this stage.")
    group.add_argument("--start", choices=STAGES, help="Stage to start from.")
    ap.add_argument("--end", choices=STAGES, help="Stop after this stage (inclusive).")
    ap.add_argument("--skip", choices=STAGES, nargs="+", help="Stages to skip (space separated).")
    args = ap.parse_args(argv)

    # If a single stage was specified, just run that one
    if args.stage:
        stages = [args.stage]
    else:
        # Otherwise select a contiguous range of stages (or all if no args given)
        stages = list(select_stages(args.start, args.end))

    # Drop any stages that were explicitly skipped
    if args.skip:
        stages = [s for s in stages if s not in args.skip]

    print("\n=== Selected stages ===")
    for s in stages:
        print(f"  - {s}")

    if "tuning" in stages:
        print("\n⚠️  Tuning may take a long time (days). Use '--skip tuning' to omit.\n")

    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        for task in TASKS[stage]:
            # print(task)
            run_task(task)

    print("\n✓ All selected stages completed.")


if __name__ == "__main__":
    main()
