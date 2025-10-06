import os

import papermill as pm

from config import ROOT

os.environ["MPLBACKEND"] = "Agg"

PATH_FIGURES = ROOT / "figures"
PATH_FIGURES.mkdir(parents=True, exist_ok=True)

notebooks_to_run = [
    "plot_sari.ipynb",
    "plot_ari.ipynb",
    "plot_delays.ipynb",
    "autocorrelation.ipynb",
    "plot_nowcasts.ipynb",
    "plot_forecasts.ipynb",
    "evaluation.ipynb",
    "evaluation_quantiles.ipynb",
    "diebold_mariano.ipynb",
]

for notebook_path in notebooks_to_run:
    print(f"Executing {notebook_path}...")
    try:
        pm.execute_notebook(
            "code/" + notebook_path,
            None,
            report_mode=True,
            cwd="code/",
            kernel_name="replication-sari",
        )
        print(f"Successfully executed {notebook_path}")
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")

print("All notebooks processed.")
