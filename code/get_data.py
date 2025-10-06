import pandas as pd
from config import ROOT

base = "https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/refs/heads/main/data"
sources = [("icosari", "sari"), ("agi", "are")]
files = [
    "latest_data-{}-{}.csv",
    "target-{}-{}.csv",
    "reporting_triangle-{}-{}.csv",
    "reporting_triangle-{}-{}-preprocessed.csv",
]

urls = [f"{base}/{src}/{disease}/{file.format(src, disease)}" for src, disease in sources for file in files]

for u in urls:
    pd.read_csv(u).to_csv(ROOT / f"data/{u.split('/')[-1]}", index=False)
