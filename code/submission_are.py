import os
from pathlib import Path
import subprocess
import time

import pandas as pd
from git import Repo
from github import Auth, Github, GithubException

from config import QUANTILES, ROOT
from src.forecasting import generate_forecasts
from src.r_utils import detect_rscript
from src.realtime_utils import (
    download_latest_data,
    get_preceding_thursday,
    wait_for_data,
)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

RSCRIPT = detect_rscript()

forecast_date = str(get_preceding_thursday(pd.Timestamp.now().date()).date())

HUB_REPO = "KITmetricslab/RESPINOW-Hub"
HUB_FORK = "dwolffram/RESPINOW-Hub"

# local clone of RESPINOW-realtime
realtime_repo = Repo(ROOT)

NOWCAST_BRANCH = f"nowcast/are/{forecast_date}"
HHH4_BRANCH = f"hhh4/are/{forecast_date}"
ML_BRANCH = f"submission/are/{forecast_date}"

DATA_MSG = f"Update data for {forecast_date}"
NOWCAST_MSG = f"Add nowcasts for {forecast_date}"
HHH4_MSG = f"Add KIT-hhh4 forecasts for {forecast_date}"
ML_MSG = f"Add KIT-LightGBM and KIT-TSMixer forecasts for {forecast_date}"


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------


def connect_repos():
    token = os.environ["GITHUB_TOKEN"]
    gh = Github(auth=Auth.Token(token))
    return gh.get_repo(HUB_FORK), gh.get_repo(HUB_REPO)


def sync_fork(fork_repo):
    fork_repo.merge_upstream("main")
    print("🔃 Synced fork with upstream/main.")


def create_branch(fork_repo, branch):
    """Delete branch if it exists, then create a fresh one from main."""
    refname = f"refs/heads/{branch}"

    # 1. Delete old branch if it exists
    try:
        existing = fork_repo.get_git_ref(refname)
        existing.delete()
        print(f"🗑️ Deleted old branch {branch}.")
    except GithubException as e:
        if e.status != 404:
            raise  # something else is wrong (permissions, etc.)

    # 2. Create new branch from main
    base_ref = fork_repo.get_git_ref("heads/main")
    fork_repo.create_git_ref(ref=refname, sha=base_ref.object.sha)
    print(f"🌱 Created branch {branch}.")


def write_file_to_branch(fork_repo, branch, file_path, content, message):
    """Create or update file on the given branch."""
    try:
        existing = fork_repo.get_contents(file_path, ref=branch)
        fork_repo.update_file(
            path=existing.path,
            message=message,
            content=content,
            sha=existing.sha,
            branch=branch,
        )
        print(f"🔁 Updated {file_path}")
    except GithubException as e:
        if e.status == 404:
            fork_repo.create_file(
                path=file_path,
                message=message,
                content=content,
                branch=branch,
            )
            print(f"🆕 Created {file_path}")
        else:
            raise


def open_pr(upstream_repo, fork_repo, branch, title, body):
    pr = upstream_repo.create_pull(
        title=title,
        body=body,
        head=f"{fork_repo.owner.login}:{branch}",
        base="main",
    )
    print(f"✅ PR opened: {pr.html_url}")


def commit_and_push(repo, path, message):
    """Stage path, commit if repo has changes, and push to origin."""
    repo.git.add(path)
    if repo.is_dirty():
        repo.index.commit(message)
        repo.remote("origin").push()
        print(f"📤 Committed & pushed: {message}")
    else:
        print(f"⚠️ No changes in {path} — skip.")

        
def wait_for_files(paths, timeout=300, interval=1):
    """
    Wait until all files in `paths` exist.
    Raise TimeoutError after `timeout` seconds.
    """
    paths = [Path(p) for p in paths]
    start = time.time()

    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return  # all good

        if time.time() - start > timeout:
            missing_str = ", ".join(str(p) for p in missing)
            raise TimeoutError(f"Files did not appear in time: {missing_str}")

        time.sleep(interval)
        
# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------


# --- Wait for latest data ---
wait_for_data(interval_min=30, max_wait_hours=24)
download_latest_data()
commit_and_push(realtime_repo, "data", DATA_MSG)

# --- R-based nowcasts and hhh4 ---
subprocess.run(
    [
        RSCRIPT,
        "--vanilla",
        "-e",
        (
            f'setwd("{(ROOT / "r").as_posix()}"); '
            "renv::activate(); "
            "renv::restore(prompt = FALSE); "
            'renv::run("nowcasting/nowcasting.R", args = c("are")); '
            'renv::run("hhh4/hhh4_default.R", args = c("are"))'
        ),
    ],
    cwd=ROOT,
    check=True,
)

# Expected output files
nowcast_file = ROOT / "nowcasts"/ "simple_nowcast" / f"{forecast_date}-agi-are-simple_nowcast.csv"
hhh4_file = ROOT / "forecasts"/ "hhh4-coupling" / f"{forecast_date}-agi-are-hhh4-coupling.csv"

# --- Wait for both files (to avoid empty commits) ---
wait_for_files([nowcast_file, hhh4_file], timeout=300, interval=2)

# --- local commits (RESPINOW-realtime repo) ---
commit_and_push(realtime_repo, "nowcasts", NOWCAST_MSG)
commit_and_push(realtime_repo, "forecasts", HHH4_MSG)

# --- connect to GitHub ---
hub_fork, hub_repo = connect_repos()

# ==========================================================================
# 1) Submit NOWCASTS
# ==========================================================================
sync_fork(hub_fork)
create_branch(hub_fork, NOWCAST_BRANCH)

nowcast_in = ROOT / f"nowcasts/simple_nowcast/{forecast_date}-agi-are-simple_nowcast.csv"
nowcast_out = f"submissions/agi/are/KIT-simple_nowcast/{forecast_date}-agi-are-KIT-simple_nowcast.csv"

df = pd.read_csv(nowcast_in)
df = df.loc[(df["type"] != "quantile") | (df["quantile"].isin(QUANTILES))]
content = df.to_csv(index=False)

write_file_to_branch(hub_fork, NOWCAST_BRANCH, nowcast_out, content, NOWCAST_MSG)

pr_body = (
    f"Automated submission from RESPINOW-realtime.\n\nAdds the **KIT-simple_nowcast** nowcasts for {forecast_date}."
)
open_pr(hub_repo, hub_fork, NOWCAST_BRANCH, NOWCAST_MSG, pr_body)

# # ==========================================================================
# # 2) Submit HHH4 forecasts
# # ==========================================================================
sync_fork(hub_fork)
create_branch(hub_fork, HHH4_BRANCH)

hhh4_in = ROOT / f"forecasts/hhh4-coupling/{forecast_date}-agi-are-hhh4-coupling.csv"
hhh4_out = f"submissions/agi/are/KIT-hhh4/{forecast_date}-agi-are-KIT-hhh4.csv"

content = hhh4_in.read_text()

write_file_to_branch(hub_fork, HHH4_BRANCH, hhh4_out, content, HHH4_MSG)

pr_body = f"Automated submission from RESPINOW-realtime.\n\nAdds the **KIT-hhh4** forecasts for {forecast_date}."
open_pr(hub_repo, hub_fork, HHH4_BRANCH, HHH4_MSG, pr_body)

# # ==========================================================================
# # 3) ML forecasts
# # ==========================================================================
# generate_forecasts("lightgbm", forecast_date, data_mode="no_covariates", modes="coupling")
# generate_forecasts("tsmixer", forecast_date, data_mode="no_covariates", modes="coupling")

# commit_and_push(realtime_repo, "forecasts", ML_MSG)

# sync_fork(hub_fork)
# create_branch(hub_fork, ML_BRANCH)

# for model, name in [
#     ("lightgbm", "KIT-LightGBM"),
#     ("tsmixer", "KIT-TSMixer"),
# ]:
#     ml_in = (
#         ROOT
#         / f"forecasts/{model}-no_covariates-coupling/{forecast_date}-icosari-sari-{model}-no_covariates-coupling.csv"
#     )
#     ml_out = f"submissions/icosari/sari/{name}/{forecast_date}-icosari-sari-{name}.csv"

#     content = ml_in.read_text()

#     write_file_to_branch(hub_fork, ML_BRANCH, ml_out, content, ML_MSG)

# pr_body = (
#     "Automated submission from RESPINOW-realtime.\n\n"
#     f"Adds the **KIT-LightGBM** and **KIT-TSMixer** forecasts for {forecast_date}."
# )
# open_pr(hub_repo, hub_fork, ML_BRANCH, ML_MSG, pr_body)
