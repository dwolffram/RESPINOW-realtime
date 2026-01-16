import os
import subprocess
import time
from pathlib import Path

import pandas as pd
from git import Repo
from github import Auth, Github, GithubException

from config import QUANTILES, ROOT
from src.forecasting import generate_forecasts
from src.r_utils import detect_rscript
from src.realtime_utils import download_latest_data, get_preceding_thursday


TARGETS = [
    {
        "target": "sari",
        "source": "icosari",
        "hub_path": "submissions/icosari/sari",
        "branch_prefix": "sari/",
    },
    {
        "target": "are",
        "source": "agi",
        "hub_path": "submissions/agi/are",
        "branch_prefix": "are/",
    },
]

HUB_REPO = "KITmetricslab/RESPINOW-Hub"
HUB_FORK = "dwolffram/RESPINOW-Hub"
ML_MODELS = [("lightgbm", "KIT-LightGBM"), ("tsmixer", "KIT-TSMixer")]


def connect_repos():
    token = os.environ["GITHUB_TOKEN"]
    gh = Github(auth=Auth.Token(token))
    return gh.get_repo(HUB_FORK), gh.get_repo(HUB_REPO)


def sync_fork(fork_repo):
    fork_repo.merge_upstream("main")
    print("Synced fork with upstream/main.")


def prepare_branch(fork_repo, branch):
    sync_fork(fork_repo)
    create_branch(fork_repo, branch)


def create_branch(fork_repo, branch):
    refname = f"refs/heads/{branch}"

    try:
        fork_repo.get_git_ref(refname).delete()
        print(f"Deleted old branch {branch}.")
    except GithubException as e:
        if e.status != 404:
            raise

    base_ref = fork_repo.get_git_ref("heads/main")
    fork_repo.create_git_ref(ref=refname, sha=base_ref.object.sha)
    print(f"Created branch {branch}.")


def write_file_to_branch(fork_repo, branch, file_path, content, message):
    try:
        existing = fork_repo.get_contents(file_path, ref=branch)
        fork_repo.update_file(
            path=existing.path,
            message=message,
            content=content,
            sha=existing.sha,
            branch=branch,
        )
        print(f"Updated {file_path}")
    except GithubException as e:
        if e.status == 404:
            fork_repo.create_file(
                path=file_path,
                message=message,
                content=content,
                branch=branch,
            )
            print(f"Created {file_path}")
        else:
            raise


def open_pr(upstream_repo, fork_repo, branch, title, body):
    pr = upstream_repo.create_pull(
        title=title,
        body=body,
        head=f"{fork_repo.owner.login}:{branch}",
        base="main",
    )
    print(f"PR opened: {pr.html_url}")


def commit_and_push(repo: Repo, path: str, message: str):
    repo.git.add(path)
    if repo.is_dirty():
        repo.index.commit(message)
        repo.remote("origin").push()
        print(f"Committed & pushed: {message}")
    else:
        print(f"No changes in {path}; skipping commit.")


def wait_for_files(paths, timeout=300, interval=2):
    paths = [Path(p) for p in paths]
    start = time.time()

    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return

        if time.time() - start > timeout:
            missing_str = ", ".join(str(p) for p in missing)
            raise TimeoutError(f"Files did not appear in time: {missing_str}")

        time.sleep(interval)


def hub_data_is_ready(source: str, target: str) -> bool:
    path = f"https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/refs/heads/main/data/{source}/{target}/reporting_triangle-{source}-{target}.csv"
    current_date = pd.Timestamp.now().date()
    expected_date = str((get_preceding_thursday(current_date) - pd.Timedelta(days=4)).date())
    df = pd.read_csv(path)
    last_date = df["date"].iloc[-1]
    print(f"{source}/{target}: last date {last_date} | expected {expected_date}")
    return last_date == expected_date


def expected_outputs(cfg, forecast_date: str):
    nowcast_file = (
        ROOT / "nowcasts" / "simple_nowcast" / f"{forecast_date}-{cfg['source']}-{cfg['target']}-simple_nowcast.csv"
    )
    hhh4_file = (
        ROOT / "forecasts" / "hhh4-coupling" / f"{forecast_date}-{cfg['source']}-{cfg['target']}-hhh4-coupling.csv"
    )
    return nowcast_file, hhh4_file


def run_r_nowcast_and_hhh4(rscript: str, forecast_date: str, targets):
    calls: list[str] = []
    for cfg in targets:
        calls.append(
            'renv::run("nowcasting/nowcasting.R", '
            f'args = c("--disease={cfg["target"]}", "--forecast_date={forecast_date}"))'
        )
        calls.append(
            'renv::run("hhh4/hhh4_default.R", '
            f'args = c("--disease={cfg["target"]}", "--forecast_date={forecast_date}"))'
        )

    expr = f'setwd("{(ROOT / "r").as_posix()}"); renv::activate(); renv::restore(prompt = FALSE); ' + "; ".join(calls)

    subprocess.run(
        [rscript, "--vanilla", "-e", expr],
        cwd=ROOT,
        check=True,
    )


def submit_target(cfg, forecast_date: str, hub_fork, hub_repo):
    nowcast_branch = f"nowcast/{cfg['branch_prefix']}{forecast_date}"
    hhh4_branch = f"hhh4/{cfg['branch_prefix']}{forecast_date}"
    ml_branch = f"submission/{cfg['branch_prefix']}{forecast_date}"

    nowcast_msg = f"Add nowcasts for {cfg['target'].upper()} ({forecast_date})"
    hhh4_msg = f"Add KIT-hhh4 forecasts for {cfg['target'].upper()} ({forecast_date})"
    ml_msg = f"Add KIT-LightGBM and KIT-TSMixer forecasts for {cfg['target'].upper()} ({forecast_date})"

    # 1) NOWCAST
    prepare_branch(hub_fork, nowcast_branch)

    nowcast_in, hhh4_in = expected_outputs(cfg, forecast_date)
    nowcast_out = (
        f"{cfg['hub_path']}/KIT-simple_nowcast/{forecast_date}-{cfg['source']}-{cfg['target']}-KIT-simple_nowcast.csv"
    )
    df = pd.read_csv(nowcast_in)
    df = df.loc[(df["type"] != "quantile") | (df["quantile"].isin(QUANTILES))]
    write_file_to_branch(
        hub_fork,
        nowcast_branch,
        nowcast_out,
        df.to_csv(index=False),
        nowcast_msg,
    )
    open_pr(
        hub_repo,
        hub_fork,
        nowcast_branch,
        nowcast_msg,
        f"Automated submission from RESPINOW-realtime.\n\nAdds the **KIT-simple_nowcast** nowcasts for {forecast_date}.",
    )

    # 2) HHH4
    prepare_branch(hub_fork, hhh4_branch)

    hhh4_out = f"{cfg['hub_path']}/KIT-hhh4/{forecast_date}-{cfg['source']}-{cfg['target']}-KIT-hhh4.csv"
    write_file_to_branch(
        hub_fork,
        hhh4_branch,
        hhh4_out,
        hhh4_in.read_text(),
        hhh4_msg,
    )
    open_pr(
        hub_repo,
        hub_fork,
        hhh4_branch,
        hhh4_msg,
        f"Automated submission from RESPINOW-realtime.\n\nAdds the **KIT-hhh4** forecasts for {forecast_date}.",
    )

    # 3) ML
    generate_forecasts(
        "lightgbm",
        forecast_date,
        target=cfg["target"],
        data_mode="no_covariates",
        modes="coupling",
    )
    generate_forecasts(
        "tsmixer",
        forecast_date,
        target=cfg["target"],
        data_mode="no_covariates",
        modes="coupling",
    )

    prepare_branch(hub_fork, ml_branch)

    for model, name in ML_MODELS:
        ml_in = (
            ROOT
            / f"forecasts/{model}-no_covariates-coupling/{forecast_date}-{cfg['source']}-{cfg['target']}-{model}-no_covariates-coupling.csv"
        )
        ml_out = f"{cfg['hub_path']}/{name}/{forecast_date}-{cfg['source']}-{cfg['target']}-{name}.csv"
        write_file_to_branch(hub_fork, ml_branch, ml_out, ml_in.read_text(), ml_msg)

    open_pr(
        hub_repo,
        hub_fork,
        ml_branch,
        ml_msg,
        "Automated submission from RESPINOW-realtime.\n\n"
        f"Adds the **KIT-LightGBM** and **KIT-TSMixer** forecasts for {forecast_date}.",
    )


def main():
    rscript = detect_rscript()
    forecast_date = str(get_preceding_thursday(pd.Timestamp.now().date()).date())

    realtime_repo = Repo(ROOT)

    # Pick whichever target becomes available first, run it, then re-check the other.
    max_wait_hours = 24
    interval_min = 30
    deadline = time.time() + max_wait_hours * 3600

    remaining = TARGETS.copy()
    hub_fork = hub_repo = None
    while remaining and time.time() < deadline:
        ready_cfg = None
        for cfg in remaining:
            if hub_data_is_ready(cfg["source"], cfg["target"]):
                ready_cfg = cfg
                break

        if ready_cfg is None:
            time.sleep(interval_min * 60)
            continue

        remaining.remove(ready_cfg)

        # Refresh local data once something is ready (downloads both SARI+ARE inputs).
        download_latest_data()
        commit_and_push(realtime_repo, "data", f"Update data for {forecast_date}")

        # If the other target isn't ready yet, we'll still run the available one now,
        # then loop back and keep checking until deadline.

        run_r_nowcast_and_hhh4(rscript, forecast_date, [ready_cfg])

        nowcast_file, hhh4_file = expected_outputs(ready_cfg, forecast_date)
        wait_for_files([nowcast_file, hhh4_file], timeout=300, interval=2)

        commit_and_push(
            realtime_repo,
            "nowcasts",
            f"Add nowcasts for {ready_cfg['target'].upper()} ({forecast_date})",
        )
        commit_and_push(
            realtime_repo,
            "forecasts",
            f"Add KIT-hhh4 forecasts for {ready_cfg['target'].upper()} ({forecast_date})",
        )

        if hub_fork is None or hub_repo is None:
            hub_fork, hub_repo = connect_repos()
        submit_target(ready_cfg, forecast_date, hub_fork, hub_repo)
        commit_and_push(
            realtime_repo,
            "forecasts",
            f"Add ML forecasts for {ready_cfg['target'].upper()} ({forecast_date})",
        )

    if remaining:
        skipped = ", ".join(f"{c['source']}/{c['target']}" for c in remaining)
        print(f"Timed out waiting for: {skipped}")


if __name__ == "__main__":
    main()
