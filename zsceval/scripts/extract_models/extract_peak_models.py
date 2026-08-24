"""Extract the checkpoint at which a stage-1 run was *best* on a chosen metric.

`extract_sp_models.py` takes init / mid / final, where final is simply the last
checkpoint. That is the right thing for building an FCP population, but it is
the wrong thing for benchmarking a reward function whose optimum and the task's
optimum can come apart: a run whose scalarized reward keeps climbing while its
sparse return collapses is scored on the collapsed policy, which measures the
end of the run rather than what the reward was capable of.

This pulls the checkpoint nearest the argmax of `--metric` and stores it beside
the others as `sp{seed}_{tag}_actor.pt`, so a benchmark can report both "where
this reward ended up" and "the best policy it passed through".

Usage:
    python extract_models/extract_peak_models.py --layout random0 --env Overcooked \
        --exp bench_morl --metric ep_sparse_r --tag peak
"""

import argparse
import os

import numpy as np
import wandb
from loguru import logger

WANDB_NAME = os.getenv("WANDB_ENTITY")
POLICY_POOL_PATH = os.getenv("POLICY_POOL")


def extract_peak(layout, exp, env, metric, tag, smooth):
    api = wandb.Api(timeout=60)
    layout_config = "config.layout_name" if "overcooked" in env.lower() else "config.scenario_name"
    runs = list(
        api.runs(
            f"{WANDB_NAME}/{env}",
            filters={
                "$and": [
                    {"config.experiment_name": exp},
                    {layout_config: layout},
                    {"state": "finished"},
                    {"tags": {"$nin": ["hidden", "unused"]}},
                ]
            },
            order="+config.seed",
        )
    )
    logger.info(f"{exp}: {len(runs)} finished runs")

    seeds = set()
    for run in runs:
        seed = run.config["seed"]
        if seed in seeds:
            continue
        history = run.history(samples=2000)
        if history.empty or metric not in history:
            logger.warning(f"run {run.id} has no {metric!r} history, skipping")
            continue
        sub = history[["_step", metric]].dropna()
        if sub.empty:
            continue
        seeds.add(seed)

        steps = sub["_step"].to_numpy().astype(int)
        values = sub[metric].to_numpy(dtype=float)
        # A single lucky logging interval is noise, not a peak. Smooth over a
        # short window so the checkpoint picked is one the run actually held.
        if smooth > 1 and len(values) >= smooth:
            kernel = np.ones(smooth) / smooth
            smoothed = np.convolve(values, kernel, mode="valid")
            offset = smooth // 2
            peak_index = int(np.argmax(smoothed)) + offset
        else:
            peak_index = int(np.argmax(values))
        peak_step, peak_value = int(steps[peak_index]), float(values[peak_index])

        actor_pts = [f for f in run.files() if f.name.startswith("actor_periodic")]
        if not actor_pts:
            logger.warning(f"run {run.id} has no actor_periodic files, skipping")
            continue
        versions = sorted(int(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts)
        version = min(versions, key=lambda v: abs(v - peak_step))

        logger.info(
            f"sp{seed}: {metric} peaks at {peak_value:.1f} near step {peak_step}; "
            f"taking checkpoint {version} (final was {values[-1]:.1f})"
        )

        tmp_dir = f"tmp/{layout}/{exp}"
        run.file(f"actor_periodic_{version}.pt").download(tmp_dir, replace=True)
        out_dir = f"{POLICY_POOL_PATH}/{layout}/fcp/s1/{exp}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/sp{seed}_{tag}_actor.pt"
        os.system(f"mv {tmp_dir}/actor_periodic_{version}.pt {out_path}")
        logger.success(f"pt store in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--env", default="Overcooked")
    parser.add_argument("--exp", required=True, action="append")
    parser.add_argument("--metric", default="ep_sparse_r")
    parser.add_argument("--tag", default="peak")
    parser.add_argument(
        "--smooth", type=int, default=3, help="Window used to smooth the metric before argmax."
    )
    args = parser.parse_args()
    for exp in args.exp:
        extract_peak(args.layout, exp, args.env, args.metric, args.tag, args.smooth)
