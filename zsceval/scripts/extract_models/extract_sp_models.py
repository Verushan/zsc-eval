import os
import socket
import sys
import time

import numpy as np
import wandb
from loguru import logger

wandb_name = os.getenv("WANDB_ENTITY")
POLICY_POOL_PATH = os.getenv("POLICY_POOL")


def extract_sp_S1_models(layout, exp, env, metric: str = "ep_sparse_r"):
    """Pull init/mid/final actor checkpoints for every finished stage-1 run.

    Args:
        layout: Overcooked layout / GRF scenario name.
        exp: `experiment_name` the runs were logged under, e.g. "sp" or "morl".
        env: W&B project, e.g. "overcooked".
        metric: History key the checkpoints are ranked by. Defaults to the
            sparse return; MORL agents should use "ep_morl_r", the reward
            they were actually trained on.
    """
    api = wandb.Api()

    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"

    runs = api.runs(
        f"{wandb_name}/{env}",
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

    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"Processing {len(runs)} runs")
    seeds = set()

    for r_i, run_id in enumerate(run_ids):
        run = runs[r_i]
        history = run.history()

        if history.empty:
            continue

        if metric not in history:
            logger.warning(f"Run {run_id} has no {metric!r} history, skipping")
            continue

        # Eval-only rows leave the training metrics NaN, which would poison the
        # interpolation below.
        history = history[["_step", metric]].dropna()

        if history.empty:
            continue

        steps = history["_step"].to_numpy().astype(int)
        scores = history[metric].to_numpy()
        final_score = np.mean(scores[-5:])

        if run.config["seed"] in seeds:
            continue

        i = run.config["seed"]

        logger.info(
            f"sp{i} Run: {run_id} Seed: {run.config['seed']} {metric} {final_score}"
        )

        seeds.add(run.config["seed"])

        # The W&B file paginator occasionally returns a null page under load --
        # four of these jobs finishing at once was enough -- and the list
        # comprehension then dies on `last_response["project"]` being None,
        # after the whole training run has already been paid for. Bounded
        # retry, because a permanent error should still surface.
        actor_pts = None
        for attempt in range(5):
            try:
                actor_pts = [
                    f for f in run.files() if f.name.startswith("actor_periodic")
                ]
                break
            except TypeError as err:
                logger.warning(
                    f"W&B file listing for {run_id} failed "
                    f"(attempt {attempt + 1}/5): {err}"
                )
                time.sleep(5 * (attempt + 1))
        if actor_pts is None:
            logger.error(f"Could not list files for {run_id}; skipping")
            continue

        if not actor_pts:
            continue

        actor_versions = [
            eval(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts
        ]

        actor_pts = {v: p for v, p in zip(actor_versions, actor_pts)}
        actor_versions = sorted(actor_versions)

        max_actor_versions = max(actor_versions) + 1
        max_steps = max(steps)

        new_steps = [steps[0]]
        new_scores = [scores[0]]

        for s, er in zip(steps[1:], scores[1:]):
            l_s = new_steps[0]
            l_er = new_scores[-1]
            for w in range(l_s + 1, s, 100):
                new_steps.append(w)
                new_scores.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))

        steps = new_steps
        scores = new_scores

        # select checkpoints
        selected_pts = dict(init=0, mid=-1, final=max_steps)
        mid_score = final_score / 2
        min_delta = 1e9

        for s, score in zip(steps, scores):
            if min_delta > abs(mid_score - score):
                min_delta = abs(mid_score - score)
                selected_pts["mid"] = s

        selected_pts = {
            k: int(v / max_steps * max_actor_versions) for k, v in selected_pts.items()
        }
        score_dict = dict(init=0, mid=mid_score, final=final_score)

        for tag, exp_version in selected_pts.items():
            version = actor_versions[0]
            for actor_version in actor_versions:
                if abs(exp_version - version) > abs(exp_version - actor_version):
                    version = actor_version
            logger.info(
                f"sp{i}: {tag} Expected: {exp_version} {score_dict[tag]} Found: {version}"
            )
            ckpt = actor_pts[version]
            tmp_dir = f"tmp/{layout}/{exp}"
            ckpt.download(tmp_dir, replace=True)
            fcp_s1_dir = f"{POLICY_POOL_PATH}/{layout}/fcp/s1"
            os.makedirs(f"{fcp_s1_dir}/{exp}", exist_ok=True)
            sp_s1_path = f"{fcp_s1_dir}/{exp}/sp{i}_{tag}_actor.pt"
            logger.info(f"pt store in {sp_s1_path}")
            os.system(f"mv {tmp_dir}/actor_periodic_{version}.pt {sp_s1_path}")


if __name__ == "__main__":
    layout = sys.argv[1]
    env = sys.argv[2]
    assert layout in [
        "random0",
        "random0_medium",
        "random1",
        "random3",
        "small_corridor",
        "unident_s",
        "random0_m",
        "random1_m",
        "random3_m",
        "academy_3_vs_1_with_keeper",
        "all",
    ], layout
    if layout == "all":
        layout = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layout = [layout]
    hostname = socket.gethostname()
    exp_names = {"random0": "sp", "random3_m": "sp"}

    # Optional overrides so non-`sp` stage-1 agents can be extracted:
    #   extract_sp_models.py random0 overcooked morl ep_morl_r
    exp_override = sys.argv[3] if len(sys.argv) > 3 else None
    metric = sys.argv[4] if len(sys.argv) > 4 else "ep_sparse_r"

    logger.info(f"hostname: {hostname}")
    for l in layout:
        exp = exp_override or exp_names[l]
        logger.info(f"Extracting {exp} for {l} ranked by {metric}")
        extract_sp_S1_models(l, exp, env, metric=metric)
