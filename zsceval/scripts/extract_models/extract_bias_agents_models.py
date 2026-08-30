import glob
import os
import os.path as osp
import shutil
import socket
import sys

import numpy as np
import wandb
from loguru import logger

from zsceval.utils.train_util import get_base_run_dir

wandb_name = os.getenv("WANDB_ENTITY")
# Every other extractor resolves the pool from $POLICY_POOL. The upstream
# relative path only lands in the right place when the cwd happens to be a
# subdirectory of scripts/overcooked, so extracting from scripts/ (as the
# pipelines do) silently filed the agents where gen_crossplay_yml.py cannot
# see them.
POLICY_POOL_PATH = os.getenv("POLICY_POOL", "../policy_pool")

def local_run_files(layout, exp, run_id, env="Overcooked", algorithm="mappo"):
    """The run's own files/ directory on this machine, or None.

    Bias agents train through the *separated* runner, whose save() wrote
    checkpoints into wandb.run.dir without registering them via wandb.save().
    wandb >= 0.13 only uploads registered files, so for every run trained before
    that was fixed the W&B file list is empty and disk holds the only copy.

    train_bias_agent.py passes dir=run_dir to wandb.init(), so the layout is
    {results}/{env}/{layout}/{algorithm}/{exp}/wandb/run-{timestamp}-{run_id}/files.
    """
    pattern = osp.join(
        get_base_run_dir(), env, layout, algorithm, exp, "wandb", f"run-*-{run_id}", "files"
    )
    matches = sorted(d for d in glob.glob(pattern) if osp.isdir(d))
    return matches[-1] if matches else None


def local_actor_versions(files_dir):
    """Periodic checkpoint steps present on disk, from agent 0's files."""
    versions = []
    for path in glob.glob(osp.join(files_dir, "actor_agent0_periodic_*.pt")):
        try:
            versions.append(int(osp.basename(path).split("_")[-1][: -len(".pt")]))
        except ValueError:
            continue
    return sorted(versions)


def extract_sp_S1_models(layout, exp, env="Overcooked"):
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
    logger.info(f"num of runs: {len(runs)}")
    seeds = set()
    missing = []
    num_agents = None
    for r_i, run_id in enumerate(run_ids):
        run = runs[r_i]
        if run.state == "finished":
            if num_agents is None:
                num_agents = run.config["num_agents"]
            if run.config["seed"] in seeds:
                continue
            i = run.config["seed"]
            history = run.history()
            history = history[["_step", "ep_sparse_r"]]
            steps = history["_step"].to_numpy().astype(int)
            ep_sparse_r = history["ep_sparse_r"].to_numpy()
            final_ep_sparse_r = np.mean(ep_sparse_r[-5:])
            logger.info(
                f"hsp{i} Run: {run_id} Seed: {run.config['seed']} Return {final_ep_sparse_r}"
            )
            seeds.add(run.config["seed"])
            actor_pts = [f for f in run.files() if f.name.startswith("actor")]
            actor_versions = sorted(
                {int(f.name.split("_")[-1][: -len(".pt")]) for f in actor_pts}
            )
            files_dir = None
            if not actor_versions:
                # env_name/algorithm_name as the run itself recorded them: the
                # results tree is keyed on those, and __main__ passes a
                # lowercased env that only works as a W&B project name.
                files_dir = local_run_files(
                    layout,
                    exp,
                    run_id,
                    env=run.config.get("env_name", "Overcooked"),
                    algorithm=run.config.get("algorithm_name", "mappo"),
                )
                if files_dir is not None:
                    actor_versions = local_actor_versions(files_dir)
                if actor_versions:
                    logger.info(
                        f"hsp{i}: W&B holds no checkpoints, reading {len(actor_versions)} from disk"
                    )
            if not actor_versions:
                logger.error(
                    f"hsp{i} (run {run_id}, seed {i}): no actor checkpoints on W&B and "
                    f"none on disk for this host; skipping"
                )
                missing.append(i)
                continue
            max_actor_versions = max(actor_versions) + 1
            max_steps = max(steps)

            new_steps = [steps[0]]
            new_ep_sparse_r = [ep_sparse_r[0]]
            for s, er in zip(steps[1:], ep_sparse_r[1:]):
                l_s = new_steps[-1]
                l_er = new_ep_sparse_r[-1]
                for w in range(l_s + 1, s, 100):
                    new_steps.append(w)
                    new_ep_sparse_r.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))
            steps = new_steps
            ep_sparse_r = new_ep_sparse_r

            # select checkpoints
            selected_pts = dict(mid=-1, final=max_steps)
            mid_ep_sparse_r = final_ep_sparse_r / 2
            min_delta = 1e9
            for s, score in zip(steps, ep_sparse_r):
                if min_delta > abs(mid_ep_sparse_r - score):
                    min_delta = abs(mid_ep_sparse_r - score)
                    selected_pts["mid"] = s

            selected_pts = {
                k: int(v / max_steps * max_actor_versions)
                for k, v in selected_pts.items()
            }
            sparse_r_dict = dict(init=0, mid=mid_ep_sparse_r, final=final_ep_sparse_r)
            for tag, exp_version in selected_pts.items():
                version = actor_versions[0]
                for actor_version in actor_versions:
                    if abs(exp_version - version) > abs(exp_version - actor_version):
                        version = actor_version
                logger.info(
                    f"hsp{i}: {tag} Expected: {exp_version} {sparse_r_dict[tag]} Found: {version}"
                )
                if files_dir is None:
                    src_dir = f"tmp/{layout}/{exp}"
                    for a_i in range(run.config["num_agents"]):
                        run.file(f"actor_agent{a_i}_periodic_{version}.pt").download(
                            src_dir, replace=True
                        )
                else:
                    src_dir = files_dir

                hsp_s1_dir = (
                    f"{POLICY_POOL_PATH}/{layout}/hsp/s1/{exp.replace('-S1', '')}"
                )
                os.makedirs(hsp_s1_dir, exist_ok=True)
                for a_i in range(run.config["num_agents"]):
                    src = osp.join(src_dir, f"actor_agent{a_i}_periodic_{version}.pt")
                    pt_path = f"{hsp_s1_dir}/hsp{i}_{tag}_w{a_i}_actor.pt"
                    logger.info(f"pt {a_i} store in {pt_path}")
                    # copy, never move: src may be the run's own wandb directory,
                    # which has to stay intact for the other tag and for reruns.
                    shutil.copyfile(src, pt_path)

    extracted = sorted(seeds - set(missing))
    if missing:
        logger.warning(
            f"{layout}: no checkpoints found for seeds {sorted(missing)} -- they are "
            "absent from W&B and from this host's results tree"
        )
    logger.success(f"Extracted {len(extracted)} models for {layout}: seeds {extracted}")


if __name__ == "__main__":
    layout = sys.argv[1]

    env = "overcooked"

    if len(sys.argv) >= 3:
        env = sys.argv[2]

    # Optional third positional: the W&B experiment_name to extract. The map
    # below says "hsp-s1" while shell/train_bias_agents.sh writes "hsp-S1", and
    # the W&B filter is case-sensitive, so the default pair does not actually
    # match. That script now defaults to "hsp-s1"; this override is for runs
    # under any other name (a reduced pilot, say).
    exp_override = sys.argv[3] if len(sys.argv) >= 4 else None

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
    exp_names = {
        "random3_m": "hsp-S1",
        "small_corridor": "hsp-S1",
        "random0": "hsp-s1",
        "random0_medium": "hsp-s1",
        "random1": "hsp-s1",
        "random3": "hsp-s1",
        "small_corridor": "hsp-s1",
        "unident_s": "hsp-s1",
        "random0_m": "hsp-s1",
        "random1_m": "hsp-s1",
        "random3_m": "hsp-s1",
        "academy_3_vs_1_with_keeper": "hsp-s1",
    }

    # logger.add(f"./extract_log/extract_{layout}_hsp_S1_models.log")
    # logger.info(f"hostname: {hostname}")
    for l in layout:
        exp = exp_override or exp_names[l]
        logger.info(f"Extracting {exp} for {l}")
        extract_sp_S1_models(l, exp, env)
