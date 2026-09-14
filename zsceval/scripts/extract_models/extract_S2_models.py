import argparse
import os
import socket

import numpy as np
import wandb

wandb_name = os.getenv("WANDB_ENTITY")
POLICY_POOL_PATH = os.environ.get("POLICY_POOL")

from loguru import logger


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_target_index(array, percentile: float):
    """
    Find the index of the target value in the array based on the percentile.
    array: numpy array
    percentile: float, between 0 and 1
    return: index of the target value in the array
    """
    q3 = np.nanpercentile(array, int(percentile * 100))

    array_without_nan = array[~np.isnan(array)]

    index_of_max = np.nanargmax(array_without_nan)
    logger.debug(f"max index {index_of_max}/{len(array_without_nan)}")

    filtered_array = array_without_nan[index_of_max + 1 :]

    if filtered_array.size > 0:
        relative_index = (np.abs(filtered_array - q3)).argmin()
        original_index = np.where(array == filtered_array[relative_index])[0][0]

        return original_index, q3
    else:
        return len(array) - 1, np.nanmax(array)


def extract_pop_S2_models(
    layout, algo, exp, env, percentile=0.8, replicates=False, checkpoint="auto"
):
    logger.info(f"exp {exp}")
    api = wandb.Api(timeout=60)
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    drop_tags = ["hidden"] if replicates else ["hidden", "unused"]
    filters = {
        "$and": [
            {"config.experiment_name": exp},
            {layout_config: layout},
            {"state": "finished"},
            {"tags": {"$nin": drop_tags}},
        ]
    }
    logger.info(f"{wandb_name}/{env}")
    logger.info(f"filters {filters}")
    runs = api.runs(
        f"{wandb_name}/{env}",
        filters=filters,
        order="+config.seed",
    )
    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")

    # Checkpoints are written to `{seed}.pt`, so two finished runs sharing a seed
    # silently overwrite each other and which one survives depends only on the
    # order W&B returned them in. That happened once already: a stale copy of
    # train_morl_stage_2.sh left a second full run per seed, and the arm ended up
    # with checkpoints picked from a 5-point history while every other arm used a
    # 79-point one. Tag the runs you do not want with 'unused' -- the filter above
    # drops those unless --replicates is set, which keeps every run instead. Prefer
    # --replicates when the repeats are genuine reruns of the same config: tagging
    # picks a subsample by hand, and doing that per-arm biases an arm comparison.
    seeds = [r.config["seed"] for r in runs]
    dupes = sorted({s for s in seeds if seeds.count(s) > 1})
    if dupes and not replicates:
        detail = ", ".join(
            f"{r.id}(seed={r.config['seed']})" for r in runs if r.config["seed"] in dupes
        )
        raise RuntimeError(
            f"{exp}: multiple finished runs for seed(s) {dupes}: {detail}. "
            "Tag the unwanted ones 'unused' in W&B and re-run, or pass "
            "--replicates to keep them all as separate checkpoints."
        )

    # --replicates: keep every finished run instead of demanding one per seed.
    #
    # PPO on GPU is not bit-deterministic, so re-running a seed gives a genuinely
    # different agent rather than the same one twice. Those repeats are extra
    # samples, and collapsing them onto `{seed}.pt` throws that away -- worse,
    # resolving the collision by tagging some 'unused' is a *selection*, and on
    # unident_s the tagged-off half of bench_sp happened to be the stronger one
    # (dropped mean 190.0 against a kept 162.8) while bench_morl kept nearly all
    # of its runs. That is a biased subsample sitting under an arm comparison.
    #
    # The oldest run of a seed keeps `{seed}.pt` so existing pools and ymls do
    # not shift underneath anything; later ones become `{seed}r2.pt`, `{seed}r3.pt`.
    # `gen_crossplay_yml.py --s2_arm_seeds` takes these as labels.
    labels = {}
    if replicates:
        by_seed = {}
        for r in runs:
            by_seed.setdefault(r.config["seed"], []).append(r)
        for seed, rs in by_seed.items():
            for n, r in enumerate(sorted(rs, key=lambda x: x.created_at)):
                labels[r.id] = f"{seed}" if n == 0 else f"{seed}r{n + 1}"
        logger.info(
            f"{exp}: {len(runs)} finished runs over {len(by_seed)} seeds -> "
            f"labels {sorted(labels.values())}"
        )

    for i, run_id in enumerate(run_ids):
        run = runs[i]
        seed = labels.get(run.id, run.config["seed"])
        if run.state == "finished":
            logger.info(f"Run: {run_id} Seed: {seed}")
            files = run.files()
            policy_name = f"{algo}_adaptive"
            history = run.history()
            history = history[["_step", f"either-{algo}_adaptive-ep_sparse_r"]]
            steps = history["_step"].to_numpy().astype(int)
            ep_sparse_r = history[f"either-{algo}_adaptive-ep_sparse_r"].to_numpy()
            i_max_ep_sparse_r, max_ep_sparse_r = find_target_index(
                ep_sparse_r, percentile
            )
            max_ep_sparse_r_step = steps[i_max_ep_sparse_r]
            files = run.files()
            actor_pts = [
                f for f in files if f.name.startswith(f"{policy_name}/actor_periodic")
            ]
            actor_versions = [
                int(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts
            ]
            actor_versions.sort()
            auto_version = find_nearest(actor_versions, max_ep_sparse_r_step)
            # `auto` is the upstream rule: walk past the peak of the training
            # curve and take the checkpoint nearest the `percentile` value after
            # it, on the theory that the peak is noise and what follows is the
            # settled policy. In practice it is very nearly a no-op -- on the 16
            # unident_s and random0 stage-2 runs it chose the final checkpoint
            # for 15, and the one exception (s2_bench_sp_s1r2, chosen at 460800
            # of 1996800) scored 156.9 against an arm mean of 156.4.
            #
            # `final` removes the choice altogether. It is worth preferring for
            # a reported result not because `auto` is wrong but because the
            # metric it ranks on -- either-fcp_adaptive-ep_sparse_r, measured
            # against the *training* population -- does not predict zero-shot
            # return (Spearman +0.02 over the unident_s agents). A selection
            # rule keyed on a metric unrelated to the reported one is a degree
            # of freedom with nothing to justify it, and the honest alternative
            # is not a better metric: selecting on held-out ZSC would be
            # choosing the model on the evaluation set.
            version = actor_versions[-1] if checkpoint == "final" else auto_version
            note = "" if version == auto_version else f" (auto would pick {auto_version})"
            logger.info(
                f"actor version {version} / {actor_versions[-1]} [{checkpoint}]{note}, "
                f"sparse_r {max_ep_sparse_r:.3f}/{np.nanmax(ep_sparse_r):.3f}"
            )
            ckpt = run.file(f"{policy_name}/actor_periodic_{version}.pt")
            tmp_dir = f"tmp/{layout}/{exp}"
            logger.info(f"Fetch {tmp_dir}/{policy_name}/actor_periodic_{version}.pt")
            ckpt.download(f"{tmp_dir}", replace=True)
            algo_s2_dir = f"{POLICY_POOL_PATH}/{layout}/{algo}/s2"
            os.makedirs(f"{algo_s2_dir}/{exp}", exist_ok=True)
            os.system(
                f"mv {tmp_dir}/{policy_name}/actor_periodic_{version}.pt {algo_s2_dir}/{exp}/{seed}.pt"
            )
            logger.success(f"{layout} {algo} {exp} {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract S2 models")
    parser.add_argument("--layout", type=str, help="layout name")
    parser.add_argument("--env", type=str, help="env name")
    parser.add_argument(
        "-a", "--algo", "--algorithm", type=str, action="append", required=True
    )
    parser.add_argument("-p", type=float, help="percentile", default=0.8)
    parser.add_argument(
        "--exp",
        action="append",
        default=None,
        help="Experiment name to extract instead of the built-in ALG_EXPS list. "
        "Repeatable. Needed for stage-2 runs whose experiment name is not one of "
        "the pipeline's fixed `{algo}-S2-s{size}` names -- the MORL arms, for "
        "instance, are logged as `fcp-S2-bench_morl_ad`.",
    )
    parser.add_argument(
        "--replicates",
        action="store_true",
        help="Keep every finished run rather than requiring one per seed. Repeats "
        "of a seed are separate samples (PPO on GPU is not bit-deterministic), so "
        "the oldest keeps `{seed}.pt` and the rest become `{seed}r2.pt` etc. Also "
        "stops excluding runs tagged 'unused', since that tag is how this "
        "collision was resolved before -- and resolving it that way selects a "
        "subsample rather than a random one. Pass the labels straight to "
        "`gen_crossplay_yml.py --s2_arm_seeds`.",
    )

    parser.add_argument(
        "--checkpoint",
        choices=("auto", "final"),
        default="auto",
        help="Which checkpoint to take from each run. 'auto' keeps the "
        "upstream rule: the checkpoint nearest the -p percentile of the "
        "training curve after its peak. 'final' takes the last one and "
        "removes the choice, which is preferable for a reported result "
        "because the metric 'auto' ranks on is measured against the "
        "training population and does not predict zero-shot return. In "
        "practice the two agree: 'auto' picked the final checkpoint for 15 "
        "of 16 stage-2 runs.",
    )

    args = parser.parse_args()
    layout = args.layout
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
        "unident_s_m",
        "random0_mx",
        "random1_mx",
        "random3_mx",
        "unident_s_mx",
        "unident_s_m",
        "academy_3_vs_1_with_keeper",
        "all",
    ]
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
    algorithms = args.algo
    percentile = args.p

    assert all([algo in ["traj", "mep", "fcp", "cole", "hsp"] for algo in algorithms])
    ALG_EXPS = {
        "fcp": [
            # The parent repo pins POP_SIZE=16 in prep/gen_S2_yml.py, so the
            # stage-2 runs are logged as fcp-S2-s16.
            "fcp-S2-s16",
            "fcp-S2-s12",
            "fcp-S2-s24",
            "fcp-S2-s36",
        ],
        "mep": [
            "mep-S2-s24",
            "mep-S2-s36",
        ],
        "hsp": [
            "hsp-S2-s12",
            "hsp-S2-s24",
            "hsp-S2-s36",
        ],
        "traj": [
            "traj-S2-s24",
            "traj-S2-s36",
        ],
        "cole": ["cole-S2-s50", "cole-S2-s75"],
    }
    if args.exp:
        ALG_EXPS = {algo: list(args.exp) for algo in algorithms}

    MAX_ATTEMPTS = 5

    hostname = socket.gethostname()
    logger.info(f"hostname: {hostname}")
    for l in layout:
        for algo in algorithms:
            logger.info(f"for layout {l}")
            i = 0
            attempts = 0
            # for exp in ALG_EXPS[algo]:
            #     extract_pop_S2_models(l, algo, exp, args.env, percentile)
            while i < len(ALG_EXPS[algo]):
                exp = ALG_EXPS[algo][i]
                try:
                    extract_pop_S2_models(
                        l, algo, exp, args.env, percentile,
                        args.replicates, args.checkpoint,
                    )
                except Exception as e:
                    logger.error(e)
                    # The retry exists for flaky W&B calls, but without a bound a
                    # permanent error (wrong exp name, missing metric) spins here
                    # forever inside a batch job.
                    attempts += 1
                    if attempts >= MAX_ATTEMPTS:
                        logger.error(f"giving up on {exp} after {attempts} attempts")
                        attempts = 0
                        i += 1
                else:
                    attempts = 0
                    i += 1
