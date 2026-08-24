"""Build the population yml for the MORL cross-play benchmark.

Unlike `gen_S2_yml.py`, which lays out a *training* population, this writes an
evaluation-only pool: every entry is `train: False` and the file is consumed by
`eval/cross_play.py`, which decides pairings itself rather than reading them
from the yml.

Three kinds of policy end up in the same pool so that they can be played against
each other in one process:

* the benchmark arms  -- stage-1 self-play agents that differ only in the reward
  they optimised (`{layout}/fcp/s1/{arm}/sp{seed}_final_actor.pt`);
* the pipeline baseline -- the FCP stage-2 adaptive agent, i.e. what ZSC-Eval
  itself produces (`{layout}/fcp/s2/{exp}/{seed}.pt`);
* the held-out partners -- checkpoints from the pre-existing stage-1 pool that
  none of the arms trained against, which is what makes the evaluation
  zero-shot.

The stage-1 checkpoints carry no `rnn.*` tensors (see the `--use_recurrent_policy`
note in `shell/train_morl_benchmark.sh`) and so must be loaded through
`mlp_policy_config.pkl`; the stage-2 adaptive agent is recurrent and needs
`rnn_policy_config.pkl`. Getting this backwards raises on `load_state_dict`
rather than failing quietly, but it is easy to get backwards.
"""

import argparse
import os
import os.path as osp

from loguru import logger

POLICY_POOL_DIR = os.getenv("POLICY_POOL")

# Stage-1 arms trained by shell/train_morl_benchmark.sh.
ARMS = ["bench_sp", "bench_sparse", "bench_morl", "bench_morl_ad"]
ARM_SEEDS = [1, 2, 3]

# Arms for which the last checkpoint is not the best one the run passed through,
# so the benchmark also carries the peak-sparse checkpoint written by
# extract_models/extract_peak_models.py. A reward whose own optimum drifts away
# from the task's would otherwise be scored only on where it ended up, which
# measures the end of the run rather than what the reward could reach.
PEAK_ARMS = ["bench_morl", "bench_morl_ad"]

# The stage-2 FCP agent already in the pool.
S2_EXP = "fcp-S2-s16"
S2_SEEDS = [1, 2, 3, 4, 5]

# Stage-2 agents trained by shell/train_morl_stage_2.sh, one per arm: the FCP
# best-response agent for a population raised on that arm's reward. These are
# what answer "does the pipeline fix MORL's zero-shot deficit", so they have to
# be in the same matrix as the stage-1 arms they came from.
S2_ARM_SEEDS = [1, 2, 3]

# Held-out partners drawn from the pre-existing stage-1 pool. Seeds 1 and 6 come
# from 1e7-step runs (competent partners), 17 and 20 from 1e6-step runs (partners
# that play a plausible but much weaker game) -- the spread is the point, since a
# ZSC metric that only ever sees good partners cannot distinguish robustness from
# raw skill. `init` checkpoints are excluded: they are barely-trained policies,
# so every ego agent scores ~0 against them and the cell carries no signal.
HELDOUT_EXP = "sp"
HELDOUT_SEEDS = [1, 6, 17, 20]
HELDOUT_TAGS = ["mid", "final"]

ENTRY = """\
{name}:
    policy_config_path: {layout}/policy_config/{config}
    featurize_type: ppo
    train: False
    model_path:
        actor: {actor}
"""


def entries(
    layout, arm_seeds=None, peak_arms=None, s2_arms=None, s2_arm_seeds=None, s2_suffix=""
):
    """(name, policy_config, actor_path) for every policy in the pool."""
    out = []
    peak_arms = PEAK_ARMS if peak_arms is None else peak_arms
    for arm in ARMS:
        for seed in arm_seeds or ARM_SEEDS:
            tags = ["final"] + (["peak"] if arm in peak_arms else [])
            for tag in tags:
                # `_final` keeps the bare arm name so the common case reads
                # cleanly; only the extra checkpoint carries a suffix.
                suffix = "" if tag == "final" else f"_{tag}"
                out.append(
                    (
                        f"{arm}_s{seed}{suffix}",
                        "mlp_policy_config.pkl",
                        osp.join(layout, "fcp", "s1", arm, f"sp{seed}_{tag}_actor.pt"),
                    )
                )
    for seed in S2_SEEDS:
        out.append(
            (
                f"fcp_s2_s{seed}",
                "rnn_policy_config.pkl",
                osp.join(layout, "fcp", "s2", S2_EXP, f"{seed}.pt"),
            )
        )
    for arm in s2_arms or []:
        for seed in s2_arm_seeds or S2_ARM_SEEDS:
            out.append(
                (
                    f"s2_{arm}_s{seed}",
                    "rnn_policy_config.pkl",
                    osp.join(layout, "fcp", "s2", f"fcp-S2-{arm}{s2_suffix}", f"{seed}.pt"),
                )
            )
    for seed in HELDOUT_SEEDS:
        for tag in HELDOUT_TAGS:
            out.append(
                (
                    f"heldout_sp{seed}_{tag}",
                    "mlp_policy_config.pkl",
                    osp.join(
                        layout, "fcp", "s1", HELDOUT_EXP, f"sp{seed}_{tag}_actor.pt"
                    ),
                )
            )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("--out", type=str, default=None, help="Output yml path.")
    parser.add_argument(
        "--arm_seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Seeds trained for each arm (default {ARM_SEEDS}).",
    )
    parser.add_argument(
        "--peak_arms",
        nargs="*",
        default=None,
        help=f"Arms that also contribute their peak-sparse checkpoint (default {PEAK_ARMS}). "
        "Pass with no values to include none.",
    )
    parser.add_argument(
        "--s2_arms",
        nargs="*",
        default=[],
        help="Arms whose stage-2 agent (fcp-S2-{arm}) joins the pool, e.g. "
        "bench_sp bench_morl_ad. Empty by default so the stage-1-only benchmark "
        "reproduces unchanged.",
    )
    parser.add_argument(
        "--s2_arm_seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Stage-2 seeds per arm (default {S2_ARM_SEEDS}).",
    )
    parser.add_argument(
        "--s2_suffix",
        default="",
        help="Experiment-name suffix the stage-2 runs were trained with, e.g. "
        "'-pilot'. Must match train_morl_stage_2.sh's 6th argument.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Drop entries whose .pt is absent instead of failing. Useful while "
        "only some arms have finished training.",
    )
    args = parser.parse_args()

    yml_dir = osp.join(POLICY_POOL_DIR, args.layout, "morl_benchmark")
    os.makedirs(yml_dir, exist_ok=True)
    yml_path = args.out or osp.join(yml_dir, "cross_play.yml")

    written, skipped = [], []
    with open(yml_path, "w", encoding="utf-8") as yml:
        for name, config, actor in entries(
            args.layout,
            args.arm_seeds,
            args.peak_arms,
            args.s2_arms,
            args.s2_arm_seeds,
            args.s2_suffix,
        ):
            if not osp.exists(osp.join(POLICY_POOL_DIR, actor)):
                if args.skip_missing:
                    skipped.append(name)
                    continue
                raise FileNotFoundError(
                    f"{name}: {osp.join(POLICY_POOL_DIR, actor)} does not exist"
                )
            yml.write(ENTRY.format(name=name, layout=args.layout, config=config, actor=actor))
            written.append(name)

    if skipped:
        logger.warning(f"skipped {len(skipped)} missing policies: {skipped}")
    logger.success(f"wrote {len(written)} policies to {yml_path}")
    print(yml_path)


if __name__ == "__main__":
    main()
