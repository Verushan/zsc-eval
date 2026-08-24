"""Build the stage-2 training population yml for one MORL-benchmark arm.

`gen_S2_yml.py` builds the pipeline's own stage-2 population: 20 `sp` agents,
subsampled 16 at a time, five different draws. This does the same job for the
benchmark arms, whose whole point is that they differ only in the reward their
stage-1 agents optimised:

    bench_sp        population raised on sparse + hand-shaped
    bench_morl      population raised on w . r_vec, fixed w
    bench_morl_ad   population raised on w . r_vec, adaptive w
    mixed           bench_sp + bench_morl_ad, i.e. reward-diverse

Each arm has three stage-1 seeds and contributes `init`/`mid`/`final` per seed,
which is FCP's defining trick -- partners at three skill levels -- so an arm's
population is 9 policies and `mixed` is 18. The populations are the same size
for every single-reward arm, which is what makes the stage-2 comparison a
comparison of the *reward* rather than of population size.

Unlike `gen_S2_yml.py` this writes one yml per arm rather than one per stage-2
seed: with only three stage-1 agents there is no subsample to vary, so five
files would be five copies. The stage-2 seed still varies initialisation and
partner sampling.

Usage:
    python prep/gen_arm_S2_yml.py random0 --arm bench_sp --arm bench_morl_ad
"""

import argparse
import os
import os.path as osp

from loguru import logger

POLICY_POOL_DIR = os.getenv("POLICY_POOL")

# Stage-1 seeds trained by shell/train_morl_benchmark.sh.
ARM_SEEDS = [1, 2, 3]

# Checkpoints per stage-1 agent. FCP's population is the *training history* of
# each agent, not just its endpoint, so a partner set that only held `final`
# would be a different algorithm.
TAGS = ["init", "mid", "final"]

# arm -> the stage-1 experiment(s) its population is drawn from. Every arm but
# `mixed` is one experiment; `mixed` exists to ask whether reward diversity in
# the population is worth anything on top of checkpoint diversity.
ARM_POPULATIONS = {
    "bench_sp": ["bench_sp"],
    "bench_sparse": ["bench_sparse"],
    "bench_morl": ["bench_morl"],
    "bench_morl_ad": ["bench_morl_ad"],
    "mixed": ["bench_sp", "bench_morl_ad"],
}

HEADER = """\
{agent_name}:
    policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: True
"""

ENTRY = """\
{name}:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {actor}
"""


def population(arm, seeds):
    """(name, actor path relative to POLICY_POOL) for every partner in the arm."""
    out = []
    for exp in ARM_POPULATIONS[arm]:
        for seed in seeds:
            for tag in TAGS:
                out.append(
                    (
                        f"{exp}{seed}_{tag}",
                        osp.join("{layout}", "fcp", "s1", exp, f"sp{seed}_{tag}_actor.pt"),
                    )
                )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument(
        "-a",
        "--arm",
        action="append",
        required=True,
        choices=sorted(ARM_POPULATIONS),
        help="Arm whose stage-1 agents become the stage-2 population. Repeatable.",
    )
    parser.add_argument(
        "--arm_seeds",
        nargs="+",
        type=int,
        default=ARM_SEEDS,
        help=f"Stage-1 seeds per arm (default {ARM_SEEDS}).",
    )
    parser.add_argument(
        "--agent_name",
        default="fcp_adaptive",
        help="Name of the trainable policy. Must match --adaptive_agent_name.",
    )
    args = parser.parse_args()

    yml_dir = osp.join(POLICY_POOL_DIR, args.layout, "fcp", "s2")
    os.makedirs(yml_dir, exist_ok=True)

    for arm in args.arm:
        entries = population(arm, args.arm_seeds)
        yml_path = osp.join(yml_dir, f"train-{arm}.yml")
        with open(yml_path, "w", encoding="utf-8") as yml:
            yml.write(HEADER.format(agent_name=args.agent_name, layout=args.layout))
            for name, actor in entries:
                actor = actor.format(layout=args.layout)
                # An absent checkpoint means the stage-1 half did not finish or
                # was not extracted. Training against a silently smaller
                # population would break the size matching the comparison rests
                # on, so fail here rather than three hours into stage 2.
                full = osp.join(POLICY_POOL_DIR, actor)
                assert osp.exists(full), f"{name}: {full} does not exist"
                yml.write(ENTRY.format(name=name, layout=args.layout, actor=actor))
        logger.success(f"{arm}: {len(entries)} partners -> {yml_path}")
        # The runner asserts --population_size == len(population), so print it
        # for the shell script to read.
        print(f"{arm} {len(entries)} {yml_path}")


if __name__ == "__main__":
    main()
