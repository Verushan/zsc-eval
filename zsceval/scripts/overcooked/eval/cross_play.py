#!/usr/bin/env python
"""Cross-play evaluation: every ego agent against every evaluation partner.

`eval/eval.py` evaluates exactly one (agent0, agent1) pair per process, which
makes a cross-play matrix cost one interpreter start-up and one policy-pool load
per cell. This script loads the pool once and then walks the whole matrix,
assigning a *different* pair to each env thread, so a round of
`n_eval_rollout_threads` threads collects one episode for each of that many
distinct pairs simultaneously.

It writes a tidy record per episode rather than a pre-aggregated summary, so
mean / worst-case / variance-across-partners can all be computed afterwards
without re-running any rollouts.

Pass `--morl_objectives default` to additionally record the per-objective
breakdown (`eval_ep_obj_*`) for every cell; the objective vector is computed by
the env and costs nothing extra, and it is what makes "did the MORL agent
actually behave differently" answerable from cross-play data.

Usage:
    python eval/cross_play.py --layout_name random0 --num_agents 2 \
        --algorithm_name population --experiment_name xp \
        --population_yaml_path .../cross_play.yml \
        --ego_policy_names a b --partner_policy_names c d \
        --eval_episodes 8 --n_eval_rollout_threads 10 --dummy_batch_size 10 \
        --eval_result_path .../cross_play.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import setup_seed


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank, evaluation=True)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ShareSubprocDummyBatchVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
        all_args.dummy_batch_size,
    )


def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--store_traj", default=False, action="store_true")
    parser.add_argument(
        "--population_yaml_path",
        type=str,
        required=True,
        help="Population yml holding every policy named below.",
    )
    parser.add_argument(
        "--ego_policy_names",
        nargs="+",
        required=True,
        help="Policies evaluated as the ego agent.",
    )
    parser.add_argument(
        "--partner_policy_names",
        nargs="+",
        required=True,
        help="Policies played against. May overlap with --ego_policy_names.",
    )
    parser.add_argument(
        "--both_orders",
        default=False,
        action="store_true",
        help="Also play every pair with the ego agent in slot 1. random0 is not "
        "position-symmetric -- the two sides of the counter have different jobs -- "
        "so a single order measures only half the story.",
    )
    parser.add_argument(
        "--population_size", type=int, default=2, help="Logged for bookkeeping only."
    )
    # Overcooked_Env binds `script:`-prefixed policies to a player slot once, at
    # env construction, so they cannot take part in a matrix whose pairings change
    # per chunk. They are declared here only because the env reads the attributes.
    parser.add_argument("--agent0_policy_name", type=str, default="")
    parser.add_argument("--agent1_policy_name", type=str, default="")
    parser.add_argument("--eval_result_path", type=str, required=True)
    all_args = parser.parse_args(args)

    from zsceval.overcooked_config import OLD_LAYOUTS

    all_args.old_dynamics = all_args.layout_name in OLD_LAYOUTS
    return all_args


def build_pairs(all_args):
    """Ordered (agent0, agent1) pairs to evaluate, de-duplicated."""
    pairs = []
    seen = set()
    for ego in all_args.ego_policy_names:
        for partner in all_args.partner_policy_names:
            for pair in [(ego, partner)] + ([(partner, ego)] if all_args.both_orders else []):
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return pairs


def chunk_pairs(pairs, n_threads):
    """Split pairs into groups of at most `n_threads`, cycling to fill the last.

    A short final group is padded by repeating its own pairs rather than by
    borrowing from another group, so the extra episodes are still episodes of a
    pair we wanted -- no rollout is wasted and no cell is contaminated.
    """
    chunks = []
    for start in range(0, len(pairs), n_threads):
        group = pairs[start : start + n_threads]
        assignment = [group[i % len(group)] for i in range(n_threads)]
        chunks.append(assignment)
    return chunks


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.algorithm_name == "population"
    assert not all_args.random_index, (
        "--random_index shuffles which base-env player slot the ego agent occupies, "
        "but the per-agent episode counters are reported in base-env order, so the "
        "cross-play attribution would silently mix the two agents up."
    )

    if all_args.cuda and torch.cuda.is_available():
        logger.info("Using GPU")
        device = torch.device("cuda:0")
    else:
        logger.info("Using CPU")
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # Cross-play produces one JSON, not a training curve, so it never opens a
    # W&B run -- which also means base_runner must take its non-wandb branch and
    # be handed a real directory. Note `--use_wandb` is `action="store_false"`,
    # so this is set directly rather than left to the flag.
    all_args.use_wandb = False

    run_dir = (
        Path(os.getenv("PYTHONPATH"))
        / "results"
        / all_args.env_name
        / all_args.layout_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    all_args.run_dir = run_dir
    Path(os.path.dirname(all_args.eval_result_path)).mkdir(parents=True, exist_ok=True)

    setup_seed(all_args.seed)

    envs = make_eval_env(all_args, run_dir)
    eval_envs = envs

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from zsceval.runner.shared.overcooked_runner import OvercookedRunner as Runner

    runner = Runner(config)
    featurize_type = runner.policy.load_population(
        all_args.population_yaml_path, evaluation=True
    )
    runner.population_size = all_args.population_size

    named = all_args.ego_policy_names + all_args.partner_policy_names
    assert not any(n.startswith("script:") for n in named), (
        "script agents are bound to a player slot when the env is built, so they "
        "cannot be re-assigned per chunk; evaluate them with eval/eval.py instead"
    )

    known = set(runner.policy.policy_pool.keys())
    missing = [
        n
        for n in all_args.ego_policy_names + all_args.partner_policy_names
        if n not in known
    ]
    assert not missing, f"policies {missing} are not in {all_args.population_yaml_path}"

    pairs = build_pairs(all_args)
    chunks = chunk_pairs(pairs, all_args.n_eval_rollout_threads)
    rounds = max(1, all_args.eval_episodes)
    logger.info(
        f"{len(pairs)} pairs in {len(chunks)} chunks x {rounds} episodes "
        f"= {len(chunks) * rounds * all_args.n_eval_rollout_threads} episode rollouts"
    )

    records = []
    for c_i, assignment in enumerate(chunks):
        map_ea2p = {}
        featurize_types = []
        for e, (p0, p1) in enumerate(assignment):
            map_ea2p[(e, 0)] = p0
            map_ea2p[(e, 1)] = p1
            featurize_types.append(
                (featurize_type.get(p0, "ppo"), featurize_type.get(p1, "ppo"))
            )
        runner.policy.set_map_ea2p(map_ea2p)
        eval_envs.reset_featurize_type(featurize_types)

        for r_i in range(rounds):
            info = runner.evaluate_one_episode_with_multi_policy(
                runner.policy.policy_pool, map_ea2p
            )
            for e, (p0, p1) in enumerate(assignment):
                record = {
                    "agent0": p0,
                    "agent1": p1,
                    "chunk": c_i,
                    "round": r_i,
                    "thread": e,
                }
                for k, v in info.items():
                    record[k] = float(v[e])
                records.append(record)
            logger.info(
                f"chunk {c_i + 1}/{len(chunks)} episode {r_i + 1}/{rounds}: "
                f"mean sparse {np.mean([r['eval_ep_sparse_r'] for r in records[-len(assignment):]]):.1f}"
            )

    summary = defaultdict(list)
    for record in records:
        summary[(record["agent0"], record["agent1"])].append(record["eval_ep_sparse_r"])
    logger.success(
        "cross-play mean sparse return:\n"
        + "\n".join(
            f"  {a0} x {a1}: {np.mean(v):.1f} (n={len(v)})"
            for (a0, a1), v in sorted(summary.items())
        )
    )

    with open(all_args.eval_result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "layout": all_args.layout_name,
                "population_yaml_path": all_args.population_yaml_path,
                "episode_length": all_args.episode_length,
                "eval_stochastic": all_args.eval_stochastic,
                "seed": all_args.seed,
                "records": records,
            },
            f,
        )
    logger.success(f"wrote {len(records)} episode records to {all_args.eval_result_path}")

    envs.close()
    if getattr(runner, "writter", None) is not None:
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    main(sys.argv[1:])
