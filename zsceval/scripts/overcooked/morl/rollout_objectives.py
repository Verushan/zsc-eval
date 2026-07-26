"""Standalone smoke test for the Overcooked multi-objective (MORL) reward vector.

Builds an `OvercookedEnv` with a vector-valued reward attached, rolls out short
episodes with random or scripted agents, prints the cumulative per-objective
scores, and cross-checks them against the environment's own (untouched) reward
accounting.

This script deliberately bypasses the `Overcooked` gym wrapper, which needs a
fully-populated training argparse Namespace. It has no dependency on `.env`,
W&B or SLURM, so it runs anywhere the `zsceval` package imports.

Usage:
    # all four objectives non-zero
    python zsceval/scripts/overcooked/morl/rollout_objectives.py

    # negative control: random agents barely score anything
    python zsceval/scripts/overcooked/morl/rollout_objectives.py --agents random

    # random0 is fully partitioned, so it is the clearest coordination demo:
    # agent 0 works the dispenser side, agent 1 relays onions across the wall
    python zsceval/scripts/overcooked/morl/rollout_objectives.py --layout random0 \
        --script-agents place_onion_and_deliver_soup put_onion_everywhere

    python zsceval/scripts/overcooked/morl/rollout_objectives.py \
        --episodes 3 --horizon 200 --objectives task_completion,coordination

Exits non-zero if any consistency check fails.
"""

import argparse
import os
import random
import sys
from typing import Callable, Dict, List, Sequence

import numpy as np

# Allow direct execution (`python .../rollout_objectives.py`) as well as
# `python -m zsceval.scripts.overcooked.morl.rollout_objectives`.
if __package__ in (None, ""):
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
    )

from zsceval.envs.morl.objectives import make_objective_vector  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import OvercookedEnv  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    BASE_REW_SHAPING_PARAMS,
    SHAPED_INFOS,
    OvercookedGridworld,
)
from zsceval.envs.overcooked.script_agent.script_agent import (
    SCRIPT_AGENTS,
)  # noqa: E402

# Layouts backed by `zsceval/envs/overcooked/` (the "old" env this script drives).
#
# Note that `random0` (the layout the pipelines train on) is fully partitioned --
# the onion and dish dispensers sit on agent 0's side of a solid counter wall, the
# pots and serving hatch on agent 1's. No solo policy can deliver there, which is
# exactly why it is the best `coordination` demo but a poor default for scripted
# single-agent play. `random1` is connected, so one scripted policy can run the
# whole loop and light up every objective.
OLD_LAYOUTS = [
    "random0",
    "random0_medium",
    "random1",
    "random3",
    "small_corridor",
    "unident_s",
]

#: Default scripted policy: picks up onions, fills pots, plates and delivers.
DEFAULT_SCRIPT_AGENT = "place_onion_and_deliver_soup"

#: An agent is any callable (mdp, state, player_idx) -> Action, mirroring the
#: `BaseScriptAgent.step` signature so script agents drop straight in.
AgentFn = Callable[[object, object, int], object]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_env(
    layout: str = "random0",
    horizon: int = 400,
    objectives: str = "default",
) -> OvercookedEnv:
    """Construct an `OvercookedEnv` with the MORL objective vector attached.

    Mirrors how the `Overcooked` gym wrapper builds its base env: reward shaping
    is switched on via `BASE_REW_SHAPING_PARAMS`, and the MDP is handed over as a
    *factory* rather than an instance, because `OvercookedEnv.reset()` calls
    `self.mdp_generator_fn()` on every episode.
    """
    mdp_params = {
        "layout_name": layout,
        "start_order_list": None,
        "rew_shaping_params": BASE_REW_SHAPING_PARAMS,
    }
    mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
    return OvercookedEnv(mdp_fn, horizon=horizon, objectives=objectives)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _make_random_agent(rng: np.random.RandomState) -> AgentFn:
    """Uniform over the full action set, INTERACT and STAY included.

    Note: the built-in `overcooked_ai_py.agents.agent.RandomAgent` samples only
    the four movement directions, so it can never fire a single objective. Do
    not substitute it here.
    """

    def act(mdp, state, player_idx):
        return Action.ALL_ACTIONS[rng.randint(len(Action.ALL_ACTIONS))]

    return act


def _make_script_agent(name: str) -> AgentFn:
    """Wrap a `SCRIPT_AGENTS` entry, resetting it lazily on the first call."""
    if name not in SCRIPT_AGENTS:
        raise KeyError(
            f"Unknown script agent {name!r}. Available: {sorted(SCRIPT_AGENTS)}"
        )
    agent = SCRIPT_AGENTS[name]()
    state = {"started": False}

    def act(mdp, s, player_idx):
        if not state["started"]:
            agent.reset(mdp, s, player_idx)
            state["started"] = True
        return agent.step(mdp, s, player_idx)

    return act


def make_agents(
    kind: str,
    rng: np.random.RandomState,
    script_names: Sequence[str] = (DEFAULT_SCRIPT_AGENT,),
    num_players: int = 2,
) -> List[AgentFn]:
    """Build the joint policy.

    Args:
        kind: One of

            * ``"script"`` -- agent ``i`` runs ``script_names[i]``. This is the
              default because uniformly random agents essentially never complete
              a delivery within a 400-step horizon, which would leave
              `task_completion` at zero and demonstrate nothing.
            * ``"random"`` -- both agents act uniformly at random. A useful
              negative control: the objective vector should stay near zero.
            * ``"mixed"`` -- agent 0 scripted, agent 1 random, which exercises
              asymmetric per-agent credit assignment.
        script_names: One name (broadcast to every agent) or one per agent.
            Differing names matter on partitioned layouts, where each side of
            the counter wall needs a different role.
    """
    if kind == "random":
        return [_make_random_agent(rng) for _ in range(num_players)]

    names = list(script_names)
    if len(names) == 1:
        names = names * num_players
    if len(names) != num_players:
        raise ValueError(
            f"--script-agents expects 1 or {num_players} names, got {len(names)}"
        )

    if kind == "script":
        return [_make_script_agent(name) for name in names]
    if kind == "mixed":
        return [_make_script_agent(names[0])] + [
            _make_random_agent(rng) for _ in range(num_players - 1)
        ]
    raise ValueError(f"Unknown agent kind {kind!r}; expected script, random or mixed")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout(
    env: OvercookedEnv, agents: Sequence[AgentFn], verbose: bool = False
) -> Dict:
    """Run one episode to termination and collect the objective bookkeeping.

    Returns:
        A dict with the objective names, the (num_players, K) cumulative
        objective matrix, the Mirror Descent proportion vector `g`, and the
        episode info block the environment produced.
    """
    env.reset()
    names = env.objectives.names
    episode_info = None
    steps = 0

    while not env.is_done():
        joint_action = tuple(
            agent(env.mdp, env.state, i) for i, agent in enumerate(agents)
        )
        _, _, done, info = env.step(joint_action)
        steps += 1

        if verbose:
            step_r = info["vec_r_by_agent"]
            if np.any(step_r):
                nonzero = {
                    name: step_r[:, k].tolist()
                    for k, name in enumerate(names)
                    if np.any(step_r[:, k])
                }
                print(f"  t={steps:4d}  {nonzero}")

        if done:
            episode_info = info["episode"]

    return {
        "names": names,
        "cumulative": env.objectives.cumulative,
        "proportions": env.objectives.proportions(),
        "proportions_by_agent": env.objectives.proportions(per_agent=True),
        "episode": episode_info,
        "steps": steps,
        "delivery_reward": env.mdp.delivery_reward,
    }


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


def check_consistency(result: Dict) -> List[str]:
    """Cross-check the objective vector against the env's pre-existing accounting.

    The objective vector is derived from the same `shaped_info` counters the
    environment already accumulates into `ep_category_r_by_agent`, so the two
    must agree exactly. Any drift means the MORL layer and the MDP have got out
    of sync.

    Returns:
        A list of failure messages; empty means everything agreed.
    """
    failures: List[str] = []
    names = result["names"]
    cum = result["cumulative"]
    episode = result["episode"]
    category = np.asarray(episode["ep_category_r_by_agent"])

    def col(name: str) -> np.ndarray:
        return cum[:, names.index(name)]

    def event(key: str) -> np.ndarray:
        return category[:, SHAPED_INFOS.index(key)]

    if "task_completion" in names:
        implied = col("task_completion").sum() * result["delivery_reward"]
        if not np.isclose(implied, episode["ep_sparse_r"]):
            failures.append(
                f"task_completion * delivery_reward = {implied} "
                f"!= ep_sparse_r = {episode['ep_sparse_r']}"
            )

    if "ingredient_prep" in names:
        if not np.allclose(col("ingredient_prep"), event("PLACEMENT_IN_POT")):
            failures.append(
                f"ingredient_prep = {col('ingredient_prep')} "
                f"!= PLACEMENT_IN_POT = {event('PLACEMENT_IN_POT')}"
            )

    if "plating" in names:
        expected = event("USEFUL_DISH_PICKUP") + event("SOUP_PICKUP")
        if not np.allclose(col("plating"), expected):
            failures.append(
                f"plating = {col('plating')} != USEFUL_DISH_PICKUP + SOUP_PICKUP = {expected}"
            )

    if "coordination" in names:
        placed = sum(event(f"put_{o}_on_X") for o in ("onion", "dish", "soup")).sum()
        taken = sum(
            event(f"pickup_{o}_from_X") for o in ("onion", "dish", "soup")
        ).sum()
        # credit_both=True credits two agents per handoff, hence the halving.
        handoffs = col("coordination").sum() / 2.0
        if handoffs > min(placed, taken) + 1e-9:
            failures.append(
                f"coordination implies {handoffs} handoffs > min(placed={placed}, taken={taken})"
            )

    g = result["proportions"]
    if not np.isclose(g.sum(), 1.0):
        failures.append(f"proportions sum to {g.sum()}, expected 1.0")
    if np.any(cum < 0):
        failures.append(
            "objective vector contains negative components; proportions are ill-defined"
        )

    return failures


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: Dict, header: str = "") -> None:
    """Print the cumulative objective scores for one episode."""
    names = result["names"]
    cum = result["cumulative"]
    episode = result["episode"]
    num_players = cum.shape[0]

    if header:
        print(f"\n{header}")
    print("=" * 72)
    print(f"episode length : {result['steps']} steps")
    print(f"ep_sparse_r    : {episode['ep_sparse_r']}  (unchanged scalar reward)")
    print(f"ep_shaped_r    : {episode['ep_shaped_r']}")
    print()

    width = max(len(n) for n in names) + 2
    agent_cols = "".join(f"{'agent' + str(i):>10}" for i in range(num_players))
    print(f"{'objective':<{width}}{agent_cols}{'team':>10}{'share g':>10}")
    print("-" * 72)
    for k, name in enumerate(names):
        cells = "".join(f"{cum[i, k]:>10.2f}" for i in range(num_players))
        print(
            f"{name:<{width}}{cells}{cum[:, k].sum():>10.2f}{result['proportions'][k]:>10.3f}"
        )
    print("-" * 72)
    print(
        f"{'total':<{width}}"
        + "".join(f"{cum[i].sum():>10.2f}" for i in range(num_players))
    )
    print()
    print(
        "g (mirror-descent proportions):", np.round(result["proportions"], 4).tolist()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out Overcooked with a vector-valued reward and print cumulative objective scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--layout",
        default="random1",
        choices=OLD_LAYOUTS,
        help="Overcooked layout. random0 is partitioned and needs two different --script-agents.",
    )
    parser.add_argument("--horizon", type=int, default=400, help="Steps per episode.")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to roll out."
    )
    parser.add_argument(
        "--agents",
        default="script",
        choices=["script", "random", "mixed"],
        help="Joint policy. 'random' rarely triggers any objective; it is the negative control.",
    )
    parser.add_argument(
        "--script-agents",
        nargs="+",
        default=[DEFAULT_SCRIPT_AGENT],
        metavar="NAME",
        help="SCRIPT_AGENTS key(s) for the 'script' and 'mixed' modes: one to broadcast, or one per agent.",
    )
    parser.add_argument(
        "--objectives",
        default="default",
        help="Objective set name, or a comma-separated list of objective names.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every step with a non-zero objective.",
    )
    parser.add_argument(
        "--no-check", action="store_true", help="Skip the consistency checks."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Fail fast on a bad --objectives spec, before building the env.
    objective_names = make_objective_vector(args.objectives).names

    print(f"layout     : {args.layout}")
    print(
        f"agents     : {args.agents}"
        + (f" {args.script_agents}" if args.agents != "random" else "")
    )
    print(f"objectives : {objective_names}")

    env = build_env(
        layout=args.layout, horizon=args.horizon, objectives=args.objectives
    )
    all_failures: List[str] = []
    totals = np.zeros((env.mdp.num_players, len(objective_names)))

    for episode in range(args.episodes):
        seed = args.seed + episode
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        agents = make_agents(args.agents, rng, args.script_agents, env.mdp.num_players)
        result = rollout(env, agents, verbose=args.verbose)
        totals += result["cumulative"]

        print_report(
            result, header=f"Episode {episode + 1}/{args.episodes} (seed {seed})"
        )

        if not args.no_check:
            failures = check_consistency(result)
            if failures:
                all_failures.extend(f"[episode {episode + 1}] {f}" for f in failures)
            else:
                print("\nconsistency checks: PASS")

    if args.episodes > 1:
        print("\n" + "=" * 72)
        print("Mean cumulative objective scores across episodes (team):")
        team_mean = totals.sum(axis=0) / args.episodes
        for name, value in zip(objective_names, team_mean):
            print(f"  {name:<20} {value:>10.2f}")

    if all_failures:
        print("\nconsistency checks: FAIL")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
