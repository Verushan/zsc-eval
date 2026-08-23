"""Assertion suite for the MORL training reward path.

`rollout_objectives.py` checks that the *objective vector* the environment
computes agrees with the environment's own event accounting. This script checks
the layer above it: that a `--use_morl` agent is actually **trained on** that
vector rather than on the sparse or shaped reward.

It therefore drives the real `Overcooked` gym wrapper -- the exact class
`train/train_sp.py` hands to the runner -- built from a real argparse Namespace,
and inspects the reward that would land in the PPO buffer.

Checks
------
1. ``provenance``    reward == scale * (w . vec_r) every step, for both
                     agent_idx orderings (catches an ego/player swap bug).
2. ``sp_equivalence``  objectives=task_only, w=[1], scale=20 reproduces
                     ``ep_sparse_r`` exactly -- the scalarization is wired to
                     the right vector.
3. ``sparse_independence``  w=[0,1,0,0] makes the return equal the
                     ``PLACEMENT_IN_POT`` count and provably *not* the sparse
                     reward. This is the "it is not using the sparse reward" test.
4. ``no_morl_regression``  without --use_morl the reward is still
                     ``sparse + reward_shaping_factor * shaped``, and no
                     objective vector is computed at all.
5. ``env_consistency``  the objective vector still agrees with the env's own
                     accounting (delegates to ``rollout_objectives.check_consistency``).
6. ``mirror_descent``  the adaptive preference update keeps w on the simplex,
                     keeps eta in range, moves weight away from over-represented
                     objectives, and resets to the target.

Usage:
    python zsceval/scripts/overcooked/morl/check_morl_reward.py
    python zsceval/scripts/overcooked/morl/check_morl_reward.py --only provenance
    python zsceval/scripts/overcooked/morl/check_morl_reward.py --layout random0 \
        --script-agents place_onion_and_deliver_soup put_onion_everywhere

Note that the sp_equivalence and sparse_independence checks need a layout/script
pair that actually delivers; they report themselves as vacuous otherwise (which
is why the default is random1, not the partitioned random0 the pipelines train on).

Exits non-zero if any check fails.
"""

import argparse
import importlib.util
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Allow direct execution as well as `python -m zsceval.scripts...`.
if __package__ in (None, ""):
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
    )

from zsceval.config import get_config  # noqa: E402
from zsceval.envs.morl.preferences import MirrorDescentPreferences  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    SHAPED_INFOS,
)
from zsceval.envs.overcooked.script_agent.script_agent import (  # noqa: E402
    SCRIPT_AGENTS,
)
from zsceval.overcooked_config import get_overcooked_args  # noqa: E402

#: Connected layout: one scripted policy can run the whole onion -> pot -> plate
#: -> serve loop, so both the sparse reward and every objective are non-zero.
#: That is what makes "the reward is the objectives, not the sparse reward"
#: falsifiable -- on a layout where nothing is ever delivered the two agree
#: trivially at zero.
DEFAULT_LAYOUT = "random1"
DEFAULT_SCRIPT_AGENT = "place_onion_and_deliver_soup"

#: Exact within float noise. The check recomputes the same expression the env
#: does, so anything above this is a real discrepancy, not accumulation error.
ATOL = 1e-9


def _load_rollout_objectives():
    """Import the sibling script by path -- `scripts/` is not a package."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rollout_objectives.py"
    )
    spec = importlib.util.spec_from_file_location("_rollout_objectives", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def build_args(layout: str, horizon: int, extra: Sequence[str] = ()):
    """Parse a training Namespace, exactly as `train/train_sp.py` does.

    Going through the real parsers rather than a hand-built Namespace means this
    script fails if a flag is renamed or a default drifts.
    """
    parser = get_config()
    parser = get_overcooked_args(parser)
    # train_sp.py adds these two outside get_overcooked_args.
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--use_task_v_out", default=False, action="store_true")

    argv = [
        "--env_name", "Overcooked",
        "--algorithm_name", "mappo",
        "--experiment_name", "morl_check",
        "--layout_name", layout,
        "--num_agents", "2",
        "--episode_length", str(horizon),
        "--overcooked_version", "old",
    ]  # fmt: skip
    return parser.parse_args(argv + list(extra))


def build_env(layout: str, horizon: int, extra: Sequence[str] = ()) -> Overcooked:
    return Overcooked(build_args(layout, horizon, extra), run_dir=".")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout(
    env: Overcooked,
    script_names: Sequence[str],
    agent_idx: int = 0,
    seed: int = 0,
) -> List[Tuple[List, Dict]]:
    """Run one episode with scripted agents and return every (reward, info).

    Args:
        env: The gym wrapper under test.
        script_names: `SCRIPT_AGENTS` key per **base env player slot**.
        agent_idx: Which base player slot is the ego agent. `--random_index` is
            off, so `reset()` preserves whatever we set here.

    Note:
        `Overcooked.step` treats `action[0]` as the ego agent's action, so the
        per-player actions are permuted on the way in. The returned rewards are
        permuted the same way, which is exactly what check 1 verifies.
    """
    random.seed(seed)
    np.random.seed(seed)

    env.agent_idx = agent_idx
    env.reset()

    agents = [SCRIPT_AGENTS[name]() for name in script_names]
    for player_idx, agent in enumerate(agents):
        agent.reset(env.base_env.mdp, env.base_env.state, player_idx)

    transitions = []
    while True:
        by_player = [
            Action.ACTION_TO_INDEX[
                agents[p].step(env.base_env.mdp, env.base_env.state, p)
            ]
            for p in range(env.num_agents)
        ]
        action = [[by_player[agent_idx]], [by_player[1 - agent_idx]]]

        _, _, reward, done, info, _ = env.step(action)
        transitions.append((reward, info))

        if done[0]:
            return transitions


def _episode(transitions) -> Dict:
    return transitions[-1][1]["episode"]


def _event_total(episode: Dict, key: str) -> float:
    """Team total of one SHAPED_INFOS counter, from the env's own accounting."""
    category = np.asarray(episode["ep_category_r_by_agent"])
    return float(category[:, SHAPED_INFOS.index(key)].sum())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_provenance(layout: str, horizon: int, scripts: Sequence[str]) -> List[str]:
    """reward == scale * (w . vec_r), every step, for both agent_idx orderings."""
    failures = []
    env = build_env(layout, horizon, ["--use_morl", "--morl_objectives", "default"])
    w, scale = env.morl_weights, env.morl_reward_scale

    for agent_idx in (0, 1):
        transitions = rollout(env, scripts, agent_idx=agent_idx)
        running = np.zeros(env.num_agents)

        for t, (reward, info) in enumerate(transitions):
            vec_r = np.asarray(info["vec_r_by_agent"])
            # reward[0] is the ego agent, i.e. base player `agent_idx`.
            expected = [
                scale * float(w @ vec_r[agent_idx]),
                scale * float(w @ vec_r[1 - agent_idx]),
            ]
            for j, want in enumerate(expected):
                got = float(reward[j][0])
                if not np.isclose(got, want, rtol=0, atol=ATOL):
                    failures.append(
                        f"[agent_idx={agent_idx}] t={t} reward[{j}]={got} != "
                        f"scale*w.vec_r={want}"
                    )
                    break
            # `expected` is in ego order, `running` in base player order.
            running[agent_idx] += expected[0]
            running[1 - agent_idx] += expected[1]

        episode = _episode(transitions)
        ep_vec_r = np.asarray(episode["ep_vec_r_by_agent"])

        if not np.allclose(episode["ep_morl_r_by_agent"], running, rtol=0, atol=1e-6):
            failures.append(
                f"[agent_idx={agent_idx}] ep_morl_r_by_agent="
                f"{np.asarray(episode['ep_morl_r_by_agent']).tolist()} != step sum "
                f"{running.tolist()}"
            )
        if not np.isclose(
            episode["ep_morl_r"], scale * float((ep_vec_r @ w).sum()), rtol=0, atol=1e-6
        ):
            failures.append(
                f"[agent_idx={agent_idx}] ep_morl_r={episode['ep_morl_r']} != "
                f"scale * sum(w . ep_vec_r)={scale * float((ep_vec_r @ w).sum())}"
            )
        if episode["ep_sparse_r"] > 0 and np.isclose(
            episode["ep_morl_r"], episode["ep_sparse_r"]
        ):
            failures.append(
                f"[agent_idx={agent_idx}] uniform-weight ep_morl_r equals ep_sparse_r "
                f"({episode['ep_sparse_r']}); the reward looks like the sparse reward"
            )
        # If both players scored identically the ego/player permutation is
        # untestable: an implementation that always credited player 0 would pass.
        if np.isclose(running[0], running[1]):
            failures.append(
                f"[agent_idx={agent_idx}] both players earned {running[0]}; the "
                "ego/player ordering check is vacuous on this layout/script pair"
            )
    return failures


def check_sp_equivalence(
    layout: str, horizon: int, scripts: Sequence[str]
) -> List[str]:
    """task_only x weight 1 x scale 20 must reproduce ep_sparse_r exactly."""
    failures = []
    env = build_env(
        layout,
        horizon,
        [
            "--use_morl",
            "--morl_objectives", "task_only",
            "--morl_weights", "1",
            "--morl_reward_scale", "20",
        ],  # fmt: skip
    )
    episode = _episode(rollout(env, scripts))

    if not np.isclose(episode["ep_morl_r"], episode["ep_sparse_r"], rtol=0, atol=1e-6):
        failures.append(
            f"task_only ep_morl_r={episode['ep_morl_r']} != "
            f"ep_sparse_r={episode['ep_sparse_r']}"
        )
    if episode["ep_sparse_r"] <= 0:
        failures.append(
            f"no deliveries on {layout} with {list(scripts)}; the sp-equivalence check "
            "is vacuous. Pick a layout/script pair that scores."
        )
    return failures


def check_sparse_independence(
    layout: str, horizon: int, scripts: Sequence[str]
) -> List[str]:
    """Zero weight on task_completion: the return must track PLACEMENT_IN_POT only."""
    failures = []
    env = build_env(
        layout,
        horizon,
        [
            "--use_morl",
            "--morl_objectives", "default",
            "--morl_weights", "0,1,0,0",
        ],  # fmt: skip
    )
    episode = _episode(rollout(env, scripts))
    placements = _event_total(episode, "PLACEMENT_IN_POT")

    if not np.isclose(episode["ep_morl_r"], placements, rtol=0, atol=1e-6):
        failures.append(
            f"ingredient_prep-only ep_morl_r={episode['ep_morl_r']} != "
            f"PLACEMENT_IN_POT={placements}"
        )
    if episode["ep_sparse_r"] <= 0:
        failures.append(
            f"no deliveries on {layout}; cannot show the reward is independent of "
            "the sparse reward"
        )
    elif np.isclose(episode["ep_morl_r"], episode["ep_sparse_r"]):
        failures.append(
            f"ep_morl_r={episode['ep_morl_r']} coincides with "
            f"ep_sparse_r={episode['ep_sparse_r']} despite a zero task_completion weight"
        )
    return failures


def check_no_morl_regression(
    layout: str, horizon: int, scripts: Sequence[str]
) -> List[str]:
    """Without --use_morl nothing changes: same reward, no objective vector."""
    failures = []
    env = build_env(layout, horizon)
    transitions = rollout(env, scripts)

    for t, (reward, info) in enumerate(transitions):
        if "vec_r_by_agent" in info:
            failures.append(
                f"t={t} objective vector computed without --morl_objectives"
            )
            break

        sparse = float(np.sum(info["sparse_r_by_agent"]))
        for player in range(env.num_agents):
            shaped = info["shaped_r_by_agent"][player]
            want = sparse + env.reward_shaping_factor * shaped
            got = float(reward[player][0])
            if not np.isclose(got, want, rtol=0, atol=ATOL):
                failures.append(
                    f"t={t} baseline reward[{player}]={got} != sparse + factor*shaped={want}"
                )
                break

    if "ep_morl_r" in _episode(transitions):
        failures.append("ep_morl_r logged on a run without --use_morl")
    return failures


def check_env_consistency(
    layout: str, horizon: int, scripts: Sequence[str]
) -> List[str]:
    """The objective vector still agrees with the env's own event accounting."""
    check_consistency = _load_rollout_objectives().check_consistency
    env = build_env(layout, horizon, ["--use_morl", "--morl_objectives", "default"])
    transitions = rollout(env, scripts)
    objectives = env.base_env.objectives

    return check_consistency(
        {
            "names": objectives.names,
            "cumulative": objectives.cumulative,
            "proportions": objectives.proportions(),
            "episode": _episode(transitions),
            "delivery_reward": env.base_mdp.delivery_reward,
        }
    )


def check_mirror_descent(
    layout: str, horizon: int, scripts: Sequence[str]
) -> List[str]:
    """Simplex, step-size range, direction, and reset behaviour of the APO update."""
    failures = []

    # -- unit level ---------------------------------------------------------
    prefs = MirrorDescentPreferences(num_objectives=4, eta_min=0.01, eta_max=0.5)
    if not np.allclose(prefs.weights, 0.25):
        failures.append(f"initial weights {prefs.weights.tolist()} are not uniform")

    # Objective 0 takes 70% of the mass: its weight must fall, the rest must rise.
    before = prefs.weights
    after = prefs.update(np.array([0.7, 0.1, 0.1, 0.1]))
    if not (after[0] < before[0] and np.all(after[1:] > before[1:])):
        failures.append(
            f"mirror descent moved weights the wrong way: {before.tolist()} -> {after.tolist()}"
        )
    if not np.isclose(after.sum(), 1.0) or np.any(after < 0):
        failures.append(f"updated weights {after.tolist()} are off the simplex")
    if not prefs.eta_min <= prefs.eta <= prefs.eta_max:
        failures.append(f"eta={prefs.eta} outside [{prefs.eta_min}, {prefs.eta_max}]")

    # A balanced g is the eta_min case; a one-hot g is the eta_max case.
    prefs.update(np.full(4, 0.25))
    if not np.isclose(prefs.eta, prefs.eta_min):
        failures.append(
            f"balanced g gave eta={prefs.eta}, expected eta_min={prefs.eta_min}"
        )
    prefs.update(np.array([1.0, 0.0, 0.0, 0.0]))
    if not np.isclose(prefs.eta, prefs.eta_max):
        failures.append(
            f"one-hot g gave eta={prefs.eta}, expected eta_max={prefs.eta_max}"
        )

    prefs.reset()
    if not np.allclose(prefs.weights, 0.25):
        failures.append(f"reset() left weights at {prefs.weights.tolist()}")

    # The update is multiplicative and runs every env step, so a persistently
    # over-represented objective is driven down hard. The floor must stop it
    # before the weight reaches zero, which would be unrecoverable.
    starved = MirrorDescentPreferences(4, eta_min=0.05, eta_max=0.5, floor=0.02)
    for _ in range(1000):
        weights = starved.update(np.array([0.0, 0.0, 0.0, 1.0]))
    if weights.min() < 0.02 - 1e-12:
        failures.append(
            f"floor breached: min weight {weights.min()} < 0.02 after 1000 updates"
        )
    if not np.isclose(weights.sum(), 1.0):
        failures.append(f"floored weights sum to {weights.sum()}, expected 1.0")

    # -- in the env ---------------------------------------------------------
    env = build_env(
        layout,
        horizon,
        ["--use_morl", "--morl_objectives", "default", "--morl_adaptive_weights"],
    )
    transitions = rollout(env, scripts)
    weights = np.asarray(_episode(transitions)["ep_morl_weights"])

    if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0):
        failures.append(f"end-of-episode weights {weights.tolist()} are off the simplex")
    if np.allclose(weights, 1.0 / len(weights)):
        failures.append(
            f"adaptive weights never moved from uniform ({weights.tolist()}); "
            "the update is not running"
        )

    env.reset()
    if not np.allclose(env.morl_weights, 1.0 / len(weights)):
        failures.append(
            f"reset() left env weights at {env.morl_weights.tolist()}, expected the target"
        )
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CHECKS = {
    "provenance": check_provenance,
    "sp_equivalence": check_sp_equivalence,
    "sparse_independence": check_sparse_independence,
    "no_morl_regression": check_no_morl_regression,
    "env_consistency": check_env_consistency,
    "mirror_descent": check_mirror_descent,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a --use_morl agent is trained on the objective vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--horizon", type=int, default=400, help="Steps per episode.")
    parser.add_argument(
        "--script-agents",
        nargs="+",
        default=[DEFAULT_SCRIPT_AGENT, DEFAULT_SCRIPT_AGENT],
        metavar="NAME",
        help="SCRIPT_AGENTS key per player. Must actually score, or the checks are vacuous.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(CHECKS),
        default=sorted(CHECKS),
        help="Run a subset of the checks.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"layout     : {args.layout}")
    print(f"agents     : {args.script_agents}")
    print(f"horizon    : {args.horizon}")
    print("=" * 72)

    all_failures = {}
    for name in args.only:
        failures = CHECKS[name](args.layout, args.horizon, args.script_agents)
        status = "PASS" if not failures else f"FAIL ({len(failures)})"
        print(f"{name:<22} {status}")
        if failures:
            all_failures[name] = failures

    print("=" * 72)
    if all_failures:
        print("FAILURES")
        for name, failures in all_failures.items():
            for failure in failures:
                print(f"  [{name}] {failure}")
        return 1

    print("all checks passed: the MORL agent's reward is the objective vector")
    return 0


if __name__ == "__main__":
    sys.exit(main())
