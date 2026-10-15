"""Checks for the complex-dish `_cx` layouts and the MORL pieces they use.

The `_cx` layouts (Phase 2a of the 15 Oct plan) keep the `_m` grids but let a
pot hold up to five ingredients, price big mixed dishes far above onion soup,
and pay 0 rather than -10 for a dish that is not on the menu.

    limits     a `_cx` pot holds 5, an `_m` pot still 3 (the limit is class-global)
    pricing    OOOTT pays 60 on `_cx`; an off-menu soup pays 0 there and -10 on `_m`
    copy       copy() and mdp_params keep the recipe settings; `_m` params unchanged
    dish_value the objective is the delivered value / 20, and conflicts with dish counts
    anneal     --morl_anneal_dense in the new env: dense weights follow the shaping factor
    team       --morl_team_task in the new env: the task columns pay both agents
    scripts    a fixed-dish scripted pair cooks its dish; the onion script cooks nothing that pays
    personas   the fill-in personalities: an onion-only potter never touches a tomato and, beside
               its complement, the pair cooks; a parked specialist steps aside (idle_move)

    python zsceval/scripts/overcooked/morl/check_complex_dishes.py
    python zsceval/scripts/overcooked/morl/check_complex_dishes.py --only pricing anneal
"""

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction  # noqa: E402
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    Recipe,
    SoupState,
)

CX, M = "unident_s_cx", "unident_s_m"


def mdp(layout=CX):
    return OvercookedGridworld.from_layout_name(layout, old_dynamics=False)


def serving_pose(m):
    for pos in m.get_valid_player_positions():
        for d in Direction.ALL_DIRECTIONS:
            if m.get_terrain_type_at_pos(Action.move_in_direction(pos, d)) == "S":
                return pos, d
    raise AssertionError("no serving window reachable")


def deliver(m, onions, tomatoes):
    """Sparse reward and shaped info for player 0 delivering a ready soup."""
    pos, facing = serving_pose(m)
    other = next(p for p in m.get_valid_player_positions() if p != pos)
    p0 = PlayerState(pos, facing)
    p0.set_object(SoupState.get_soup(pos, num_onions=onions, num_tomatoes=tomatoes, finished=True))
    state = OvercookedState([p0, PlayerState(other, Direction.NORTH)], {}, all_orders=m.start_all_orders)
    _, info = m.get_state_transition(state, [Action.INTERACT, Action.STAY])
    return info["sparse_reward_by_agent"][0], info["shaped_info_by_agent"][0]


def gym_env(layout, extra):
    from zsceval.config import get_config
    from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked
    from zsceval.overcooked_config import get_overcooked_args

    parser = get_overcooked_args(get_config())
    parser.add_argument("--use_phi", default=False, action="store_true")
    args = parser.parse_args(["--env_name", "Overcooked", "--algorithm_name", "mappo", "--experiment_name", "cx_check",
                              "--layout_name", layout, "--num_agents", "2", "--episode_length", "400",
                              "--overcooked_version", "new", *extra])  # fmt: skip
    args.old_dynamics = False
    return Overcooked(args, run_dir=os.environ.get("TMPDIR", "/tmp"))


# ---------------------------------------------------------------------------


def check_limits():
    m = mdp(CX)
    assert Recipe.MAX_NUM_INGREDIENTS == 5 and m.max_num_items_for_soup == 5
    pot = SoupState((0, 0), ingredients=[])
    for i in range(5):
        assert not pot.is_full, i
        pot.add_ingredient_from_str("onion" if i < 3 else "tomato")
    assert pot.is_full and sorted(pot.ingredients) == ["onion"] * 3 + ["tomato"] * 2
    mdp(M)
    assert Recipe.MAX_NUM_INGREDIENTS == 3, "building an _m layout after a _cx one must restore the 3-item limit"


def check_pricing():
    m = mdp(CX)
    r, info = deliver(m, 3, 2)
    assert r == 60 and info["delivery"] == 1 and info["deliver_size_three_order"] == 1, (r, info)
    r, info = deliver(m, 5, 0)
    assert r == 0 and info["deliver_useless_order"] == 1, (r, info)
    r, _ = deliver(m, 3, 0)
    assert r == 20
    r, _ = deliver(mdp(M), 1, 0)
    assert r == -10, "the _m layouts keep upstream's -10 for an off-menu dish"


def check_copy():
    m = mdp(CX)
    c = m.copy()
    assert Recipe.MAX_NUM_INGREDIENTS == 5 and c.max_num_items_for_soup == 5 and c.wrong_dish_value == 0
    assert Recipe(["onion"] * 3 + ["tomato"] * 2).value == 60
    assert c.mdp_params["max_num_items_for_soup"] == 5 and c.mdp_params["recipe_values"] == [20, 10, 40, 60, 60]
    keys = set(mdp(M).mdp_params)
    assert keys <= {"layout_name", "terrain", "start_player_positions", "start_bonus_orders", "rew_shaping_params",
                    "start_all_orders", "recipe_values", "recipe_times"}, keys  # fmt: skip


def _scripted(layout, names, objectives="dishes", episodes=1):
    from zsceval.scripts.overcooked.morl import rollout_objectives as ro

    env = ro.build_env(layout=layout, objectives=objectives, version="new")
    rng = np.random.RandomState(0)
    out = []
    for _ in range(episodes):
        agents = ro.make_agents("script", rng, names)
        out.append(ro.rollout(env, agents))
    return out


def check_dish_value():
    big = _scripted(CX, ["dish_OOOTT", "dish_OOOTT"])[0]
    small = _scripted(CX, ["dish_OOO", "dish_OOO"])[0]
    k = big["names"]
    tc, dv = k.index("task_completion"), k.index("dish_value")
    for res in (big, small):
        assert np.isclose(res["cumulative"][:, dv].sum() * 20, res["episode"]["ep_sparse_r"]), res["episode"]
    # The trade-off the preset exists for: more dishes one way, more value the other.
    assert small["cumulative"][:, tc].sum() > big["cumulative"][:, tc].sum()
    assert big["cumulative"][:, dv].sum() > small["cumulative"][:, dv].sum()


def check_anneal():
    w = "0,20,3,3,3,3"
    env = gym_env(CX, ["--use_morl", "--morl_objectives", "dishes", "--morl_weights", w, "--morl_anneal_dense"])
    vec = np.array([[1.0, 3.0, 2.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    wv = np.array([0, 20, 3, 3, 3, 3], dtype=float)
    init = env._initial_reward_shaping_factor
    assert init > 0
    env.reward_shaping_factor = init
    assert np.allclose(env._morl_reward(vec), vec @ wv)
    env.reward_shaping_factor = 0.0
    assert np.allclose(env._morl_reward(vec), vec[:, 1] * 20), "only dish_value may survive the anneal"
    env.reward_shaping_factor = init / 2
    dense = np.array([0, 0, 1, 1, 1, 1], dtype=float)
    assert np.allclose(env._morl_reward(vec), vec @ (wv * (1 - dense * 0.5)))
    plain = gym_env(CX, ["--use_morl", "--morl_objectives", "dishes", "--morl_weights", w])
    plain.reward_shaping_factor = 0.0
    assert np.allclose(plain._morl_reward(vec), vec @ wv), "without the flag nothing is annealed"


def check_team():
    env = gym_env(CX, ["--use_morl", "--morl_objectives", "dishes", "--morl_weights", "1,1,1,1,1,1",
                       "--morl_team_task"])  # fmt: skip
    vec = np.array([[1.0, 3.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    r = env._morl_reward(vec)
    # The delivering agent's task columns (1 + 3) are paid to both; dense stay per agent.
    assert np.allclose(r, [4 + 2, 4 + 1]), r


def check_scripts():
    res = _scripted(CX, ["dish_OOOTT", "dish_OOOTT"])[0]
    assert res["episode"]["ep_sparse_r"] >= 300, res["episode"]["ep_sparse_r"]
    res = _scripted(CX, ["place_onion_and_deliver_soup", "place_onion_and_deliver_soup"])[0]
    assert res["episode"]["ep_sparse_r"] == 0, "five-onion soups are off the menu and pay nothing"


def check_personas():
    import random

    from zsceval.envs.overcooked_new.script_agent.script_agent import SCRIPT_AGENTS

    random.seed(0)
    np.random.seed(0)
    m = mdp("random1_cx")
    state = m.get_standard_start_state()
    agents = [SCRIPT_AGENTS["menu_potter_onion"](), SCRIPT_AGENTS["menu_cook_tomato"]()]
    for i, a in enumerate(agents):
        a.reset(m, state, i)
    sparse, potter_held = 0, set()
    for _ in range(400):
        joint = [a.step(m, state, i) for i, a in enumerate(agents)]
        state, info = m.get_state_transition(state, joint)
        sparse += sum(info["sparse_reward_by_agent"])
        if state.players[0].has_object():
            potter_held.add(state.players[0].get_object().name)
        assert info["shaped_info_by_agent"][0]["pickup_tomato_from_T"] == 0
    assert "tomato" not in potter_held and "onion" in potter_held, potter_held
    assert sparse > 0, "the onion potter and the tomato cook should cook together on random1_cx"
    # Without idle_move the same specialist parks once it has nothing to do.
    assert SCRIPT_AGENTS["menu_potter"]().idle_move and not SCRIPT_AGENTS["order_potter"]().idle_move


CHECKS = {
    "limits": check_limits,
    "pricing": check_pricing,
    "copy": check_copy,
    "dish_value": check_dish_value,
    "anneal": check_anneal,
    "team": check_team,
    "scripts": check_scripts,
    "personas": check_personas,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(CHECKS))
    args = ap.parse_args()
    failed = 0
    for name in args.only or CHECKS:
        try:
            CHECKS[name]()
            print(f"  ok    {name}")
        except Exception as e:  # report every check, not just the first failure
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
