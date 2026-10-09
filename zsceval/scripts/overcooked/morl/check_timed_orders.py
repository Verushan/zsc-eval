"""Checks for the timed-order mechanics in the multi-recipe env.

The mechanics are opt-in (`timed_orders` on the MDP, `--timed_orders` on the
command line), so the first check is that leaving them off changes nothing. The
rest exercise each rule of experiments/report/timed-orders-env-spec.md:

    off        no flag: no queue, 25 planes, static pricing, old to_dict
    lifecycle  first order at t=0, arrivals on schedule, full queue holds back, expiry
    penalty    an expired order costs the team `penalty`, split evenly, once
    pay        speed-scaled pay at both ends of the window; unmatched soup pays 0
    match      a delivery fills the most urgent open order of its recipe
    replay     same seed -> same orders; to_dict/from_dict round-trips
    obs        4 planes per queue slot + 1, integer-valued, most urgent first
    wrapper    the gym env builds from flags, obs space matches obs, episode stats add up

    python zsceval/scripts/overcooked/morl/check_timed_orders.py
    python zsceval/scripts/overcooked/morl/check_timed_orders.py --only pay match
"""

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction  # noqa: E402
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    Order,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    Recipe,
    SoupState,
)

LAYOUT = "unident_s_m"
PARAMS = {"queue": 3, "arrival": 60, "deadline": 150, "penalty": 10, "min_pay": 0.5}


def mdp(timed=True, **overrides):
    p = dict(PARAMS, **overrides)
    return OvercookedGridworld.from_layout_name(LAYOUT, old_dynamics=False, timed_orders=p if timed else None)


def stay(m, state, n=1):
    """Advance n steps with both cooks standing still; returns (state, list of infos)."""
    infos = []
    for _ in range(n):
        state, info = m.get_state_transition(state, [Action.STAY, Action.STAY])
        infos.append(info)
    return state, infos


def recipe(*ingredients):
    return Recipe(list(ingredients))


def serving_pose(m):
    """A floor tile next to a serving window, and the direction that faces it."""
    for pos in m.get_valid_player_positions():
        for d in Direction.ALL_DIRECTIONS:
            if m.get_terrain_type_at_pos(Action.move_in_direction(pos, d)) == "S":
                return pos, d
    raise AssertionError("no serving window reachable")


def delivery_state(m, soup_recipe, orders, timestep=10):
    """Player 0 at a window holding a ready soup of `soup_recipe`; player 1 elsewhere."""
    pos, facing = serving_pose(m)
    other = next(p for p in m.get_valid_player_positions() if p != pos)
    p0 = PlayerState(pos, facing)
    soup = SoupState.get_soup(
        pos,
        num_onions=soup_recipe.ingredients.count("onion"),
        num_tomatoes=soup_recipe.ingredients.count("tomato"),
        finished=True,
    )
    p0.set_object(soup)
    state = OvercookedState(
        [p0, PlayerState(other, Direction.NORTH)],
        {},
        all_orders=m.start_all_orders,
        timestep=timestep,
        orders=[o.to_dict() for o in orders],
        next_order_t=10_000,
    )
    return state


# ---------------------------------------------------------------------------


def check_off():
    m = mdp(timed=False)
    s = m.get_standard_start_state()
    assert m.timed_orders is None and s.orders is None
    assert "orders" not in s.to_dict()
    obs = m.lossless_state_encoding(s, 400)
    assert obs[0].shape[-1] == 25, obs[0].shape
    s2, infos = stay(m, s, 5)
    assert "order_info" not in infos[-1] and s2.orders is None
    # Static pricing is untouched: a menu recipe pays its value.
    r = recipe("onion", "onion", "onion")
    st = delivery_state(m, r, [])
    st.orders = None
    _, info = m.get_state_transition(st, [Action.INTERACT, Action.STAY])
    assert info["sparse_reward_by_agent"][0] == r.value, info["sparse_reward_by_agent"]


def check_lifecycle():
    np.random.seed(0)
    m = mdp()
    s = m.get_standard_start_state()
    assert len(s.orders) == 1 and s.orders[0].arrival == 0 and s.orders[0].deadline == 150
    assert 45 <= s.next_order_t <= 75, s.next_order_t
    arrivals = [0]
    for _ in range(149):
        before = len(s.orders)
        s, _ = stay(m, s)
        if len(s.orders) > before:
            arrivals.append(s.timestep)
    gaps = np.diff(arrivals)
    assert len(s.orders) == 3, s.orders  # the queue fills and holds at its cap
    assert all(45 <= g <= 75 for g in gaps[:2]), arrivals
    # The first order expires the moment the clock reaches its deadline.
    s, infos = stay(m, s)
    assert s.timestep == 150 and infos[-1]["order_info"]["expired"] == 1
    assert all(o.deadline > 150 for o in s.orders)


def check_penalty():
    np.random.seed(1)
    m = mdp(penalty=10)
    s = m.get_standard_start_state()
    total, expired = np.zeros(2), 0
    for _ in range(400):
        s, infos = stay(m, s)
        total += infos[-1]["sparse_reward_by_agent"]
        expired += infos[-1]["order_info"]["expired"]
    assert expired >= 3, expired
    assert np.array_equal(total, -5 * expired * np.ones(2)), (total, expired)


def check_pay():
    m = mdp()
    r = recipe("onion", "onion", "onion")
    assert r.value == 20
    assert m.order_pay(r, 150) == 20  # served on arrival: full value
    assert m.order_pay(r, 0) == 10  # at the deadline: min_pay of it
    assert m.order_pay(r, 75) == 15
    # An unmatched (menu) soup pays nothing and counts as a useless delivery.
    other = recipe("onion", "tomato")
    st = delivery_state(m, r, [Order(other, 0, 150)])
    _, info = m.get_state_transition(st, [Action.INTERACT, Action.STAY])
    assert info["sparse_reward_by_agent"][0] == 0
    assert info["order_info"]["unmatched_by_agent"] == [1, 0]
    assert info["shaped_info_by_agent"][0]["deliver_useless_order"] == 1


def check_match():
    m = mdp()
    r, other = recipe("onion", "onion", "onion"), recipe("onion", "tomato")
    orders = [Order(other, 0, 40), Order(r, 5, 100), Order(r, 0, 60)]
    st = delivery_state(m, r, orders, timestep=30)
    s2, info = m.get_state_transition(st, [Action.INTERACT, Action.STAY])
    # The deadline-60 order is the most urgent of the two matching ones.
    assert [o.deadline for o in s2.orders] == [40, 100], s2.orders
    assert info["order_info"]["delivered_by_agent"] == [1, 0]
    assert info["sparse_reward_by_agent"][0] == m.order_pay(r, 60 - 30) == 12


def check_replay():
    def run(seed):
        np.random.seed(seed)
        m = mdp()
        s = m.get_standard_start_state()
        trace = []
        for _ in range(400):
            s, _ = stay(m, s)
            trace.append(tuple((o.recipe, o.arrival, o.deadline) for o in s.orders))
        return s, trace

    s_a, a = run(7)
    s_b, b = run(7)
    _, c = run(8)
    assert a == b and a != c
    assert OvercookedState.from_dict(s_a.to_dict()) == s_a
    assert s_a.deepcopy() == s_a and hash(s_a.deepcopy()) == hash(s_a)


def check_obs():
    np.random.seed(2)
    m = mdp()
    s = m.get_standard_start_state()
    s, _ = stay(m, s, 130)
    obs = m.lossless_state_encoding(s, 400)
    q = PARAMS["queue"]
    assert obs[0].shape[-1] == 25 + 4 * q + 1, obs[0].shape
    assert tuple(m.get_lossless_state_encoding_shape()) == obs[0].shape
    planes = obs[0][0, 0, 25:]  # constant planes: any cell will do
    assert np.issubdtype(obs[0].dtype, np.integer)
    for k, o in enumerate(s.orders):
        open_, onions, tomatoes, left = planes[4 * k : 4 * k + 4]
        assert open_ == 1 and onions == o.recipe.ingredients.count("onion")
        assert tomatoes == o.recipe.ingredients.count("tomato")
        assert left == int(np.ceil(10 * (o.deadline - s.timestep) / PARAMS["deadline"]))
    assert all(planes[4 * k] == 0 for k in range(len(s.orders), q))
    assert planes[-1] == int(np.ceil(10 * (400 - s.timestep) / 400))
    # Most urgent first.
    lefts = [planes[4 * k + 3] for k in range(len(s.orders))]
    assert lefts == sorted(lefts), lefts


def check_wrapper():
    from zsceval.config import get_config
    from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked
    from zsceval.overcooked_config import get_overcooked_args

    parser = get_overcooked_args(get_config())
    parser.add_argument("--use_phi", default=False, action="store_true")
    argv = ["--env_name", "Overcooked", "--algorithm_name", "mappo", "--experiment_name", "timed_check",
            "--layout_name", LAYOUT, "--num_agents", "2", "--episode_length", "400",
            "--overcooked_version", "new", "--timed_orders"]  # fmt: skip
    args = parser.parse_args(argv)
    args.old_dynamics = False
    env = Overcooked(args, run_dir=os.environ.get("TMPDIR", "/tmp"))
    obs, share_obs, _avail = env.reset()
    assert env.observation_space[0].shape == obs[0].shape, (env.observation_space[0].shape, obs[0].shape)
    assert obs[0].shape[-1] == 25 + 4 * args.order_queue + 1
    assert env.share_observation_space[0].shape == share_obs[0].shape
    rng = np.random.default_rng(0)
    sparse, info = 0, None
    for _ in range(400):
        obs, share_obs, reward, done, info, _avail = env.step(rng.integers(0, 6, size=(2, 1)))
        sparse += sum(info["sparse_r_by_agent"])
        if done[0]:
            break
    ep = info["episode"]
    assert ep["ep_sparse_r"] == sparse == ep["ep_order_pay"] - args.order_penalty * ep["ep_orders_expired"]
    assert ep["ep_orders_expired"] > 0 and 0 <= ep["ep_on_time_rate"] <= 1
    # The old env refuses the flag rather than ignoring it.
    from zsceval.envs.overcooked.Overcooked_Env import Overcooked as OldOvercooked

    old_argv = [{LAYOUT: "unident_s", "new": "old"}.get(a, a) for a in argv]
    old = parser.parse_args(old_argv)
    try:
        OldOvercooked(old, run_dir=os.environ.get("TMPDIR", "/tmp"))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("old env accepted --timed_orders")


CHECKS = {
    "off": check_off,
    "lifecycle": check_lifecycle,
    "penalty": check_penalty,
    "pay": check_pay,
    "match": check_match,
    "replay": check_replay,
    "obs": check_obs,
    "wrapper": check_wrapper,
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
