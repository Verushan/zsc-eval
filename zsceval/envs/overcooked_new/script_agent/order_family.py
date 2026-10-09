"""A parametrised family of order-aware scripted partners for the timed-order env.

The counterpart of envs/overcooked/script_agent/family.py, built on OrderCook
(order_script.py). A member is named

    ofam_{P}{J}_l{L}_w{W}

    P  preference: u = most urgent order first, v = most valuable first,
       b = blind to the queue (works through the menu in a fixed random order)
    J  jobs: p = fills pots only, s = plates and serves only, b = both
    L  laziness, 0-10: share of the game spent idle, in stints of 5-20 steps
    W  wandering, 0-10 tenths: chance per step of a random move instead of the planned one

and built on first lookup by SCRIPT_AGENTS, like the old env's family.
"""

import random
import re

import numpy as np

from zsceval.envs.overcooked_new.script_agent.base import BaseScriptAgent
from zsceval.envs.overcooked_new.script_agent.order_script import OrderCook
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction

NAME = re.compile(r"^ofam_([uvb])([psb])_l(\d+)_w(\d+)$")
PREF = {"u": "urgent", "v": "valuable", "b": "blind"}
JOBS = {"p": ("pot",), "s": ("serve",), "b": ("pot", "serve")}
STINT = (5, 20)


def parse(name):
    m = NAME.match(name)
    if not m:
        return None
    p, j, lz, w = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if not (0 <= lz <= 10 and 0 <= w <= 10):
        return None
    return PREF[p], JOBS[j], lz / 10, w / 10


def name_of(pref, jobs, lazy, wander):
    p = {v: k for k, v in PREF.items()}[pref]
    j = {v: k for k, v in JOBS.items()}[tuple(jobs)]
    return f"ofam_{p}{j}_l{round(lazy * 10)}_w{round(wander * 10)}"


class OrderFamilyMember(BaseScriptAgent):
    def __init__(self, pref, jobs, lazy, wander):
        super().__init__()
        self.inner = OrderCook(preference=pref, jobs=jobs)
        self.lazy, self.wander = lazy, wander
        mean_stint = sum(STINT) / 2
        self.p_idle = 0.0 if lazy <= 0 else (1.0 if lazy >= 1 else lazy / ((1 - lazy) * mean_stint))
        self.idle_left = 0

    def reset(self, mdp, state, player_idx):
        self.inner.reset(mdp, state, player_idx)
        self.idle_left = 0

    def step(self, mdp, state, player_idx):
        if self.lazy >= 1.0:
            return Action.STAY
        if self.idle_left > 0:
            self.idle_left -= 1
            return Action.STAY
        if self.p_idle > 0 and np.random.rand() < self.p_idle:
            self.idle_left = np.random.randint(STINT[0], STINT[1] + 1) - 1
            return Action.STAY
        action = self.inner.step(mdp, state, player_idx)
        if self.wander > 0 and np.random.rand() < self.wander:
            return random.choice(Direction.ALL_DIRECTIONS)
        return action


def make(name):
    params = parse(name)
    if params is None:
        raise KeyError(name)
    return lambda: OrderFamilyMember(*params)


def sample(k, seed=0):
    """k members: the extremes first, then spread-out interior points (max-min distance)."""
    rng = np.random.default_rng(seed)
    corners = [
        ("urgent", ("pot", "serve"), 0.0, 0.0),  # the order-reading oracle
        ("urgent", ("pot",), 0.0, 0.0),  # order-aware potter
        ("urgent", ("serve",), 0.0, 0.0),  # order-aware server
        ("blind", ("pot", "serve"), 0.0, 0.0),  # cooks, ignores the queue
        ("valuable", ("pot", "serve"), 0.0, 0.0),  # chases value over deadlines
        ("urgent", ("pot", "serve"), 1.0, 0.0),  # idle
    ]
    grid = [
        (p, j, lz, w)
        for p in ("urgent", "valuable", "blind")
        for j in (("pot",), ("serve",), ("pot", "serve"))
        for lz in (0.0, 0.3, 0.6)
        for w in (0.0, 0.1, 0.2)
    ]

    def vec(m):
        p, j, lz, w = m
        return np.array([["urgent", "valuable", "blind"].index(p), "pot" in j, "serve" in j, lz / 0.6, w / 0.2], float)

    chosen = list(corners[:k])
    pts = np.array([vec(g) for g in grid])
    while len(chosen) < k:
        c = np.array([vec(m) for m in chosen])
        d = np.min(np.abs(pts[:, None, :] - c[None]).sum(-1), 1) + rng.random(len(pts)) * 1e-6
        chosen.append(grid[int(np.argmax(d))])
    return [name_of(*m) for m in chosen]
