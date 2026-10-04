"""A parametrised family of scripted partners, named by their parameters.

The hand-written specialists (potter, server, idle, dial 2/8) gave the best
zero-shot result we have, and every extra one helped. This generalises them so
a population can be drawn from the extremes *and* the interior of a partner
space, instead of five points someone picked. A member is named

    fam_o{O}_n{N}_l{L}_w{W}        each 0-10, read as tenths

and built on demand by SCRIPT_AGENTS, so env, swap pool and population yml
take it like any other script name. The four knobs:

    O  task mix: share of work spent filling pots (the rest is plating and
       serving). O=10 would be a pure potter; 0 a pure server.
    N  noise: share of picked-up items put down somewhere useless instead of
       used (the clutter behaviour, in degrees).
    L  laziness: share of the game spent idle, in stints of 5-20 steps.
    W  wandering: chance per step of a random move instead of the planned one
       -- varies movement habits, which hand-written scripts otherwise share.

O=10 and O=0 use the dedicated potter/server routines: the dial agent's
fallback ("no pot has room, go serve") divides by zero at the extremes.
Members are deterministic given the global numpy/random seeds, like every
other script.
"""

import random
import re

import numpy as np

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.script_agent.base import BaseScriptAgent

NAME = re.compile(r"^fam_o(\d+)_n(\d+)_l(\d+)_w(\d+)$")
STINT = (5, 20)  # idle stint length, steps


def parse(name):
    m = NAME.match(name)
    if not m:
        return None
    o, n, lz, w = (int(g) for g in m.groups())
    if not all(0 <= v <= 10 for v in (o, n, lz, w)):
        return None
    return o / 10, n / 10, lz / 10, w / 10


def name_of(onion, noise, lazy, wander):
    return f"fam_o{round(onion * 10)}_n{round(noise * 10)}_l{round(lazy * 10)}_w{round(wander * 10)}"


class FamilySpecialist(BaseScriptAgent):
    def __init__(self, onion, noise, lazy, wander):
        super().__init__()
        from zsceval.envs.overcooked.script_agent.script_agent import Noisy_Agent, RandomScriptAgent

        self.onion, self.noise, self.lazy, self.wander = onion, noise, lazy, wander
        if onion >= 1.0:
            self.inner = RandomScriptAgent(
                {
                    "pickup_onion_and_place_in_pot": dict(prob=1.0 - noise or 1e-9, args=dict()),
                    "pickup_onion_and_place_random": dict(prob=noise or 1e-9, args=dict()),
                }
            )
        elif onion <= 0.0:
            self.inner = RandomScriptAgent(
                {
                    "pickup_soup_and_deliver": dict(prob=1.0 - noise or 1e-9, args=dict()),
                    "pickup_soup_and_place_random": dict(prob=noise or 1e-9, args=dict()),
                }
            )
        else:
            self.inner = Noisy_Agent(onion, 1.0 - onion, noise)
        # Per-step chance of starting an idle stint, so that the expected idle
        # share is `lazy`: q * S / (1 + q * S) = lazy with S the mean stint.
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
    return lambda: FamilySpecialist(*params)


def sample(k, seed=0, corners=True):
    """k family members: the corners of the space first, then spread-out interior points.

    Corners are the extremes the hypothesis is about (pure potter, pure server,
    fully lazy, noisy). The interior is filled greedily by max-min distance over
    a grid, so the members are as different from each other as k allows.
    """
    rng = np.random.default_rng(seed)
    grid = [
        (o, n, lz, w)
        for o in (0.0, 0.2, 0.5, 0.8, 1.0)
        for n in (0.0, 0.3, 0.6)
        for lz in (0.0, 0.3, 0.6)
        for w in (0.0, 0.1, 0.2)
    ]
    chosen = []
    if corners:
        chosen += [(1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0), (0.5, 0.0, 1.0, 0.0), (0.5, 0.6, 0.0, 0.0)]
    pts = np.array(grid)
    scale = np.array([1.0, 0.6, 0.6, 0.2])
    while len(chosen) < k:
        c = np.array(chosen) if chosen else np.zeros((0, 4))
        d = np.full(len(pts), np.inf) if not len(c) else np.min(np.abs(pts[:, None, :] - c[None]).__truediv__(scale).sum(-1), 1)
        d = d + rng.random(len(pts)) * 1e-6
        chosen.append(tuple(pts[int(np.argmax(d))]))
    return [name_of(*p) for p in chosen[:k]]
