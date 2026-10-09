"""Scripted cooks that read the timed-order queue.

The other multi-recipe scripts ignore recipes: they fill pots with whatever is
at hand and serve whatever is ready, so with timed orders they mostly cook
soups nobody ordered. `OrderCook` decides afresh every step:

    holding a soup        take it to a window
    holding a plate       scoop a ready soup, wait by a cooking one, or put the plate down
    holding an ingredient put it in a pot whose target recipe still needs it, else put it down
    empty-handed          start a pot whose contents complete its target recipe,
                          fetch a plate if a soup is ready or cooking,
                          fetch the next missing ingredient for the most wanted order

Targets: open orders are ranked by `preference` ("urgent" = earliest deadline,
"valuable" = highest value, then urgency, "blind" = the menu in a fixed random
order, ignoring the queue). Orders already being cooked or
waiting in a pot are crossed off; idle pots are matched to the remaining orders
whose recipe contains what is already in them, empty pots to the next ones.
`jobs` restricts which of "pot" and "serve" the cook does, which is how the
order-aware specialists are built from the same rules.

Needs `state.orders` (an MDP built with timed_orders); without a queue the
layout's menu is used, all equally urgent.
"""

import random
from collections import Counter

import numpy as np

import zsceval.envs.overcooked_new.script_agent.utils as utils
from zsceval.envs.overcooked_new.script_agent.base import BaseScriptAgent
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction

INGREDIENTS = ("onion", "tomato")


def _missing(target, contents):
    """Ingredients still needed to turn `contents` into `target`, or None if impossible."""
    need = Counter(target) - Counter(contents)
    extra = Counter(contents) - Counter(target)
    return None if extra else need


class OrderCook(BaseScriptAgent):
    def __init__(self, preference="urgent", jobs=("pot", "serve")):
        super().__init__()
        assert preference in ("urgent", "valuable", "blind")
        assert set(jobs) <= {"pot", "serve"} and jobs
        self.preference = preference
        self.jobs = tuple(jobs)

    def reset(self, mdp, state, player_idx):
        self.last_pos = state.players[player_idx].position
        self.stuck_time = 0
        # "blind" ignores the queue: it works through the menu in an order fixed
        # for the episode, the way an order-unaware cook would.
        self.menu = list(state.all_orders)
        random.shuffle(self.menu)

    # -- planning -----------------------------------------------------------

    def _ranked_orders(self, state):
        if self.preference == "blind":
            return [(r, 0) for r in self.menu]
        if state.orders is not None:
            orders = [(o.recipe, o.deadline) for o in state.orders]
        else:
            orders = [(r, 0) for r in state.all_orders]
        if self.preference == "valuable":
            return sorted(orders, key=lambda o: (-o[0].value, o[1]))
        return sorted(orders, key=lambda o: o[1])

    def _plan(self, mdp, state):
        """Pot -> (target recipe ingredients, missing Counter) for idle/empty pots,
        plus the lists of ready and cooking pots."""
        remaining = [r for r, _ in self._ranked_orders(state)]
        ready, cooking, idle, empty = [], [], [], []
        for pos in mdp.get_pot_locations():
            if not state.has_object(pos):
                empty.append(pos)
                continue
            soup = state.get_object(pos)
            if soup.is_ready:
                ready.append(pos)
            elif soup.is_cooking:
                cooking.append(pos)
            else:
                idle.append(pos)
            if soup.is_ready or soup.is_cooking:
                # Already on its way: cross off one order for that recipe.
                for i, r in enumerate(remaining):
                    if r == soup.recipe:
                        del remaining[i]
                        break
        targets = {}
        for pos in idle:
            contents = state.get_object(pos).ingredients
            for i, r in enumerate(remaining):
                need = _missing(r.ingredients, contents)
                if need is not None:
                    targets[pos] = (r.ingredients, need)
                    del remaining[i]
                    break
        for pos in empty:
            if remaining:
                r = remaining.pop(0)
                targets[pos] = (r.ingredients, Counter(r.ingredients))
        return targets, ready, cooking

    # -- moving -------------------------------------------------------------

    def _go(self, mdp, state, idx, terrain, objs, only=None):
        mask = None
        if only is not None:
            mask = np.zeros((len(mdp.terrain_mtx), len(mdp.terrain_mtx[0])), dtype=np.int32)
            mask[only[1], only[0]] = 1
        action, _ = utils.interact(mdp, state, idx, None, False, terrain, objs, pos_mask=mask)
        return action

    def _decide(self, mdp, state, idx):
        player = state.players[idx]
        held = player.get_object().name if player.has_object() else None
        targets, ready, cooking = self._plan(mdp, state)
        serve, pot = "serve" in self.jobs, "pot" in self.jobs

        if held == "soup":
            return self._go(mdp, state, idx, "S", ["can_put"])
        if held == "dish":
            if ready:
                return self._go(mdp, state, idx, "P", ["soup"])
            if cooking:
                return self._go(mdp, state, idx, "P", ["cooking_soup"])
            return self._go(mdp, state, idx, "X", ["can_put"])
        if held in INGREDIENTS:
            for pos, (_target, need) in targets.items():
                if need[held] > 0:
                    return self._go(mdp, state, idx, "P", ["can_put"], only=pos)
            return self._go(mdp, state, idx, "X", ["can_put"])

        # Empty-handed.
        if pot:
            for pos, (_target, need) in targets.items():
                if sum(need.values()) == 0:
                    return self._go(mdp, state, idx, "P", ["idle_soup"], only=pos)
        if serve and ready:
            return self._go(mdp, state, idx, "DX", ["dish"])
        if pot:
            for _pos, (_target, need) in targets.items():
                for ing in INGREDIENTS:
                    if need[ing] > 0:
                        return self._go(mdp, state, idx, "OTX", [ing])
        if serve and cooking:
            return self._go(mdp, state, idx, "DX", ["dish"])
        return Action.STAY

    def step(self, mdp, state, player_idx):
        action = self._decide(mdp, state, player_idx)
        pos = state.players[player_idx].position
        if pos == self.last_pos and action in Direction.ALL_DIRECTIONS:
            self.stuck_time += 1
            if self.stuck_time >= 3:
                action = random.choice(Direction.ALL_DIRECTIONS)
        else:
            self.last_pos = pos
            self.stuck_time = 0
        return action
