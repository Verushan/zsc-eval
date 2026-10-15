"""The kitchen as a list of jobs: which tasks it needs now, and who did them.

The MORL objectives count *events* an agent produced -- how much prep, how much
plating -- and the adaptive rule steered that mix toward a target. That says
nothing about what the team needed at the moment. A partner who never plates
leaves soups sitting in the pot; an objective share cannot see the sitting soup,
only that this agent has done less plating than some target.

A task is different: it is a job the kitchen *needs doing now*, read from the
state, and it is either done by someone or left undone. For the onion kitchens
there are four, and each maps one-to-one onto an event the env already counts,
so "who did it" needs no new bookkeeping:

    fill_pot    a pot has room for an onion              PLACEMENT_IN_POT
    fetch_dish  a soup is on its way with no dish for it  USEFUL_DISH_PICKUP
    plate_soup  a pot holds a finished soup              SOUP_PICKUP
    deliver     a finished soup is out of the pot         delivery (sparse reward)

Demand is read from the state *before* the joint action, completions from the
step's info, so a step's row says "this was needed, and this is what got done".
The dish rule mirrors the env's own USEFUL_DISH_PICKUP condition exactly (pots
that are cooking, ready or partially full, against dishes already in hands),
so a completed fetch_dish is always a demanded one and the two never disagree.

The multi-recipe env (`overcooked_new`) keeps the same four tasks so that every
analysis reads both, with two differences that follow from its rules:

    fill_pot    a pot is not yet cooking (empty, partially or fully loaded), and
                is done by a *viable* placement -- one that keeps a menu dish
                reachable -- or by starting the cook, which there takes an
                INTERACT. An onion that turns a pot into an off-menu soup is
                not a completed task.
    fetch_dish  the env's own rule there, which leaves out pots that are full
                but not cooking.

The version is read from the objects themselves (the two `get_pot_states`
return differently shaped dicts, and only the new env counts `viable_placement`).
"""

from typing import Dict, Sequence

import numpy as np

TASKS = ("fill_pot", "fetch_dish", "plate_soup", "deliver")
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}

# The env event that completes each task.
COMPLETION_EVENT = {
    "fill_pot": "PLACEMENT_IN_POT",
    "fetch_dish": "USEFUL_DISH_PICKUP",
    "plate_soup": "SOUP_PICKUP",
    "deliver": "delivery",
}


def _pot_counts(mdp, state) -> Dict[str, int]:
    pots = mdp.get_pot_states(state)
    if "onion" in pots:  # the old env: {empty, onion: {...}, tomato: {...}}
        count = lambda key: sum(len(pots[kind][key]) for kind in ("onion", "tomato"))  # noqa: E731
        return {
            "empty": len(pots["empty"]),
            "partial": count("partially_full"),
            "full_idle": 0,  # the old env starts cooking a full pot by itself
            "cooking": count("cooking"),
            "ready": count("ready"),
        }
    # The new env: {empty, "{n}_items" for pots not yet cooking, cooking, ready}.
    return {
        "empty": len(pots["empty"]),
        "partial": len(mdp.get_partially_full_pots(pots)),
        "full_idle": len(mdp.get_full_but_not_cooking_pots(pots)),
        "cooking": len(pots["cooking"]),
        "ready": len(pots["ready"]),
    }


def task_demand(mdp, state) -> np.ndarray:
    """How much of each task the kitchen needs right now, as a (4,) int array.

    fill_pot    pots with room (empty or partially full); in the new env also a
                full pot whose cook has not been started
    fetch_dish  soups on their way minus dishes already in hands, floored at 0,
                and 0 while a dish sits on a counter (the env's own rule for
                when a dish pickup is useful)
    plate_soup  pots holding a finished soup
    deliver     finished soups outside a pot (held, or set on a counter)
    """
    pots = _pot_counts(mdp, state)
    held = state.player_objects_by_type
    dishes_held = len(held["dish"])
    dishes_on_counters = len(mdp.get_counter_objects_dict(state)["dish"])
    on_the_way = pots["cooking"] + pots["ready"] + pots["partial"]
    fetch = 0 if dishes_on_counters else max(0, on_the_way - dishes_held)

    pot_locations = set(mdp.get_pot_locations())
    soups_out = sum(
        1 for obj in state.all_objects_list if obj.name == "soup" and obj.position not in pot_locations
    )
    return np.array(
        [pots["empty"] + pots["partial"] + pots["full_idle"], fetch, pots["ready"], soups_out],
        dtype=np.int64,
    )


def task_completions(shaped_info_by_agent: Sequence[dict], sparse_r_by_agent) -> np.ndarray:
    """Which tasks each agent completed this step, as a (num_agents, 4) int array.

    Deliveries come from the per-agent sparse reward rather than the `delivery`
    counter, for the same reason TaskCompletion uses it: a soup delivered
    against the wrong order earns nothing and is not a completed task.
    """
    sparse = np.asarray(sparse_r_by_agent, dtype=np.float64).reshape(-1)
    out = np.zeros((len(shaped_info_by_agent), len(TASKS)), dtype=np.int64)
    for a, info in enumerate(shaped_info_by_agent):
        for t, task in enumerate(TASKS):
            if task == "deliver":
                out[a, t] = int(sparse[a] > 0)
            elif task == "fill_pot" and "viable_placement" in info:
                # The new env: only a placement that keeps a menu dish reachable,
                # or starting the cook, gets a pot closer to a soup.
                out[a, t] = int(info["viable_placement"] + info.get("cook", 0))
            else:
                out[a, t] = int(info.get(COMPLETION_EVENT[task], 0))
    return out
