#!/usr/bin/env python
"""Pick a spread subset of HSP bias-agent seeds for a layout.

`train_bias_agent.py` does not sample w0 randomly: it enumerates the product of
the bracketed ranges in shell/train_bias_agents.sh, filters to at most three
non-zero bias terms, and selects `candidates[(seed + w0_offset) % len]`. Two
consequences that a hand-picked seed range gets wrong:

* The enumeration is ordered, and `itertools.product` varies the *last*
  dimension fastest. `sparse_r` is last, so it alternates with period 2 -- any
  even stride aliases onto a single sparse level.
* `seed_max` in the shell script is not the candidate count. For random1 /
  random3 / unident_s it says 176 while the space is 52, so seeds past that wrap
  and retrain duplicates.

It also drops the vectors whose reward makes idling dominant: when
`w[STAY] * episode_length` exceeds what ten deliveries pay
(`w[sparse_r] * delivery_reward * 10`), the agent is being paid more to stand
still than to cook, and it collapses to zero sparse return at any budget.
select_bias_agent_br.py would discard it anyway, so training it is wasted.

Usage:
    python prep/select_hsp_seeds.py random0 -k 16
    python prep/select_hsp_seeds.py unident_s -k 16 --exclude 1 5 2
"""

import argparse
from itertools import product

import numpy as np
from loguru import logger

EVENTS = [
    "put_onion_on_X", "put_dish_on_X", "put_soup_on_X", "pickup_onion_from_X",
    "pickup_onion_from_O", "pickup_dish_from_X", "pickup_dish_from_D",
    "pickup_soup_from_X", "USEFUL_DISH_PICKUP", "SOUP_PICKUP", "PLACEMENT_IN_POT",
    "delivery", "STAY", "MOVEMENT", "IDLE_MOVEMENT", "IDLE_INTERACT_X",
    "IDLE_INTERACT_EMPTY", "sparse_r",
]
I_STAY, I_SPARSE = EVENTS.index("STAY"), EVENTS.index("sparse_r")

# Verbatim from shell/train_bias_agents.sh; keep in sync if that changes.
W0 = {
    "random0": "0,0,0,0,[0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]",
    "random0_medium": "0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]",
    "small_corridor": "0,0,0,0,[-20:0:5],0,[-20:0:5],0,3,5,3,[-20:0],[-0.1:0],0,0,0,0,[0.1:1]",
    "_default": "0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]",
}
EPISODE_LENGTH = 400
DELIVERY_REWARD = 20


def parse_value(s):
    if s.startswith("["):
        return list(map(float, s[1:-1].split(":")))
    return [float(s)]


def candidate_space(layout):
    spec = W0.get(layout, W0["_default"])
    dims, bias_index = [], []
    for i, s in enumerate(spec.split(",")):
        v = parse_value(s)
        dims.append(v)
        if len(v) > 1:
            bias_index.append(i)
    cands = [list(c) for c in product(*dims)]
    bias_index = np.array(bias_index)
    cands = [c for c in cands if sum(np.array(c)[bias_index] != 0) <= 3]
    return cands, bias_index


def idle_dominant(cand, bias_index):
    """True when standing still is the agent's best available play.

    Comparing the STAY term against the sparse term alone is not enough: a
    vector can pay +10 for a dispenser pickup, which sits inside the delivery
    loop and makes acting profitable even when idling out-earns deliveries on
    their own. Seeds 18 and 26 both look idle-dominant by the naive test and
    both trained to 70-117 sparse return. Require that there is no positive
    incentive to act anywhere in the vector before writing one off.
    """
    if cand[I_STAY] <= 0:
        return False
    others = [cand[i] for i in bias_index if i not in (I_STAY, I_SPARSE)]
    if any(w > 0 for w in others):
        return False
    return cand[I_STAY] * EPISODE_LENGTH > cand[I_SPARSE] * DELIVERY_REWARD * 10


def farthest_point(vectors, k, seeded=()):
    """Greedy max-min selection, so the subset spans the space rather than a corner."""
    # Scale each dimension to unit range first: the raw terms span -20..10 and
    # -0.1..0.1, so an unscaled distance is decided entirely by the big ones.
    v = np.asarray(vectors, dtype=float)
    span = v.max(axis=0) - v.min(axis=0)
    span[span == 0] = 1.0
    v = v / span
    chosen = list(seeded)
    if not chosen:
        chosen = [int(np.argmax(np.linalg.norm(v - v.mean(axis=0), axis=1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(v[:, None, :] - v[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("-k", type=int, default=16, help="How many agents to train.")
    ap.add_argument("--exclude", nargs="*", type=int, default=[],
                    help="Seeds already trained; kept out of the new list.")
    ap.add_argument("--keep_idle_dominant", action="store_true",
                    help="Do not drop the vectors that pay more for idling than for cooking.")
    args = ap.parse_args()

    cands, bias_index = candidate_space(args.layout)
    n = len(cands)
    logger.info(f"{args.layout}: {n} candidates, varying {[EVENTS[i] for i in bias_index]}")

    # seed s selects candidate s % n, so index 0 is reached by seed n.
    idx_to_seed = {i: (i if i > 0 else n) for i in range(n)}
    excluded_idx = {s % n for s in args.exclude}

    usable = []
    dropped = 0
    for i, c in enumerate(cands):
        if i in excluded_idx:
            continue
        if idle_dominant(c, bias_index) and not args.keep_idle_dominant:
            dropped += 1
            continue
        usable.append(i)
    if dropped:
        logger.warning(f"dropped {dropped} idle-dominant candidates (idling out-earns cooking)")
    logger.info(f"{len(usable)} usable candidates, selecting {args.k}")

    k = min(args.k, len(usable))
    picked_local = farthest_point([np.array(cands[i])[bias_index] for i in usable], k)
    picked = sorted(usable[j] for j in picked_local)

    print(f"\n{'seed':>5}  " + "  ".join(f"{EVENTS[i][:20]:>20}" for i in bias_index))
    for i in picked:
        print(f"{idx_to_seed[i]:>5}  " + "  ".join(f"{cands[i][j]:>20}" for j in bias_index))
    arr = np.array([np.array(cands[i])[bias_index] for i in picked])
    full = np.array([np.array(c)[bias_index] for c in cands])
    print(f"\ndistinct values per dim: selected {[len(set(arr[:, j])) for j in range(arr.shape[1])]}"
          f"  vs whole space {[len(set(full[:, j])) for j in range(full.shape[1])]}")
    print("\nseeds: " + " ".join(str(idx_to_seed[i]) for i in picked))


if __name__ == "__main__":
    main()
