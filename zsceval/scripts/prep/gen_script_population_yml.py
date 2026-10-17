"""Write a stage-2 population yml whose partners are scripted, for the fill-in suite.

Each partner is an ordinary population entry -- it carries a policy config so
the policy and trainer pools can hold it like any frozen member -- whose
featurize type names a script, "script:NAME". The env plays the script from the
true state (PartialPolicyEnv during training, the eval featurize types during
evaluation); the entry's network never acts.

The default partners are the Step 7 training set. Dial 5, the generalist and the
clutter script are deliberately left out: Step 8 tests on them.

    python prep/gen_script_population_yml.py unident_s
    python prep/gen_script_population_yml.py unident_s --partners potter server --name pilot
    python prep/gen_script_population_yml.py pantry_cx          # multi-recipe: menu_* cooks, 7 partners
"""

import argparse
import os
import os.path as osp

POLICY_POOL = os.environ.get("POLICY_POOL")

# Readable name -> SCRIPT_AGENTS key.
SCRIPTS = {
    "potter": "place_onion_in_pot",
    "server": "deliver_soup",
    "idle": "idle",
    "dial2": "2onion_8soup_0noise",
    "dial5": "5onion_5soup_0noise",
    "dial8": "8onion_2soup_0noise",
    "generalist": "place_onion_and_deliver_soup",
    "clutter": "put_onion_everywhere",
}
TRAIN_PARTNERS = ["potter", "server", "idle", "dial2", "dial8"]

# The same personalities for multi-recipe layouts (overcooked_new), played by the
# dish-aware menu_* cooks: an onion-only potter would fill a `_cx` pot with an
# off-menu five-onion soup. The two ingredient specialists are the new axis
# those kitchens add, so they train alongside the original five; dial 5, the
# generalist and the clutterer stay held out for the fill-in suite, as before.
NEW_SCRIPTS = {
    "potter": "menu_potter",
    "server": "menu_server",
    "idle": "idle",
    "dial2": "menu_dial2",
    "dial5": "menu_dial5",
    "dial8": "menu_dial8",
    "generalist": "menu_cook",
    "clutter": "put_onion_everywhere",
    "onion_potter": "menu_potter_onion",
    "tomato_potter": "menu_potter_tomato",
}
NEW_TRAIN_PARTNERS = TRAIN_PARTNERS + ["onion_potter", "tomato_potter"]


def included_partners(path, agent_name):
    """The frozen entries of another population yml, verbatim.

    Entries are top-level keys followed by indented fields; the trainable one
    (`agent_name`, or any entry with `train: True`) is left out.
    """
    blocks, cur = [], None
    for line in open(path).read().splitlines():
        if line and not line.startswith((" ", "\t")):
            cur = [line]
            blocks.append(cur)
        elif cur is not None and line.strip():
            cur.append(line)
    out = []
    for b in blocks:
        name = b[0].rstrip(":")
        if name == agent_name or any(l.strip() == "train: True" for l in b[1:]):
            continue
        out += b
    assert out, f"no frozen partners in {path}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument(
        "--partners",
        nargs="+",
        default=None,
        choices=sorted(set(SCRIPTS) | set(NEW_SCRIPTS)) + ["none"],
        help="Default: the layout's training set (TRAIN_PARTNERS, or NEW_TRAIN_PARTNERS on a multi-recipe layout)",
    )
    ap.add_argument("--name", default="scripted", help="Writes fcp/s2/train-{name}.yml")
    ap.add_argument("--agent_name", default="fcp_adaptive")
    ap.add_argument(
        "--include",
        default=None,
        help="Pool-relative population yml whose frozen partners are added before the "
        "scripts (e.g. {layout}/fcp/s2/train-bench_sp.yml): the usual learned population "
        "plus the specialists. Its trainable entry is dropped; this file writes its own.",
    )
    ap.add_argument(
        "--family",
        type=int,
        default=0,
        help="Add K members of the parametrised specialist family (family.sample: the "
        "extremes first, then spread-out interior points). With --partners none, the "
        "population is the family alone.",
    )
    ap.add_argument("--family_seed", type=int, default=0)
    ap.add_argument(
        "--order_family",
        type=int,
        default=0,
        help="Add K members of the order-aware family for the timed-order env "
        "(envs/overcooked_new/script_agent/order_family.py). Multi-recipe layouts only; "
        "use with --partners none and --config_suffix _timed.",
    )
    ap.add_argument(
        "--config_suffix",
        default="",
        help="Appended to every policy config name, e.g. _timed for "
        "{rnn,mlp}_policy_config_timed.pkl (prep/store_policy_config.py --timed_orders).",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Copies of each script entry. Partners are sampled uniformly per entry, so "
        "with --include this sets the share of games against specialists: 18 learned "
        "partners plus 5 scripts x 4 copies puts about half the games against scripts.",
    )
    args = ap.parse_args()

    assert POLICY_POOL, "POLICY_POOL is unset"
    from zsceval.overcooked_config import OLD_LAYOUTS

    scripts, default = (SCRIPTS, TRAIN_PARTNERS) if args.layout in OLD_LAYOUTS else (NEW_SCRIPTS, NEW_TRAIN_PARTNERS)
    args.partners = args.partners or default
    unknown = [p for p in args.partners if p != "none" and p not in scripts]
    assert not unknown, f"no {unknown} script for {args.layout}'s env version"
    cfg = f"{args.layout}/policy_config"
    lines = [
        f"{args.agent_name}:",
        f"    policy_config_path: {cfg}/rnn_policy_config{args.config_suffix}.pkl",
        "    featurize_type: ppo",
        "    train: True",
    ]
    if args.include:
        lines += included_partners(osp.join(POLICY_POOL, args.include), args.agent_name)
    partners = [p for p in args.partners if p != "none"]
    keys = {p: scripts[p] for p in partners}
    if args.family:
        from zsceval.envs.overcooked.script_agent import family

        keys.update({m: m for m in family.sample(args.family, seed=args.family_seed)})
    if args.order_family:
        from zsceval.envs.overcooked_new.script_agent import order_family

        keys.update({m: m for m in order_family.sample(args.order_family, seed=args.family_seed)})
    for p in keys:
        for c in range(args.repeat):
            lines += [
                f"script_{p}" + (f"_c{c + 1}" if args.repeat > 1 else "") + ":",
                f"    policy_config_path: {cfg}/mlp_policy_config{args.config_suffix}.pkl",
                f"    featurize_type: script:{keys[p]}",
                "    train: False",
            ]
    out = osp.join(POLICY_POOL, args.layout, "fcp", "s2", f"train-{args.name}.yml")
    os.makedirs(osp.dirname(out), exist_ok=True)
    # Concurrent array tasks all write this file; a rename is atomic, so none
    # of them can read a half-written population.
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out)
    print(f"{args.name} {len(keys)} {out}")
    # The swap pool is the same set of scripts, by SCRIPT_AGENTS key.
    print("swap pool: " + ",".join(keys.values()))


if __name__ == "__main__":
    main()
