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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument("--partners", nargs="+", default=TRAIN_PARTNERS, choices=sorted(SCRIPTS))
    ap.add_argument("--name", default="scripted", help="Writes fcp/s2/train-{name}.yml")
    ap.add_argument("--agent_name", default="fcp_adaptive")
    args = ap.parse_args()

    assert POLICY_POOL, "POLICY_POOL is unset"
    cfg = f"{args.layout}/policy_config"
    lines = [
        f"{args.agent_name}:",
        f"    policy_config_path: {cfg}/rnn_policy_config.pkl",
        "    featurize_type: ppo",
        "    train: True",
    ]
    for p in args.partners:
        lines += [
            f"script_{p}:",
            f"    policy_config_path: {cfg}/mlp_policy_config.pkl",
            f"    featurize_type: script:{SCRIPTS[p]}",
            "    train: False",
        ]
    out = osp.join(POLICY_POOL, args.layout, "fcp", "s2", f"train-{args.name}.yml")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.name} {len(args.partners)} {out}")
    # The swap pool is the same set of scripts, by SCRIPT_AGENTS key.
    print("swap pool: " + ",".join(SCRIPTS[p] for p in args.partners))


if __name__ == "__main__":
    main()
