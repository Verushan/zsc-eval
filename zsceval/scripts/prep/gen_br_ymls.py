#!/usr/bin/env python
"""One best-response training yml per held-out partner.

The denominator of BR-Prox should be the best response to a partner -- an agent
trained specifically to work with it. `analyze_crossplay.py` currently
approximates that with the best score any agent *already in the pool* achieved
against that partner, which is biased low: nothing in the pool was trained to
exploit anyone. Every number built on it is optimistic in the same direction --
BR-Prox overstates how much of the achievable return a method captures, regret
understates the shortfall, and CVaR understates the worst case.

The proxy also misleads about the *partner*, which is how it produced a wrong
conclusion here. On random3 thirteen of sixteen partners had `br_hat` 0, read
at the time as "the layout has no headroom". It means only that nothing in our
pool scored with them: pairing hsp12 with the `w1` partner it trained beside
returns 180.0, and `w1` is never in the cross-play pool because
`gen_crossplay_yml.py` references only `w0`. True headroom on that layout is
large and every method captures approximately none of it -- a far stronger
statement than the one the proxy supported.

Upstream's `gen_train_br_ymls.py` reads a `benchmarks-s{2K}.yml` produced by
the bias-agent selection flow this project replaced with
`prep/select_hsp_seeds.py`, so it cannot run against this pool. This writes the
same thing from the checkpoints that are actually present.

Each yml holds the trainable `br_agent` and exactly one frozen partner, which
is what `--population_size 1 --adaptive_agent_name br_agent` expects.

    python prep/gen_br_ymls.py random0
    python prep/gen_br_ymls.py random3 --tag final --partners 12 13

Self-play held-out partners (the complex-dish kitchens have no HSP set) are keyed
`sp{seed}_{tag}`, the cross-play pool's `heldout_sp{seed}_{tag}` without the prefix:

    python prep/gen_br_ymls.py pantry_cx --sp_exp bench_sp-cx-lr --sp_seeds 7 8 9 10 11 12 --sp_tags mid final
"""

import argparse
import glob
import os
import os.path as osp
import re

import yaml
from loguru import logger

POLICY_POOL_DIR = os.getenv("POLICY_POOL")


def discover(layout, hsp_exp, tag):
    """Partner indices with a `w0` checkpoint at this tag, ascending.

    Only `w0` is a real partner: `w1` is the plain sparse-reward agent trained
    alongside it as an artefact of how the pair trains, and the cross-play pool
    references only `w0`.
    """
    pattern = osp.join(
        POLICY_POOL_DIR, layout, "hsp", "s1", hsp_exp, f"hsp*_{tag}_w0_actor.pt"
    )
    found = []
    for path in glob.glob(pattern):
        m = re.match(rf"hsp(\d+)_{tag}_w0_actor\.pt$", osp.basename(path))
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def write_pool(layout, out_dir, key, name, actor):
    config = f"{layout}/policy_config/mlp_policy_config.pkl"
    pool = {
        "br_agent": {"policy_config_path": config, "featurize_type": "ppo", "train": True},
        name: {"policy_config_path": config, "featurize_type": "ppo", "train": False, "model_path": {"actor": actor}},
    }
    with open(osp.join(out_dir, f"train_br_{key}.yml"), "w") as f:
        yaml.safe_dump(pool, f, sort_keys=False)


def write_sp(args):
    """One BR yml per self-play held-out checkpoint, keyed sp{seed}_{tag}."""
    assert args.sp_seeds, "--sp_exp needs --sp_seeds"
    out_dir = osp.join(POLICY_POOL_DIR, args.layout, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for seed in args.sp_seeds:
        for tag in args.sp_tags:
            actor = f"{args.layout}/fcp/s1/{args.sp_exp}/sp{seed}_{tag}_actor.pt"
            if not osp.isfile(osp.join(POLICY_POOL_DIR, actor)):
                logger.warning(f"skipping sp{seed}_{tag}: missing {actor}")
                continue
            key = f"sp{seed}_{tag}"
            write_pool(args.layout, out_dir, key, key, actor)
            written.append(key)
    logger.success(f"wrote {len(written)} br ymls to {out_dir}: {written}")
    print(" ".join(written))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument("--hsp_exp", default="hsp-s1")
    ap.add_argument("--tag", default="final")
    ap.add_argument(
        "--partners",
        nargs="*",
        type=int,
        default=None,
        help="Partner indices. Default: every one with a checkpoint.",
    )
    ap.add_argument("--out_subdir", default="br")
    ap.add_argument("--sp_exp", default=None, help="Stage-1 exp under {layout}/fcp/s1 whose seeds are the held-out set")
    ap.add_argument("--sp_seeds", nargs="+", type=int, default=[])
    ap.add_argument("--sp_tags", nargs="+", default=["mid", "final"])
    args = ap.parse_args()

    assert POLICY_POOL_DIR, "POLICY_POOL is unset; source .env first"
    if args.sp_exp:
        return write_sp(args)
    partners = args.partners or discover(args.layout, args.hsp_exp, args.tag)
    if not partners:
        raise SystemExit(
            f"no hsp*_{args.tag}_w0_actor.pt under "
            f"{args.layout}/hsp/s1/{args.hsp_exp}"
        )

    out_dir = osp.join(POLICY_POOL_DIR, args.layout, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for i in partners:
        actor = f"{args.layout}/hsp/s1/{args.hsp_exp}/hsp{i}_{args.tag}_w0_actor.pt"
        if not osp.isfile(osp.join(POLICY_POOL_DIR, actor)):
            logger.warning(f"skipping partner {i}: missing {actor}")
            continue
        write_pool(args.layout, out_dir, i, f"hsp{i}_{args.tag}_w0", actor)
        written.append(i)

    logger.success(
        f"wrote {len(written)} br ymls to {out_dir} for partners {written}"
    )
    # The training loop reads this to know which indices exist.
    print(" ".join(str(i) for i in written))


if __name__ == "__main__":
    main()
