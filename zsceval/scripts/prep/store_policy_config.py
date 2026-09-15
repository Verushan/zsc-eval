"""Write a layout's `mlp_policy_config.pkl` / `rnn_policy_config.pkl`.

Every entry in an evaluation yml names one of these two files, and
`envs/wrappers/env_policy.py` rebuilds the frozen policy from it: the pickle is
the 4-tuple `(all_args, obs_space, share_obs_space, act_space)` that
`runner/shared/base_runner.py` dumps at the top of a run. The spaces are
layout-shaped -- `random0` is `(5, 5, 20)`, a wider kitchen is not -- so the
files cannot be copied between layouts, and a layout without them cannot be
cross-played at all.

Upstream produces them with `shell/store_config.sh`, which trains two throwaway
agents for 1e7 steps each and then `shell/mv_policy_config.sh` copies the pickle
out of the wreckage. Nothing in the file depends on training having happened:
the spaces come from the env's constructor and the args from the parser. This
builds one env, takes the spaces, and writes the pickle -- seconds instead of
days, and it works for a layout nothing has been trained on yet.

Only the network-shaping arguments matter to a frozen policy, so the two
argument sets below mirror the ones the pool was actually trained under
(`shell/train_morl_benchmark.sh` for the MLP arms, `train_fcp_stage_2.sh` for the
recurrent stage-2 agent) rather than upstream's `store_config.sh` literals. A
pickle written here was verified to differ from `random0`'s existing
`store_config_mlp` pickle only in run-time knobs -- thread counts, horizons,
`run_dir` -- and in none of `hidden_size`, `layer_N`, `cnn_layers_params`,
`use_ReLU`, `use_orthogonal`, `recurrent_N` or `use_recurrent_policy`.

    python prep/store_policy_config.py unident_s
    python prep/store_policy_config.py random3 unident_s --check
"""

import argparse
import os
import os.path as osp
import pickle
import sys

from loguru import logger

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "..", "overcooked"))

from train.train_sp import parse_args  # noqa: E402

from zsceval.config import get_config  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new  # noqa: E402
from zsceval.overcooked_config import OLD_LAYOUTS  # noqa: E402

POLICY_POOL_DIR = os.getenv("POLICY_POOL")

# The CNN trunk every Overcooked policy in this pool uses.
CNN = "32,3,1,1 64,3,1,1 32,3,1,1"

# `--use_recurrent_policy` is `action="store_false"`, so *passing* it turns the
# recurrent policy off. That inversion is why the stage-1 checkpoints are MLPs
# despite the flag appearing in `train_sp.sh`, and why it is absent below from
# the config that has to be recurrent.
CONFIGS = {
    "mlp_policy_config.pkl": ["--algorithm_name", "mappo", "--use_recurrent_policy"],
    "rnn_policy_config.pkl": ["--algorithm_name", "rmappo"],
}


def pid_flags(dim):
    """Actor-side partner-id flags, matching what train_morl_stage_2.sh passes.

    `dim` 0 is the raw scalar the critic already sees (one extra channel); a
    positive `dim` one-hots the id over that many partners and must equal the
    entry count of the population yml the ids came from.
    """
    return ["--use_agent_policy_id_obs", "--agent_policy_id_obs_dim", str(dim)]


def morl_obs_flags(objective_set):
    """Flags for an actor that sees its own live preference vector.

    `--use_morl_obs_weights` appends w to the observation as K constant
    channels, so the width depends on how many objectives the set has and a
    policy trained with it cannot load a config written without it.
    """
    return [
        "--use_morl",
        "--morl_objectives",
        objective_set,
        "--use_morl_obs_weights",
    ]


def build(layout, extra):
    """The 4-tuple `base_runner` pickles, without running a base_runner."""
    version = "old" if layout in OLD_LAYOUTS else "new"
    argv = [
        "--env_name", "Overcooked",
        "--experiment_name", "store_policy_config",
        "--layout_name", layout,
        "--num_agents", "2",
        "--seed", "1",
        "--episode_length", "400",
        "--overcooked_version", version,
        "--cnn_layers_params", CNN,
        "--use_proper_time_limits",
        # store_false: passing it disables wandb, which is what we want here.
        "--use_wandb",
    ] + extra
    all_args = parse_args(argv, get_config())

    # The multi-recipe layouts are a different env package with differently
    # shaped observations, so the class has to follow the version the same way
    # `train_sp.make_train_env` picks it -- building a `*_m` layout through the
    # old class silently writes a pickle with the wrong spaces.
    env_cls = Overcooked if version == "old" else Overcooked_new
    # Overcooked's constructor takes a run_dir only to write layout artifacts it
    # does not write on this path; the env is discarded after its spaces are read.
    env = env_cls(all_args, run_dir=osp.dirname(osp.abspath(__file__)))
    share_obs_space = (
        env.share_observation_space[0]
        if all_args.use_centralized_V
        else env.observation_space[0]
    )
    return (all_args, env.observation_space[0], share_obs_space, env.action_space[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layouts", nargs="+")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would be written; do not touch the pool.",
    )
    parser.add_argument(
        "--pid_obs_dim",
        type=int,
        default=None,
        help="Also write configs for a partner-id-conditioned actor, as "
        "`{mlp,rnn}_policy_config_pid{DIM}.pkl`. --use_agent_policy_id_obs "
        "appends the partner's identity to the *actor's* observation, so a "
        "ladder rung trained with it needs a wider observation than the shared "
        "config describes -- which is why the rungs could not be cross-played. "
        "0 is the raw scalar (one extra channel); a positive value one-hots the "
        "id and must equal the entry count of the population yml the ids came "
        "from. Repeatable via separate invocations, one per rung width.",
    )
    parser.add_argument(
        "--morl_obs_weights",
        default=None,
        metavar="OBJECTIVE_SET",
        help="Write configs for an actor that sees its own live preference "
        "vector, as `{mlp,rnn}_policy_config_mow-{SET}.pkl`. The named objective "
        "set fixes K and so the observation width -- `anchored` gives four extra "
        "channels, `anchored_live3` three. Needed to cross-play a "
        "bench_morl_ad_obs agent, which cannot load the shared config.",
    )
    parser.add_argument(
        "--morl_obs_shares",
        default=None,
        metavar="OBJECTIVE_SET",
        help="Write configs for an actor that sees its own and its partner's "
        "objective mix (2K extra channels), as `*_mos-{SET}.pkl`; combined with "
        "--morl_obs_weights the suffix is `_mow-{SET}_mos`. --use_morl_obs_shares "
        "does not need --use_morl, so this config also serves a hand-shaped "
        "agent given the partner state as an ablation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a config that already exists. Off by default: every policy "
        "in the pool was built from the existing file, and silently swapping the "
        "spaces underneath them would fail later at load_state_dict.",
    )
    args = parser.parse_args()

    assert POLICY_POOL_DIR, "POLICY_POOL is unset; source .env first"

    configs = dict(CONFIGS)
    if args.morl_obs_weights or args.morl_obs_shares:
        extra, suffix = [], ""
        if args.morl_obs_weights:
            extra += morl_obs_flags(args.morl_obs_weights)
            suffix += f"_mow-{args.morl_obs_weights}"
        if args.morl_obs_shares:
            if args.morl_obs_weights and args.morl_obs_weights != args.morl_obs_shares:
                raise SystemExit("--morl_obs_weights and --morl_obs_shares must name the same set")
            extra += ["--morl_objectives", args.morl_obs_shares, "--use_morl_obs_shares"]
            suffix += "_mos" if args.morl_obs_weights else f"_mos-{args.morl_obs_shares}"
        configs = {
            name.replace(".pkl", f"{suffix}.pkl"): base + extra
            for name, base in CONFIGS.items()
        }
    elif args.pid_obs_dim is not None:
        extra = pid_flags(args.pid_obs_dim)
        configs = {
            name.replace(".pkl", f"_pid{args.pid_obs_dim}.pkl"): base + extra
            for name, base in CONFIGS.items()
        }

    for layout in args.layouts:
        out_dir = osp.join(POLICY_POOL_DIR, layout, "policy_config")
        for name, extra in configs.items():
            path = osp.join(out_dir, name)
            config = build(layout, extra)
            shapes = f"obs {config[1].shape} share_obs {config[2].shape} act {config[3]}"
            if osp.exists(path) and not args.overwrite:
                old = pickle.load(open(path, "rb"))
                same = old[1].shape == config[1].shape and old[2].shape == config[2].shape
                logger.info(
                    f"{layout}/{name} exists, {'shapes agree' if same else 'SHAPES DIFFER'}: {shapes}"
                )
                continue
            if args.check:
                logger.info(f"{layout}/{name} would be written: {shapes}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(config, f)
            logger.success(f"wrote {path}: {shapes}")


if __name__ == "__main__":
    main()
