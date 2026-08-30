import argparse

from zsceval.config import scientific_notation

OLD_LAYOUTS = [
    "random0",
    "random0_medium",
    "random1",
    "random3",
    "small_corridor",
    "unident_s",
]


def get_overcooked_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--layout_name",
        type=str,
        default="cramped_room",
        help="Name of Submap, 40+ in choice. See /src/data/layouts/.",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument(
        "--use_timestep_feature",
        action="store_true",
        default=False,
        help="add timestep as a feature",
    )
    parser.add_argument(
        "--use_identity_feature",
        action="store_true",
        default=False,
        help="add id as a feature",
    )
    parser.add_argument(
        "--use_agent_policy_id",
        default=False,
        action="store_true",
        help="Add policy id into share obs, default False",
    )
    parser.add_argument(
        "--initial_reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_horizon",
        type=scientific_notation,
        default=2.5e6,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--random_start_prob",
        default=0.0,
        type=float,
        help="Probability to use a random start state, default 0.",
    )
    parser.add_argument("--use_random_terrain_state", default=False, action="store_true")
    parser.add_argument("--use_random_player_pos", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--w0_offset", default=0, type=int)
    parser.add_argument(
        "--w0",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 0 in overcooked env.",
    )
    parser.add_argument(
        "--w1",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 1 in overcooked env.",
    )

    parser.add_argument("--num_initial_state", type=int, default=5)
    parser.add_argument("--replay_return_threshold", type=float, default=0.75)

    # ------------------------------------------------------------------
    # MORL: vector-valued reward (zsceval.envs.morl)
    # ------------------------------------------------------------------
    # `--morl_objectives` alone only *tracks* the objective vector, so an
    # existing sp/fcp/mep run can log per-objective breakdowns without its
    # reward changing. `--use_morl` additionally makes w . r_vec the reward.
    parser.add_argument(
        "--morl_objectives",
        type=str,
        default=None,
        help="Objective set name (e.g. 'default', 'task_only') or a comma-separated list of "
        "objective names. Default None disables the objective vector entirely.",
    )
    parser.add_argument(
        "--use_morl",
        default=False,
        action="store_true",
        help="Use the scalarized objective vector as the RL reward instead of sparse + shaped "
        "reward. Implies --morl_objectives default when that is unset.",
    )
    parser.add_argument(
        "--morl_weights",
        type=str,
        default="",
        help="Comma-separated preference weights over --morl_objectives. Empty means uniform 1/K.",
    )
    # Adaptive weights make the reward non-stationary: w moves during an episode
    # while the agent has no way to see that it moved, so identical observations
    # carry different returns. Putting w in the observation restores the Markov
    # property. Opt-in because it widens the observation space, and a policy
    # trained at one width cannot load a checkpoint saved at another -- every
    # existing agent in the pool was trained without it.
    parser.add_argument(
        "--use_morl_obs_weights",
        default=False,
        action="store_true",
        help="Append the current preference vector w to the observation as K extra "
        "channels. Requires --use_morl.",
    )
    parser.add_argument(
        "--morl_reward_scale",
        type=float,
        default=1.0,
        help="Global multiplier on the scalarized objective reward.",
    )
    parser.add_argument(
        "--morl_adaptive_weights",
        default=False,
        action="store_true",
        help="Adapt the preference weights online with the mirror descent update.",
    )
    # The update is multiplicative and runs every env step, so what moves w is
    # eta * episode_length; these defaults are scaled for the 400-step horizon.
    parser.add_argument(
        "--morl_eta_min",
        type=float,
        default=1e-4,
        help="Mirror descent step size when the objective proportions are perfectly balanced.",
    )
    parser.add_argument(
        "--morl_eta_max",
        type=float,
        default=5e-3,
        help="Mirror descent step size when the objective proportions are maximally imbalanced.",
    )
    parser.add_argument(
        "--morl_weight_floor",
        type=float,
        default=0.01,
        help="Smallest preference weight any objective may hold, so an objective is never "
        "switched off permanently. Must be below 1/K; 0 disables it.",
    )
    parser.add_argument(
        "--morl_weight_update_interval",
        type=int,
        default=1,
        help="Env steps between mirror descent preference updates.",
    )
    return parser
