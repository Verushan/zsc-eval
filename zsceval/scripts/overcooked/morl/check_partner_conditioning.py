"""Checks for --morl_anneal_dense, --use_morl_obs_shares and --morl_adaptive_target complement.

Builds the real env, injects known per-agent objective counts, and checks the
observation widths, the share planes, the per-agent complement update, the
effective (annealed, modulated) weights the reward uses, and that the legacy
adaptive arm is unchanged.

    PYTHONPATH=$PYTHONPATH python morl/check_partner_conditioning.py
"""
import sys
import numpy as np

import os
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.envs.overcooked.Overcooked_Env import Overcooked

CNN = "[[32, 3, 1, 1], [64, 3, 1, 1], [32, 3, 1, 1]]"


def make(extra, layout="unident_s", obj="anchored_live3"):
    argv = ["--env_name", "Overcooked", "--experiment_name", "smoke", "--layout_name", layout,
            "--num_agents", "2", "--seed", "1", "--episode_length", "400", "--overcooked_version", "old",
            "--cnn_layers_params", CNN, "--use_proper_time_limits", "--use_wandb",
            "--morl_objectives", obj] + extra
    parser = get_overcooked_args(get_config())
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--store_traj", default=False, action="store_true")
    args = parser.parse_known_args(argv)[0]
    return Overcooked(args, os.path.join(os.environ.get("TMPDIR", "/tmp"), "check_partner_conditioning"), rank=0, evaluation=False)


def step(env, n=1):
    for _ in range(n):
        obs, share, rew, done, info, avail = env.step([[4], [4]])  # stay
    return tuple(np.asarray(o) for o in obs), np.asarray(share), np.asarray(rew).reshape(-1)


print("== fill arm: anneal + complement + shares + weights")
env = make(["--use_morl", "--morl_weights", "20,3,3", "--morl_anneal_dense",
            "--morl_adaptive_weights", "--morl_adaptive_target", "complement",
            "--use_morl_obs_weights", "--use_morl_obs_shares"])
env.reset()
obs, share, rew = step(env)
assert obs[0].shape == env.ppo_observation_space.shape == (9, 5, 29), obs[0].shape
assert share.shape[-1] == env.share_observation_space[0].shape[-1] == 58
print("widths ok:", obs[0].shape, share.shape)
assert np.allclose(env.morl_weights_by_agent, 1 / 3), "no events -> w untouched"
assert np.allclose(obs[0][0, 0, 20:], 255 / 3), "uniform planes before any event"

# Inject: agent 0 has done all the prep (10) and nothing else; agent 1 all the plating (6).
env.base_env.objectives._cumulative = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 6.0]])
for _ in range(50):
    env.step_count += 1
    env._update_morl_weights()
w = env.morl_weights_by_agent
print("per-agent w after 50 updates:\n", np.round(w, 3))
# agent 0 does only prep; its target is (1/3 task, 2/3 prep, 0 plating): task up, prep down
assert w[0, 0] > 1 / 3 and w[0, 1] < 1 / 3 and np.isclose(w[0, 2], 1 / 3, atol=0.01), w
# agent 1 does only plating; mirror image
assert w[1, 0] > 1 / 3 and w[1, 2] < 1 / 3 and np.isclose(w[1, 1], 1 / 3, atol=0.01), w
eff = env._effective_weights(0)
print("effective w agent0 (base 20,3,3 * K * w_adapt):", np.round(eff, 3))
assert np.allclose(eff, np.array([20, 3, 3]) * 3 * w[0])
planes0 = env._share_planes(obs[0], 255.0, 0)[0, 0]
planes1 = env._share_planes(obs[1], 255.0, 1)[0, 0]
print("share planes agent0 (own, partner):", np.round(planes0), " agent1:", np.round(planes1))
assert np.allclose(planes0, [0, 255, 0, 0, 0, 255]) and np.allclose(planes1, [0, 0, 255, 0, 255, 0])
obs, share, rew = step(env)
print("obs tail agent0 (w*255 | own | partner):", np.round(obs[0][0, 0, 20:]))
assert np.allclose(obs[0][0, 0, 23:], planes0) and np.allclose(obs[0][0, 0, 20:23], w[0] * 255, atol=1)
# reward uses the per-agent effective weights
vec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
r = env._morl_reward(vec)
print("reward for [delivery by 0, prep by 1]:", np.round(r, 3))
assert np.isclose(r[0], env._effective_weights(0)[0]) and np.isclose(r[1], env._effective_weights(1)[1])
# anneal: at factor 0 the dense weights vanish, task stays
env.reward_shaping_factor = 0.0
eff0 = env._effective_weights(0)
print("effective w at anneal 0:", np.round(eff0, 3))
assert np.allclose(eff0[1:], 0.0) and eff0[0] > 0
env.reward_shaping_factor = 0.5
print("effective w at anneal 0.5:", np.round(env._effective_weights(0), 3))
# reset restores uniform per-agent w
env.reset()
assert np.allclose(env.morl_weights_by_agent, 1 / 3)
print("reset ok")

print("\n== ann arm (fixed w, anneal only)")
env = make(["--use_morl", "--morl_weights", "20,3,3", "--morl_anneal_dense"])
env.reset(); obs, share, rew = step(env)
assert obs[0].shape == (9, 5, 20)
env.reward_shaping_factor = 0.25
print("effective w at 0.25:", env._effective_weights(0))
assert np.allclose(env._effective_weights(0), [20, 0.75, 0.75])

print("\n== sp_shares arm (no MORL reward, shares obs only)")
env = make(["--use_morl_obs_shares"])
env.reset(); obs, share, rew = step(env)
assert obs[0].shape == env.ppo_observation_space.shape == (9, 5, 26), obs[0].shape
assert share.shape[-1] == 52
print("widths ok:", obs[0].shape, share.shape, "shaped reward path:", rew)

print("\n== legacy adaptive arm unchanged")
env = make(["--use_morl", "--morl_adaptive_weights", "--use_morl_obs_weights"])
env.reset(); obs, share, rew = step(env)
assert obs[0].shape == (9, 5, 23)
env.base_env.objectives._cumulative = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 6.0]])
env.step_count += 1; env._update_morl_weights()
print("w after one team update:", np.round(env.morl_weights, 4))
assert env.morl_weights[0] > 1 / 3  # task under-realised -> weight rises
print("\nALL OK")
