"""Checks for the fill-in suite's Step 7 training machinery, on the real env.

1. equivalence  The `tasks` objective set at weights 20,3,3,5, with team delivery
                credit and annealing, pays exactly the hand-shaped reward, step
                by step, for both seats, at shaping factors 1 and 0.5. This is
                why the planned fixed-weight task arm is the hand-shaped arm.
2. neglect      Neglect weights start even, fall for a task the partner does,
                rise for one it leaves undone, never touch delivery, reset per
                episode, and an even split pays the base weights.
3. scripts      A featurize type "script:NAME" installs a scripted partner in
                that seat; a plain one clears it.
4. swaps        With a swap step and probability 1 the partner changes at that
                step, and reset() restores the original.
5. loader       PartialPolicyEnv drives a seat whose population entry is
                scripted, with no network acting for it.

    PYTHONPATH=$PYTHONPATH python morl/check_fillin_training.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
os.environ.setdefault("POLICY_POOL", "/tmp")
from zsceval.config import get_config  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS  # noqa: E402
from zsceval.overcooked_config import get_overcooked_args  # noqa: E402

CNN = "[[32, 3, 1, 1], [64, 3, 1, 1], [32, 3, 1, 1]]"
RUN_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "check_fillin_training")
TASK_W = "20,3,3,5"


def make(extra, layout="unident_s"):
    argv = ["--env_name", "Overcooked", "--experiment_name", "check", "--layout_name", layout,
            "--num_agents", "2", "--seed", "1", "--episode_length", "400", "--overcooked_version", "old",
            "--cnn_layers_params", CNN, "--use_proper_time_limits", "--use_wandb"] + extra
    parser = get_overcooked_args(get_config())
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--store_traj", default=False, action="store_true")
    args = parser.parse_known_args(argv)[0]
    return Overcooked(args, RUN_DIR, rank=0, evaluation=True)


def rewards(step_out):
    return np.asarray(step_out[2], dtype=np.float64).reshape(-1)[:2]


# ---------------------------------------------------------------- 1. equivalence
print("== 1. tasks at 20,3,3,5 + team credit + anneal == hand-shaped")
legacy = make(["--morl_objectives", "tasks"])
morl = make(["--use_morl", "--morl_objectives", "tasks", "--morl_weights", TASK_W,
             "--morl_team_task", "--morl_anneal_dense"])
# Record a scripted game in `legacy` (two scripted cooks, so real deliveries
# happen), then replay the identical joint actions in `morl`.
np.random.seed(0)
import random  # noqa: E402

random.seed(0)
legacy.reset()
scripts = [SCRIPT_AGENTS["place_onion_and_deliver_soup"](), SCRIPT_AGENTS["deliver_soup"]()]
for a, s in enumerate(scripts):
    s.reset(legacy.base_env.mdp, legacy.base_env.state, a)
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402

for factor in (1.0, 0.5):
    legacy.reset(); morl.reset()
    for a, s in enumerate(scripts):
        s.reset(legacy.base_env.mdp, legacy.base_env.state, a)
    legacy.reward_shaping_factor = factor
    morl.reward_shaping_factor = factor
    total_l = np.zeros(2); total_m = np.zeros(2); worst = 0.0; deliveries = 0
    for t in range(400):
        joint = [[Action.ACTION_TO_INDEX[s.step(legacy.base_env.mdp, legacy.base_env.state, a)]] for a, s in enumerate(scripts)]
        rl = rewards(legacy.step(np.array(joint)))
        out = morl.step(np.array(joint))
        rm = rewards(out)
        deliveries += int(np.sum(np.asarray(out[4]["sparse_r_by_agent"]) > 0))
        total_l += rl; total_m += rm
        worst = max(worst, float(np.abs(rl - rm).max()))
    print(f"   factor {factor}: hand-shaped {total_l.round(2)} tasks {total_m.round(2)} "
          f"max per-step diff {worst:.2e} ({deliveries} deliveries)")
    assert deliveries > 0, "the scripted game must deliver for the check to mean anything"
    assert worst < 1e-9, "the tasks reward must equal the hand-shaped reward"

# ---------------------------------------------------------------- 2. neglect
print("\n== 2. neglect weights")
env = make(["--use_morl", "--morl_objectives", "tasks", "--morl_weights", TASK_W, "--morl_team_task",
            "--morl_anneal_dense", "--morl_adaptive_target", "neglect", "--use_morl_obs_weights"])
env.reset()
assert np.allclose(env.morl_weights_by_agent, 0.5)
assert np.allclose(env._effective_weights(0), [20, 3, 3, 5]), env._effective_weights(0)
obs = np.asarray(env.step(np.array([[4], [4]]))[0][0])
assert obs.shape[-1] == 24, obs.shape  # 20 + 4 weight channels
# Partner (seat 1) fills pots 10 times; nobody plates or fetches dishes.
for _ in range(10):
    env._last_vec_r = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    env._neglect_update()
w0, w1 = env.morl_weights_by_agent
print(f"   agent 0 (partner fills pots): {w0.round(3)}   agent 1: {w1.round(3)}")
assert w0[1] < 0.01 and w1[1] > 0.99, "fill_pot: the filler's partner stops being paid for it"
assert np.isclose(w0[0], 0.5) and np.isclose(w1[0], 0.5), "delivery is never weighted"
assert np.isclose(w0[3], 0.5), "no one has plated yet: neutral"
eff = env._effective_weights(0)
print(f"   agent 0 effective weights: {eff.round(3)}")
assert np.isclose(eff[0], 20) and eff[1] < 0.1
# Now agent 0 plates repeatedly: its plating weight rises toward 1 (partner does none).
for _ in range(10):
    env._last_vec_r = np.array([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float64)
    env._neglect_update()
print(f"   agent 0 plating weight after doing all the plating: {env.morl_weights_by_agent[0][3]:.3f}")
assert env.morl_weights_by_agent[0][3] > 0.99
env.reset()
assert np.allclose(env.morl_weights_by_agent, 0.5) and not env._neglect_counts.any()
print("   reset restores even weights")

# The Step 7 rule (prior 0) never relaxes: once the partner has done a task and
# the agent has not, the agent's weight for it stays 0 however long the partner
# has stopped. A prior pseudo-count lets it return toward an even split.
for prior, expect_recovery in ((0.0, False), (0.5, True)):
    env = make(["--use_morl", "--morl_objectives", "tasks", "--morl_weights", TASK_W, "--morl_team_task",
                "--morl_adaptive_target", "neglect", "--morl_neglect_halflife", "10",
                "--morl_neglect_prior", str(prior)])
    env.reset()
    for _ in range(10):  # partner fills pots, then stops for 60 steps
        env._last_vec_r = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        env._neglect_update()
    during = env.morl_weights_by_agent[0][1]
    for _ in range(60):
        env._last_vec_r = np.zeros((2, 4))
        env._neglect_update()
    after = env.morl_weights_by_agent[0][1]
    print(f"   prior {prior}: agent 0 fill_pot weight {during:.2f} while partner fills, {after:.2f} 60 steps after it stops")
    assert (after > 0.4) == expect_recovery, "prior 0 must stay stuck, prior > 0 must relax"
    assert during < 0.2

# ---------------------------------------------------------------- 3. scripts via featurize type
print("\n== 3. scripted partner from a featurize type")
env = make(["--morl_objectives", "tasks"])
env.reset_featurize_type(("ppo", "script:place_onion_in_pot"))
env.reset()
assert env.script_agent[1] is not None and env.script_agent[0] is None
assert env.featurize_type == ("ppo", "ppo")
moved = False
start = env.base_env.state.players[1].position
for _ in range(20):
    env.step(np.array([[4], [4]]))  # seat 1 asks to stay; the script moves it
    moved |= env.base_env.state.players[1].position != start
assert moved, "the script, not the passed action, must drive seat 1"
env.reset_featurize_type(("ppo", "ppo"))
assert env.script_agent[1] is None
print("   installed, drives its seat, cleared by a plain featurize type")

# ---------------------------------------------------------------- 4. swaps
print("\n== 4. swaps")
env = make(["--morl_objectives", "tasks", "--script_swap_steps", "5", "--script_swap_pool",
            "idle,deliver_soup", "--script_swap_prob", "1.0"])
env.set_script_agent(1, "place_onion_in_pot")
env.reset()
names = []
for t in range(8):
    env.step(np.array([[4], [4]]))
    names.append(env._script_current[1])
print(f"   partner by step: {names}")
assert names[3] == "place_onion_in_pot" and names[4] in ("idle", "deliver_soup")
env.reset()
assert env._script_current[1] == "place_onion_in_pot"
print("   swapped at step 5, restored on reset")

# ---------------------------------------------------------------- 5. loader
print("\n== 5. PartialPolicyEnv with a scripted population entry")
from zsceval.envs.wrappers.env_policy import SCRIPT_SEAT, PartialPolicyEnv  # noqa: E402

base = make(["--morl_objectives", "tasks"])
ppe = PartialPolicyEnv(base.all_args if hasattr(base, "all_args") else None, base)
info = {"featurize_type": "script:deliver_soup", "id": 0.5, "policy_config_path": "unused", "train": False}
ppe.load_policy([None, ("script_server", info)])
assert ppe.policy[1] is SCRIPT_SEAT and base.script_agent[1] is not None
ppe.reset()
for _ in range(5):
    ppe.step([np.array([4]), None])
ppe.load_policy([None, None])
assert base.script_agent[1] is None
print("   loader installs the script, steps without a network, clears on unload")
print("\nALL OK")
