import os
import pickle
import warnings

import numpy as np
import torch

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.runner.shared.base_runner import make_trainer_policy_cls

POLICY_POOL_PATH = os.environ["POLICY_POOL"]
ACTOR_POOL_PATH = os.environ.get("EVOLVE_ACTOR_POOL")


class _ScriptSeat:
    """Marks a seat whose actions the env's scripted partner supplies."""

    def reset(self, *args, **kwargs):
        pass

    def register_control_agent(self, *args, **kwargs):
        pass


SCRIPT_SEAT = _ScriptSeat()


def extract(x, a):
    if x is None:
        return x
    return x[a]


class PartialPolicyEnv:
    def __init__(self, args, env):
        self.all_args = args
        self.__env = env
        self.num_agents = args.num_agents
        self.use_agent_policy_id = dict(args._get_kwargs()).get("use_agent_policy_id", False)
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]

        self.policy = [None for _ in range(self.num_agents)]
        self.policy_name = [None for _ in range(self.num_agents)]
        self.mask = np.ones((self.num_agents, 1), dtype=np.float32)
        # Width of the observation each frozen policy was built for, taken from
        # its own policy_config pickle. The env emits one observation width for
        # every seat, but the pool can hold policies trained at different widths
        # -- see the slice in `step`.
        self.policy_obs_width = [None for _ in range(self.num_agents)]

        self.observation_space, self.share_observation_space, self.action_space = (
            self.__env.observation_space,
            self.__env.share_observation_space,
            self.__env.action_space,
        )

    def reset(self, reset_choose=True):
        self.__env._set_agent_policy_id(self.agent_policy_id)
        obs, share_obs, available_actions = self.__env.reset(reset_choose)
        self.mask = np.ones((self.num_agents, 1), dtype=np.float32)
        self.obs, self.share_obs, self.available_actions = (
            obs,
            share_obs,
            available_actions,
        )
        for a in range(self.num_agents):
            policy = self.policy[a]
            if policy is not None:
                policy.reset(1, 1)
                policy.register_control_agent(0, 0)
        return obs, share_obs, available_actions

    def load_policy(self, load_policy_config):
        assert len(load_policy_config) == self.num_agents
        for a in range(self.num_agents):
            if load_policy_config[a] is None:
                if self.policy[a] is SCRIPT_SEAT:
                    self.__env.set_script_agent(a, None)
                self.policy[a] = None
                self.policy_name[a] = None
                self.agent_policy_id[a] = -1.0
            else:
                policy_name, policy_info = load_policy_config[a]
                featurize = str(policy_info.get("featurize_type", ""))
                if featurize.startswith("script:"):
                    # A scripted partner: the env plays it, from the true state.
                    # The population entry still carries a policy config, so the
                    # policy and trainer pools can hold it like any frozen
                    # member, but that network never acts.
                    if policy_name != self.policy_name[a]:
                        self.__env.set_script_agent(a, featurize[len("script:"):])
                        self.policy[a] = SCRIPT_SEAT
                        self.policy_name[a] = policy_name
                        self.agent_policy_id[a] = policy_info["id"]
                        self.policy_obs_width[a] = None
                    continue
                if self.policy[a] is SCRIPT_SEAT:
                    self.__env.set_script_agent(a, None)
                if policy_name != self.policy_name[a]:
                    policy_config_path = os.path.join(POLICY_POOL_PATH, policy_info["policy_config_path"])
                    policy_config = pickle.load(open(policy_config_path, "rb"))
                    policy_args = policy_config[0]
                    _, policy_cls = make_trainer_policy_cls(
                        policy_args.algorithm_name,
                        use_single_network=policy_args.use_single_network,
                    )

                    policy = policy_cls(*policy_config, device=torch.device("cpu"))
                    policy.to(torch.device("cpu"))

                    if "model_path" in policy_info:
                        if self.all_args.algorithm_type == "co-play":
                            path_prefix = POLICY_POOL_PATH
                        else:
                            path_prefix = ACTOR_POOL_PATH
                        model_path = add_path_prefix(path_prefix, policy_info["model_path"])
                        policy.load_checkpoint(model_path)
                    else:
                        warnings.warn(f"Policy {policy_name} does not have a valid checkpoint.")
                    policy = EvalPolicy(policy_args, policy)

                    policy.reset(1, 1)
                    policy.register_control_agent(0, 0)

                    self.policy[a] = policy
                    self.policy_name[a] = policy_name
                    self.agent_policy_id[a] = policy_info["id"]
                    obs_space = policy_config[1]
                    self.policy_obs_width[a] = (
                        obs_space.shape[-1] if getattr(obs_space, "shape", None) else None
                    )

    def _policy_obs(self, a):
        """This seat's observation, trimmed to what its frozen policy expects.

        The env emits one observation width for every seat, but a pool can hold
        policies built at different widths. `--use_agent_policy_id_obs` appends
        the partner's identity to the actor's observation as trailing channels,
        so an agent trained with it needs a wider observation than every partner
        already in the pool, which was trained without it. Without this the
        partner-conditioning ladder cannot be cross-played at all: rungs 2 and 3
        want 21 channels (or more, one-hot) while the shared
        `mlp_policy_config.pkl` every frozen partner is rebuilt from has 20.

        The identity channels are appended, so dropping the trailing ones gives
        a partner exactly the observation it was trained on. Only extra channels
        are ever removed -- a policy expecting *more* than the env provides is a
        real mismatch and is left to fail at its own forward pass rather than be
        padded with zeros the network never saw in training.
        """
        obs = self.obs[a]
        want = self.policy_obs_width[a]
        if want is not None and obs.shape[-1] > want:
            return obs[..., :want]
        return obs

    def step(self, actions):
        for a in range(self.num_agents):
            if self.policy[a] is SCRIPT_SEAT:
                assert actions[a] is None, "Expected None action for policy already set in parallel envs."
                actions[a] = np.array([4])  # placeholder; the env substitutes the script's move
            elif self.policy[a] is not None:
                assert actions[a] is None, "Expected None action for policy already set in parallel envs."
                actions[a] = self.policy[a].step(
                    np.array([self._policy_obs(a)]),
                    [(0, 0)],
                    deterministic=False,
                    masks=np.array([self.mask[a]]),
                    available_actions=np.array([self.available_actions[a]]),
                )[0]
            else:
                assert actions[a] is not None, f"Agent {a} is given NoneType action."
        obs, share_obs, reward, done, info, available_actions = self.__env.step(actions)
        self.obs, self.share_obs, self.available_actions = (
            obs,
            share_obs,
            available_actions,
        )
        done = np.array(done)
        self.mask[done == True] = np.zeros(((done == True).sum(), 1), dtype=np.float32)
        return obs, share_obs, reward, done, info, available_actions

    def render(self, mode):
        if mode == "rgb_array":
            fr = self.__env.render(mode=mode)
            return fr
        elif mode == "human":
            self.__env.render(mode=mode)

    def close(self):
        self.__env.close()

    def anneal_reward_shaping_factor(self, data):
        self.__env.anneal_reward_shaping_factor(data)

    def reset_featurize_type(self, data):
        self.__env.reset_featurize_type(data)

    def seed(self, seed):
        self.__env.seed(seed)
