import numpy as np
import torch


def _t2n(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


class EvalPolicy:
    """A policy for evaluation.
    It maintains hidden states on its own.
    For usage, 'reset' before every eval episode, 'register_control_agents' to indicate agents controlled by this policy and 'step' means an env step.
    """

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self._control_agents = []
        self._map_a2id = dict()

        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size

    @property
    def default_hidden_state(self):
        return np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)

    @property
    def control_agents(self):
        return self._control_agents

    def reset(self, num_envs, num_agents):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self._control_agents = []
        self._map_a2id = dict()
        self._rnn_states = dict()

    def reset_state(self, e, a):
        assert (e, a) in self._control_agents
        self._rnn_states[(e, a)] = self.default_hidden_state

    def register_control_agent(self, e, a):
        if (e, a) not in self._control_agents:
            self._control_agents.append((e, a))
            self._map_a2id[(e, a)] = len(self._control_agents)
            self._rnn_states[(e, a)] = self.default_hidden_state

    def fit_obs(self, obs):
        """Trim trailing features this policy's network was not built for.

        Observation-widening flags (`--use_agent_policy_id_obs`,
        `--use_morl_obs_weights`) change the width the environment emits, but a
        frozen population member was built from a pickled policy config at the
        original width. The extra features are always appended last, so a slice
        is exactly the observation this policy was trained on -- and it has no
        use for the extras, since it is not learning.

        Every frozen policy acts through EvalPolicy -- the population evaluation
        path and PartialPolicyEnv both wrap here -- so this is the one place the
        trim is needed outside the training buffers.
        """
        space = getattr(self.policy, "obs_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            return obs
        want, have = shape[-1], obs.shape[-1]
        if have == want:
            return obs
        if have < want:
            raise ValueError(
                f"environment supplies {have} observation features but this policy "
                f"expects {want}; widening flags only ever add features, so this is "
                "a genuine configuration mismatch"
            )
        return obs[..., :want]

    def step(self, obs, agents, deterministic=False, masks=None, **kwargs):
        num = len(agents)
        assert obs.shape[0] == num
        obs = self.fit_obs(obs)
        rnn_states = [self._rnn_states[ea] for ea in agents]
        if masks is None:
            masks = np.ones((num, 1), dtype=np.float32)
        action, rnn_states = self.policy.act(
            obs, np.stack(rnn_states, axis=0), masks, deterministic=deterministic, **kwargs
        )
        for ea, rnn_state in zip(agents, _t2n(rnn_states)):
            self._rnn_states[ea] = rnn_state
        return _t2n(action)

    def to(self, device):
        self.policy.to(device)

    def prep_rollout(self):
        self.policy.prep_rollout()
