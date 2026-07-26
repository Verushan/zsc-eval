"""Vector-valued (multi-objective) reward for Overcooked.

Motivation
----------
The environment natively emits a single scalar: ``+20`` when an order is delivered
(``OvercookedGridworld.delivery_reward``), optionally augmented with hand-tuned
reward shaping. Multi-Objective RL instead needs a reward *vector*
``r_t in R^K``, one component per objective, so that a preference vector ``w``
can be adjusted online in real time

Design
------
The MDP already counts every event we need. ``resolve_interacts`` builds a
per-agent ``shaped_info`` dict keyed by ``SHAPED_INFOS``
(``PLACEMENT_IN_POT``, ``USEFUL_DISH_PICKUP``, ``SOUP_PICKUP``, ``delivery``,
``put_*_on_X``, ``pickup_*_from_X``, idle counters, ...). This module maps those
existing counters onto objectives, so the MDP itself is left untouched --
important, because ``resolve_interacts`` is on the hot path of every SP / FCP /
MEP / HSP training run.

Two invariants are maintained deliberately, both required by the Mirror Descent
preference update ``w_i <- w_i exp(eta (t_i - g_i)) / Z``:

1. **Non-negativity.** ``g_i`` is a *proportion* of objective ``i``; a negative
   component would make it meaningless. Objectives express costs as forgone
   gains rather than penalties.
2. **Commensurate magnitudes.** Components are unit counts by default rather
   than raw reward. If ``task_completion`` contributed the raw ``20`` while every
   other objective contributed ``1``, ``g`` would sit permanently near a vertex of
   the simplex and the preference update would have nothing to do.

Adding an objective
-------------------
Subclass :class:`Objective`, implement ``__call__``, and add one line to
:data:`OBJECTIVE_REGISTRY`. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action

# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveContext:
    """Everything an :class:`Objective` may need to score a single env step.

    Attributes:
        mdp: The ``OvercookedGridworld`` instance driving the episode.
        prev_state: State *before* the joint action was applied. Player poses
            here are the poses at interact-resolution time, since ``INTERACT``
            changes neither position nor orientation.
        next_state: State after the joint action.
        joint_action: Tuple of per-agent ``Action`` values.
        shaped_info_by_agent: Per-agent event counters for *this step only*
            (the ``SHAPED_INFOS`` dict from ``resolve_interacts``).
        sparse_r_by_agent: Per-agent sparse reward for this step (delivery only).
        shaped_r_by_agent: Per-agent scalar shaping reward for this step.
        num_players: Number of agents.
        t: Timestep index within the episode.
    """

    mdp: Any
    prev_state: Any
    next_state: Any
    joint_action: Sequence[Any]
    shaped_info_by_agent: Sequence[Dict[str, int]]
    sparse_r_by_agent: Sequence[float]
    shaped_r_by_agent: Sequence[float]
    num_players: int
    t: int

    def count(self, agent_idx: int, key: str) -> int:
        """Read one event counter, tolerating keys absent from this env version.

        The "old" and "new" Overcooked packages define overlapping but unequal
        ``SHAPED_INFOS`` lists, so every lookup goes through here.
        """
        return int(self.shaped_info_by_agent[agent_idx].get(key, 0))


# ---------------------------------------------------------------------------
# Objective base class
# ---------------------------------------------------------------------------


class Objective(ABC):
    """A single scalar component of the reward vector.

    Subclasses must set :attr:`name` and :attr:`description` and implement
    :meth:`__call__`. Objectives may hold per-episode state, in which case they
    must clear it in :meth:`reset`.
    """

    name: str = "unnamed"
    description: str = ""

    def reset(self, num_players: int) -> None:
        """Clear any per-episode state. Called on every ``env.reset()``."""

    @abstractmethod
    def __call__(self, context: ObjectiveContext) -> np.ndarray:
        """Score one step.

        Returns:
            A ``(num_players,)`` float array of non-negative per-agent rewards.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete objectives
# ---------------------------------------------------------------------------


class TaskCompletion(Objective):
    """Orders successfully delivered.

    Derived from ``sparse_r_by_agent`` rather than the ``delivery`` counter so
    that soups delivered against the wrong order (which earn nothing) are not
    counted as completed tasks, and so that

        ``cumulative[:, task_completion].sum() * delivery_reward == ep_sparse_r``

    holds exactly. ``scale=20.0`` reproduces the raw sparse reward verbatim; the
    default ``scale=1.0`` counts deliveries, which keeps this component
    commensurate with the unit-count objectives below.
    """

    name = "task_completion"
    description = "Orders delivered (the environment's native sparse reward)."

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def __call__(self, context: ObjectiveContext) -> np.ndarray:
        delivery_reward = float(getattr(context.mdp, "delivery_reward", 0.0) or 0.0)
        deliveries = [0]

        if delivery_reward > 0.0:
            deliveries = (
                np.asarray(context.sparse_r_by_agent, dtype=np.float64)
                / delivery_reward
            )
        else:
            # No positive delivery reward configured (e.g. the "new" env prices
            # recipes individually); fall back to the raw delivery counter.
            deliveries = np.array(
                [context.count(i, "delivery") for i in range(context.num_players)],
                dtype=np.float64,
            )
        return np.maximum(deliveries, 0.0) * self.scale


class EventCountObjective(Objective):
    """Sum of one or more ``SHAPED_INFOS`` event counters.

    Covers every stateless objective: the component is just how many times the
    listed events fired for that agent this step.
    """

    def __init__(
        self,
        name: str,
        event_keys: Sequence[str],
        description: str = "",
        scale: float = 1.0,
    ):
        self.name = name
        self.description = description
        self.event_keys = tuple(event_keys)
        self.scale = float(scale)

    def __call__(self, context: ObjectiveContext) -> np.ndarray:
        counts = np.array(
            [
                sum(context.count(i, k) for k in self.event_keys)
                for i in range(context.num_players)
            ],
            dtype=np.float64,
        )
        return counts * self.scale


def IngredientPrep(scale: float = 1.0) -> EventCountObjective:
    """Sub-task efficiency: ingredients placed into a pot."""
    return EventCountObjective(
        name="ingredient_prep",
        event_keys=("PLACEMENT_IN_POT",),
        description="Ingredients (onion/tomato) placed into a cooking pot.",
        scale=scale,
    )


def Plating(scale: float = 1.0) -> EventCountObjective:
    """Sub-task efficiency: getting a finished soup out of the pot and onto a dish."""
    return EventCountObjective(
        name="plating",
        event_keys=("USEFUL_DISH_PICKUP", "SOUP_PICKUP"),
        description="Useful dish pickups plus soups collected from a pot.",
        scale=scale,
    )


class CounterHandoff(Objective):
    """Coordination quality: objects passed between agents via a counter.

    An agent placing an object on a counter (``X``) that the *partner* later
    collects is the canonical Overcooked handoff, and the clearest observable
    signal that the two agents are working as a team rather than in parallel.

    Implementation notes:

    * The trigger is the existing ``put_*_on_X`` / ``pickup_*_from_X`` counters,
      so the terrain and object-ownership logic in ``resolve_interacts`` is never
      duplicated here.
    * The counter position is recovered from ``prev_state``. This is exact:
      ``INTERACT`` changes neither position nor orientation, so an agent's pose
      before the step is its pose at interact-resolution time.
    * Agents are replayed in index order, matching ``resolve_interacts``. That
      makes a same-step handoff (agent 0 places, agent 1 immediately collects)
      score correctly.
    * Two agents cannot place on the same counter in one step -- the second sees
      it occupied and records ``IDLE_INTERACT_X`` instead -- so ownership never
      needs conflict resolution.
    """

    name = "coordination"
    description = "Objects handed to the partner across a counter."

    #: Object names that can be placed on / taken from a counter.
    OBJECT_NAMES = ("onion", "tomato", "dish", "soup")

    def __init__(self, scale: float = 1.0, credit_both: bool = True):
        """
        Args:
            scale: Multiplier applied to each handoff.
            credit_both: If ``True`` both the placing and the collecting agent are
                credited (a handoff is a joint event). If ``False`` only the
                collector is credited, which attributes the value to the point at
                which the exchange actually paid off.
        """
        self.scale = float(scale)
        self.credit_both = credit_both
        self._counter_owner: Dict[Any, int] = {}

    def reset(self, num_players: int) -> None:
        self._counter_owner = {}

    @staticmethod
    def _interact_position(state: Any, agent_idx: int):
        """Counter tile the agent was facing when its INTERACT resolved."""
        player = state.players[agent_idx]
        return Action.move_in_direction(player.position, player.orientation)

    def _placed_on_counter(self, context: ObjectiveContext, agent_idx: int) -> bool:
        return any(
            context.count(agent_idx, f"put_{obj}_on_X") for obj in self.OBJECT_NAMES
        )

    def _took_from_counter(self, context: ObjectiveContext, agent_idx: int) -> bool:
        return any(
            context.count(agent_idx, f"pickup_{obj}_from_X")
            for obj in self.OBJECT_NAMES
        )

    def __call__(self, context: ObjectiveContext) -> np.ndarray:
        reward = np.zeros(context.num_players, dtype=np.float64)

        for agent_index in range(context.num_players):
            placed = self._placed_on_counter(context, agent_index)
            took = self._took_from_counter(context, agent_index)

            if not placed and not took:
                continue

            interaction_pos = self._interact_position(context.prev_state, agent_index)

            if placed:
                self._counter_owner[interaction_pos] = agent_index
            elif took:
                owner_index = self._counter_owner.pop(interaction_pos, None)

                if owner_index is not None and owner_index != agent_index:
                    reward[agent_index] += self.scale

                    if self.credit_both:
                        reward[owner_index] += self.scale

        return reward


# ---------------------------------------------------------------------------
# The vector
# ---------------------------------------------------------------------------


class ObjectiveVector:
    """An ordered collection of :class:`Objective` plus a per-episode accumulator.

    Example:
        >>> vec = make_objective_vector("default")
        >>> vec.reset(num_players=2)
        >>> step_r = vec.step(context)      # (num_players, K)
        >>> vec.cumulative              # (num_players, K)
        >>> vec.proportions()           # (K,) -- the `g` of the Mirror Descent update
    """

    def __init__(self, objectives: Sequence[Objective]):
        if not objectives:
            raise ValueError("ObjectiveVector requires at least one objective")
        names = [o.name for o in objectives]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate objective names: {sorted(duplicates)}")

        self.objectives: List[Objective] = list(objectives)
        self._num_players = 0
        self._cumulative = np.zeros((0, len(self.objectives)), dtype=np.float64)

    # -- introspection ------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return [o.name for o in self.objectives]

    @property
    def descriptions(self) -> Dict[str, str]:
        return {o.name: o.description for o in self.objectives}

    def index(self, name: str) -> int:
        """Position of ``name`` in the reward vector."""
        return self.names.index(name)

    def __len__(self) -> int:
        return len(self.objectives)

    def __repr__(self) -> str:
        return f"ObjectiveVector({self.names})"

    def reset(self, num_players: int) -> None:
        """Clear the accumulator and every objective's internal state."""
        self._num_players = num_players
        self._cumulative = np.zeros(
            (num_players, len(self.objectives)), dtype=np.float64
        )
        for objective in self.objectives:
            objective.reset(num_players)

    def step(self, context: ObjectiveContext) -> np.ndarray:
        """Score one env step and fold it into the accumulator.

        Returns:
            ``(num_players, K)`` float array of this step's per-agent rewards.
        """
        columns = []

        for objective in self.objectives:
            value = np.asarray(objective(context), dtype=np.float64).reshape(-1)

            if value.shape[0] != context.num_players:
                raise ValueError(
                    f"Objective {objective.name!r} returned shape {value.shape}, "
                    f"expected ({context.num_players},)"
                )

            columns.append(value)

        step_reward = np.stack(columns, axis=-1)
        self._cumulative = self._cumulative + step_reward
        return step_reward

    @property
    def cumulative(self) -> np.ndarray:
        """``(num_players, K)`` episode-to-date objective scores."""
        return self._cumulative.copy()

    @property
    def team_cumulative(self) -> np.ndarray:
        """``(K,)`` objective scores summed over agents."""
        return self._cumulative.sum(axis=0)

    def proportions(self, per_agent: bool = False) -> np.ndarray:
        """Normalised objective shares -- the ``g`` term of the Mirror Descent update.

        Args:
            per_agent: Return one distribution per agent instead of one for the team.

        Returns:
            ``(K,)``, or ``(num_players, K)`` when ``per_agent``. Rows sum to 1.
            A row whose total is zero (episode start, or a rollout in which
            nothing happened) falls back to uniform ``1/K`` rather than dividing
            by zero, so the preference update is a no-op there instead of NaN.
        """
        k = len(self.objectives)
        totals = self._cumulative if per_agent else self.team_cumulative.reshape(1, -1)
        row_sums = totals.sum(axis=-1, keepdims=True)
        shares = np.where(
            row_sums > 0, totals / np.where(row_sums > 0, row_sums, 1.0), 1.0 / k
        )
        return shares if per_agent else shares.reshape(k)

    def summary(self) -> Dict[str, Any]:
        """Plain-Python snapshot, convenient for logging or JSON serialisation."""
        return {
            "names": self.names,
            "descriptions": self.descriptions,
            "cumulative_by_agent": self._cumulative.tolist(),
            "team_cumulative": dict(zip(self.names, self.team_cumulative.tolist())),
            "proportions": dict(zip(self.names, self.proportions().tolist())),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# All usable objectives
OBJECTIVE_REGISTRY: Dict[str, Callable[[], Objective]] = {
    "task_completion": TaskCompletion,
    "ingredient_prep": IngredientPrep,
    "plating": Plating,
    "coordination": CounterHandoff,
}

# Named presets
OBJECTIVE_SETS: Dict[str, List[str]] = {
    "default": ["task_completion", "ingredient_prep", "plating", "coordination"],
    "task_only": ["task_completion"],
}


def make_objective_vector(
    spec: Optional[Union[str, Sequence[str], ObjectiveVector]] = None,
) -> Optional[ObjectiveVector]:
    """Build an :class:`ObjectiveVector` from a flexible specification.

    Args:
        spec: One of
            * ``None`` -- multi-objective tracking is disabled (returns ``None``).
            * a key of :data:`OBJECTIVE_SETS`, e.g. ``"default"``.
            * a sequence of :data:`OBJECTIVE_REGISTRY` keys, e.g.
              ``["task_completion", "coordination"]``.
            * a comma-separated string of registry keys.
            * an existing :class:`ObjectiveVector`, returned unchanged.

    Raises:
        KeyError: If any requested objective is not registered.
    """
    if spec is None:
        return None

    if isinstance(spec, ObjectiveVector):
        return spec

    if isinstance(spec, str):
        if spec in OBJECTIVE_SETS:
            names = list(OBJECTIVE_SETS[spec])
        else:
            names = [part.strip() for part in spec.split(",") if part.strip()]
    else:
        names = list(spec)

    if not names:
        return None

    unknown = [n for n in names if n not in OBJECTIVE_REGISTRY]

    if unknown:
        raise KeyError(
            f"Unknown objective(s) {unknown}. "
            f"Available: {sorted(OBJECTIVE_REGISTRY)}; sets: {sorted(OBJECTIVE_SETS)}"
        )

    return ObjectiveVector([OBJECTIVE_REGISTRY[n]() for n in names])
