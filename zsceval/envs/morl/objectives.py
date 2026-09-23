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


# The kitchen's jobs as separate objectives (see envs/morl/tasks.py). `plating`
# below lumps fetching a dish with scooping a soup; the task view keeps them
# apart, because a partner who fetches dishes but never plates leaves a
# different job undone from one who does neither. Each counts exactly the event
# the hand-shaped reward pays for, so the `tasks` preset at weights 20,3,3,5
# with team delivery credit and annealing *is* the hand-shaped reward.
def FillPot(scale: float = 1.0) -> EventCountObjective:
    return EventCountObjective(
        name="fill_pot",
        event_keys=("PLACEMENT_IN_POT",),
        description="An ingredient placed into a pot with room for it.",
        scale=scale,
    )


def FetchDish(scale: float = 1.0) -> EventCountObjective:
    return EventCountObjective(
        name="fetch_dish",
        event_keys=("USEFUL_DISH_PICKUP",),
        description="A dish taken from the dispenser while a soup is on its way without one.",
        scale=scale,
    )


def PlateSoup(scale: float = 1.0) -> EventCountObjective:
    return EventCountObjective(
        name="plate_soup",
        event_keys=("SOUP_PICKUP",),
        description="A finished soup scooped out of its pot.",
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


def RecipeQuality(scale: float = 1.0) -> EventCountObjective:
    """Recipe judgement: ingredients potted towards a recipe worth cooking.

    Multi-recipe only. `resolve_interacts` in the new MDP grades every
    `PLACEMENT_IN_POT` against the recipes still reachable from the pot's
    contents -- `optimal_placement` when it keeps the best recipe alive,
    `viable_placement` when the result is still deliverable, and
    `catastrophic_placement` / `useless_placement` when it is not. The old
    single-recipe env cannot grade anything, because there is only one recipe,
    which is why this objective has no counterpart in the `default` preset.

    Only `optimal_placement` is counted. The four grades are raised by four
    independent `if`s rather than an `elif` chain, so a placement that is both
    optimal and viable increments both counters -- summing the two would score
    a good placement twice and make the objective exceed `ingredient_prep`,
    which counts the same placements once each.
    """
    return EventCountObjective(
        name="recipe_quality",
        event_keys=("optimal_placement",),
        description="Pot placements that keep the best reachable recipe alive.",
        scale=scale,
    )


def RecipeValue(scale: float = 1.0) -> EventCountObjective:
    """Recipe ambition: delivering the larger, higher-priced order.

    A size-three order pays more but takes longer to assemble than a size-two,
    so `task_completion` (which counts deliveries regardless of price) and this
    objective genuinely conflict: maximising one costs the other. That trade-off
    is the reason the multi-recipe layouts are worth running -- a scalar shaped
    reward has to pick a point on it when the reward is written, while a weight
    vector can move along it.
    """
    return EventCountObjective(
        name="recipe_value",
        event_keys=("deliver_size_three_order",),
        description="Deliveries of the larger, higher-priced order.",
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


class AnchoredCounterHandoff(CounterHandoff):
    """Coordination quality, credited only once the handoff is *used*.

    :class:`CounterHandoff` credits a handoff the instant the partner collects
    the object, and that is farmable: the put and the pickup are exact inverses
    that advance nothing, so two agents can pass an onion back and forth for the
    whole episode. They do. On random0 the measured ratio of the best score
    reached while delivering *nothing* to the median score reached while
    delivering is 3.69 -- the objective is better farmed than earned -- and six
    stage-1 seeds across two arms found it independently. The other three
    objectives in the `default` set sit at 0.18, 0.03 and 0.00 on the same
    measure, because each is gated behind an irreversible step.

    This version restores that gate without giving up the thing coordination is
    for. A handoff becomes a *pending* claim when the partner collects it, and
    pays out only when the receiver commits the object to progress that cannot
    be undone:

        onion, tomato   ->  PLACEMENT_IN_POT
        dish            ->  SOUP_PICKUP
        soup            ->  delivery

    and the claim is *cancelled* if the receiver instead puts the object back on
    a counter. That is exactly the farming loop, so it now earns nothing, while
    a genuine pass -- hand over an onion, partner pots it -- is credited in full.

    Deferring the credit rather than dropping the objective matters: a purely
    structural rule ("count only irreversible events") would delete coordination
    altogether, and coordination is the one component that measures the thing
    this project is about. The rule to take away is not *avoid* such objectives
    but *anchor* them to an irreversible consequence.

    An agent resolves one INTERACT per step, so placing, collecting and
    anchoring are mutually exclusive for a given agent on a given step, and the
    three branches below cannot race.
    """

    name = "coordination"
    description = "Objects handed to the partner and then actually used."

    #: What the receiver must do with a handed object for the claim to pay out.
    ANCHOR_EVENTS = {
        "onion": ("PLACEMENT_IN_POT",),
        "tomato": ("PLACEMENT_IN_POT",),
        "dish": ("SOUP_PICKUP",),
        "soup": ("delivery",),
    }

    def __init__(self, scale: float = 1.0, credit_both: bool = True):
        super().__init__(scale=scale, credit_both=credit_both)
        self._pending: Dict[int, List[Any]] = {}

    def reset(self, num_players: int) -> None:
        super().reset(num_players)
        self._pending = {i: [] for i in range(num_players)}

    def _object_placed(self, context: ObjectiveContext, agent_idx: int):
        for obj in self.OBJECT_NAMES:
            if context.count(agent_idx, f"put_{obj}_on_X"):
                return obj
        return None

    def _object_taken(self, context: ObjectiveContext, agent_idx: int):
        for obj in self.OBJECT_NAMES:
            if context.count(agent_idx, f"pickup_{obj}_from_X"):
                return obj
        return None

    def __call__(self, context: ObjectiveContext) -> np.ndarray:
        reward = np.zeros(context.num_players, dtype=np.float64)
        if not self._pending:
            self._pending = {i: [] for i in range(context.num_players)}

        for agent_index in range(context.num_players):
            placed = self._object_placed(context, agent_index)
            took = self._object_taken(context, agent_index)

            if placed is not None:
                position = self._interact_position(context.prev_state, agent_index)
                self._counter_owner[position] = agent_index
                # Putting it straight back down is the farming loop, not a use.
                for entry in list(self._pending[agent_index]):
                    if entry[0] == placed:
                        self._pending[agent_index].remove(entry)
                        break
                continue

            if took is not None:
                position = self._interact_position(context.prev_state, agent_index)
                owner_index = self._counter_owner.pop(position, None)
                if owner_index is not None and owner_index != agent_index:
                    self._pending[agent_index].append((took, owner_index))
                continue

            # Neither placed nor collected: did this agent cash a claim in?
            for entry in list(self._pending[agent_index]):
                obj, giver = entry
                if any(
                    context.count(agent_index, event)
                    for event in self.ANCHOR_EVENTS.get(obj, ())
                ):
                    reward[agent_index] += self.scale
                    if self.credit_both:
                        reward[giver] += self.scale
                    self._pending[agent_index].remove(entry)
                    break

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

    def __init__(self, objectives: Sequence[Objective], diminishing_alpha: float = 1.0):
        if not objectives:
            raise ValueError("ObjectiveVector requires at least one objective")
        if not 0.0 < diminishing_alpha <= 1.0:
            raise ValueError(
                f"diminishing_alpha must be in (0, 1], got {diminishing_alpha}"
            )
        names = [o.name for o in objectives]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate objective names: {sorted(duplicates)}")

        self.objectives: List[Objective] = list(objectives)
        # Diminishing returns. `alpha` 1.0 is the plain event count; below 1 the
        # step reward becomes the marginal value of a concave function of the
        # episode-to-date total, `f(c + d) - f(c)` with `f(x) = x**alpha`, so the
        # three-hundredth handoff is worth far less than the first.
        #
        # This exists because raw counts are hackable. On random0 the
        # `coordination` objective counts counter handoffs, and an agent can put
        # an onion down and pick it up forever: bench_morl_div's plating+coord
        # seed reached coordination 317.8 with a sparse return of 0, and two of
        # bench_morl's six uniform-weight seeds did the same. The reward rises
        # the whole time, so the failure is invisible from the training curve.
        #
        # It also repairs the preference vector. `w` multiplies raw counts, and
        # on random0 coordination averages 18.5 against task_completion's 1.85 --
        # so "uniform" w = 0.25 each gives coordination ten times the influence.
        # Concavity compresses that: at alpha 0.5 a 10x count gap becomes 3.2x,
        # which is what makes w mean roughly what it says.
        self.diminishing_alpha = float(diminishing_alpha)
        self._num_players = 0
        self._cumulative = np.zeros((0, len(self.objectives)), dtype=np.float64)
        # Raw counts, kept separately: `cumulative` stays the reported episode
        # total in event units so `ep_obj_*` and the mirror-descent `g` term
        # remain comparable with every run made before this existed.
        self._raw_cumulative = np.zeros((0, len(self.objectives)), dtype=np.float64)

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
        self._raw_cumulative = np.zeros(
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

        if self.diminishing_alpha == 1.0:
            self._raw_cumulative = self._cumulative
            return step_reward

        # Marginal value of the concave f over the episode-to-date raw total.
        # Negative components (an objective may penalise) are passed through
        # untransformed: f is defined for non-negative totals, and clamping the
        # running total at zero keeps a penalty from being silently discounted
        # by however much credit was banked before it.
        alpha = self.diminishing_alpha
        before = self._raw_cumulative
        after = before + step_reward
        marginal = np.sign(after) * np.abs(after) ** alpha - np.sign(
            before
        ) * np.abs(before) ** alpha
        self._raw_cumulative = after
        return marginal

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
    "coordination_anchored": AnchoredCounterHandoff,
    # task_completion priced at the environment's own delivery reward rather
    # than counted. See the `anchored_valued` preset for why.
    "task_completion_valued": lambda: TaskCompletion(scale=20.0),
    "recipe_quality": RecipeQuality,
    "recipe_value": RecipeValue,
    "fill_pot": FillPot,
    "fetch_dish": FetchDish,
    "plate_soup": PlateSoup,
}

# Named presets
OBJECTIVE_SETS: Dict[str, List[str]] = {
    "default": ["task_completion", "ingredient_prep", "plating", "coordination"],
    "task_only": ["task_completion"],
    # Same four objectives as `default`, with coordination credited only once
    # the handed object is used. `default` is kept unchanged so every run made
    # before the farming was found stays reproducible; `anchored` is what new
    # runs should use.
    "anchored": [
        "task_completion",
        "ingredient_prep",
        "plating",
        "coordination_anchored",
    ],
    # `anchored`, with deliveries priced at what the environment pays for them.
    #
    # Counting a delivery as 1 makes w uninterpretable, because the components
    # are not commensurate: on random0 a delivering agent reaches roughly
    # task 7, ingredient_prep 23, plating 14, coordination 60 per episode. So
    # w = (0.25, 0.75, 0, 0) -- nominally a quarter of the preference on
    # finishing the task -- actually pays 1.75 for deliveries against 17.25 for
    # potting, and the agent correctly optimises potting. bench_morl_div's
    # prep-weighted members scored 0 sparse for exactly this reason, and
    # anchoring did not help them because their failure was never farming.
    #
    # scale=20 is the MDP's own delivery_reward, so the component is the sparse
    # reward it already represents rather than an arbitrary constant.
    #
    # Caveat, and the reason `anchored` keeps unit counts: the module's
    # non-negativity and commensurate-magnitude invariants exist for the
    # mirror-descent update, whose `g` is a proportion of realised objective
    # mass. Pricing one component 20x higher pushes `g` toward that vertex and
    # leaves the update less to do. That is a real cost for --morl_adaptive_weights
    # and no cost at all for fixed w, which is what bench_morl_div uses.
    # For layouts where counter handoffs never happen. On unident_s the
    # coordination component is identically 0.00 for every arm, so the four-way
    # set spends a quarter of its preference mass on an objective that cannot
    # fire.
    #
    # For a fixed w that is merely a scale factor -- the three live components
    # stay equally weighted. For --morl_adaptive_weights it is a weight sink:
    # the mirror-descent update is w_i <- w_i exp(eta (t_i - g_i)), and a
    # component whose realised share g is permanently 0 against a target of 0.25
    # has a permanently positive exponent, so its weight grows without bound
    # while the live objectives are starved. That is the 0.286 the adaptive arm
    # was observed parking on coordination.
    "anchored_live3": [
        "task_completion",
        "ingredient_prep",
        "plating",
    ],
    "anchored_valued": [
        "task_completion_valued",
        "ingredient_prep",
        "plating",
        "coordination_anchored",
    ],
    # Multi-recipe (`*_m`) layouts only. `default` plus the two objectives the
    # single-recipe env has no counters for. The events behind these are only
    # ever non-zero in `envs/overcooked_new`, so using this preset on an old
    # layout is not an error but leaves two components pinned at 0 -- which
    # the adaptive weight update will then happily pour weight into.
    # coordination_anchored, not coordination: the plain one credits a handoff
    # the moment the partner collects the object, which is farmable, and six
    # seeds on random0 found the loop. A multi-recipe run on the plain set would
    # simply reproduce that.
    "recipe": [
        "task_completion",
        "ingredient_prep",
        "plating",
        "coordination_anchored",
        "recipe_quality",
        "recipe_value",
    ],
    # The fill-in suite's task view: deliveries plus the three jobs that lead to
    # one. Old (onion-only) layouts; see FillPot above.
    "tasks": ["task_completion", "fill_pot", "fetch_dish", "plate_soup"],
}


def make_objective_vector(
    spec: Optional[Union[str, Sequence[str], ObjectiveVector]] = None,
    diminishing_alpha: float = 1.0,
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

    return ObjectiveVector(
        [OBJECTIVE_REGISTRY[n]() for n in names], diminishing_alpha=diminishing_alpha
    )
