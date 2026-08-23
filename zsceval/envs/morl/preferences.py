"""Adaptive Preference Optimization via Mirror Descent.

Scalarizing a reward vector ``r in R^K`` into ``w . r`` needs a preference
vector ``w``. Fixed weights presume the relative importance of the objectives
never changes; in Overcooked it plainly does (early in an episode nothing has
been plated yet, late in an episode the pots are the bottleneck).

This module implements the multiplicative-weights / mirror-descent update that
keeps ``w`` on the probability simplex while steering the *realised* objective
proportions ``g`` towards a target ``t``:

    w_i <- w_i exp(eta (t_i - g_i)) / Z

``g`` is the share of the episode's objective mass currently attributable to
objective ``i`` -- exactly what :meth:`ObjectiveVector.proportions` returns.
The sign convention means an objective that is *over*-represented (``g_i > t_i``)
has its weight pushed down and an under-represented one pushed up, so the agent
keeps attention on whatever it is currently neglecting.

Variable step size
------------------
``eta`` is scaled by how imbalanced ``g`` currently is, so a badly skewed episode
is corrected hard and a balanced one is barely touched. ``g`` lies on the
simplex, so its variance is maximal when one component is 1 and the rest are 0:

    Var_max = (K - 1) / K^2
    s       = Var(g) / Var_max                in [0, 1]
    eta     = eta_min + (eta_max - eta_min) s

Both extremes are meaningful: ``s = 0`` (perfectly balanced) gives ``eta_min``,
``s = 1`` (maximally imbalanced) gives ``eta_max``.

Compounding, and the floor
--------------------------
The update is multiplicative and Overcooked calls it on every one of the 400
steps in an episode, so what actually moves ``w`` is the *sum* of the exponents,
roughly ``eta * T * (t - g)``. An ``eta`` that looks small per step is therefore
large per episode, and a persistently over-represented objective is driven to a
vertex of the simplex within a single episode. Two things keep that in check:

* the ``eta`` defaults are scaled so that ``eta_max * T`` is order 1, which
  moves a persistently over-represented weight by roughly a factor of ``e`` over
  one episode rather than annihilating it; scale ``eta`` up if you raise
  ``morl_weight_update_interval``, since that is what sets ``T``;
* ``floor`` mixes a little of the target back in after every update
  (``w <- (1 - K f) w + f``), which is the standard Hedge/EXP3 safeguard. It
  guarantees ``w_i >= floor``, so no objective can be switched off permanently
  and later become unrecoverable when its ``g`` finally drops.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class MirrorDescentPreferences:
    """Preference weights over a :class:`~zsceval.envs.morl.ObjectiveVector`.

    Example:
        >>> prefs = MirrorDescentPreferences(num_objectives=4)
        >>> prefs.weights                       # uniform
        array([0.25, 0.25, 0.25, 0.25])
        >>> prefs.update(np.array([0.7, 0.1, 0.1, 0.1]))   # objective 0 dominates
        array([0.24979036, 0.25006988, 0.25006988, 0.25006988]) # weight 0 pushed down, rest up

    Attributes:
        eta: Step size used by the most recent :meth:`update`, for logging.
    """

    def __init__(
        self,
        num_objectives: int,
        eta_min: float = 1e-4,
        eta_max: float = 5e-3,
        target: Optional[Sequence[float]] = None,
        floor: float = 0.01,
    ):
        """
        Args:
            num_objectives: ``K``, the length of the reward vector.
            eta_min: Step size when ``g`` is perfectly balanced.
            eta_max: Step size when ``g`` sits at a vertex of the simplex.
            target: The ``t`` of the update rule, i.e. the objective mix being
                steered towards. Defaults to uniform ``1/K``. Normalised on the
                way in, so unnormalised ratios are accepted.
            floor: Smallest weight any objective may hold. Mixed back in after
                every update, so ``w`` stays in the interior of the simplex.
                ``0`` disables it. Must be below ``1/K``.

        Raises:
            ValueError: On a non-positive ``K``, a negative or wrongly-shaped
                target, ``eta_min > eta_max``, or an out-of-range floor.
        """
        if num_objectives < 1:
            raise ValueError(f"num_objectives must be >= 1, got {num_objectives}")
        if eta_min > eta_max:
            raise ValueError(f"eta_min ({eta_min}) must not exceed eta_max ({eta_max})")
        if not 0 <= floor < 1.0 / num_objectives:
            raise ValueError(
                f"floor must be in [0, 1/K) = [0, {1.0 / num_objectives}), got {floor}"
            )

        self.num_objectives = int(num_objectives)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.floor = float(floor)
        self.target = self._normalise_target(target)

        # Maximal variance of a point on the K-simplex: some g_i is 1, rest are 0.
        # Degenerate for K == 1, where g is always [1.0] and no update is possible.
        self._var_max = (self.num_objectives - 1) / self.num_objectives**2
        self.eta = self.eta_min

        self._weights = self.target.copy()

    def _normalise_target(self, target: Optional[Sequence[float]]) -> np.ndarray:
        if target is None:
            return np.full(self.num_objectives, 1.0 / self.num_objectives)

        t = np.asarray(target, dtype=np.float64).reshape(-1)
        if t.shape[0] != self.num_objectives:
            raise ValueError(
                f"target has {t.shape[0]} entries, expected {self.num_objectives}"
            )
        if np.any(t < 0):
            raise ValueError(f"target must be non-negative, got {t.tolist()}")
        total = t.sum()
        if total <= 0:
            raise ValueError("target must have positive total mass")
        return t / total

    @property
    def weights(self) -> np.ndarray:
        """Current ``w``. A copy, so callers cannot mutate the internal state."""
        return self._weights.copy()

    def reset(self) -> None:
        """Return ``w`` to the target. Called on every ``env.reset()``."""
        self._weights = self.target.copy()
        self.eta = self.eta_min

    def update(self, proportions: Sequence[float]) -> np.ndarray:
        """Apply one mirror descent step and return the new ``w``.

        Args:
            proportions: The ``g`` of the update rule, normally
                ``ObjectiveVector.proportions()``. Must be non-negative; it is
                renormalised defensively so a caller passing raw cumulative
                scores still gets sensible behaviour.

        Returns:
            The updated ``(K,)`` weight vector, on the simplex.
        """
        g = np.asarray(proportions, dtype=np.float64).reshape(-1)
        if g.shape[0] != self.num_objectives:
            raise ValueError(
                f"proportions has {g.shape[0]} entries, expected {self.num_objectives}"
            )
        if np.any(g < 0):
            raise ValueError(f"proportions must be non-negative, got {g.tolist()}")

        total = g.sum()
        # An all-zero g carries no information (nothing has happened yet), so
        # fall back to the target rather than dividing by zero.
        g = g / total if total > 0 else self.target.copy()

        if self._var_max > 0:
            s = float(np.clip(g.var() / self._var_max, 0.0, 1.0))
            self.eta = self.eta_min + (self.eta_max - self.eta_min) * s
            # exp() of a shifted exponent is identical after renormalisation and
            # cannot overflow, which matters once eta_max is turned up.
            exponent = self.eta * (self.target - g)
            updated = self._weights * np.exp(exponent - exponent.max())
            total = updated.sum()
            if total > 0:
                self._weights = self._apply_floor(updated / total)

        return self.weights

    def _apply_floor(self, weights: np.ndarray) -> np.ndarray:
        """Lift any weight that has fallen below `floor`, staying on the simplex.

        Only the entries that actually hit the floor are pinned; the rest keep
        their relative proportions and share the remaining mass. That way the
        floor binds only when it has to, instead of damping every update the way
        mixing a fixed amount of the target back in would.
        """
        if self.floor <= 0:
            return weights

        pinned = weights < self.floor

        if not pinned.any():  # the common case: nothing to do
            return weights

        while True:
            free = np.where(pinned, 0.0, weights)
            total = free.sum()
            if total <= 0:  # everything pinned: nothing to distribute
                return np.full_like(weights, 1.0 / weights.size)

            free_mass = 1.0 - pinned.sum() * self.floor
            floored = np.where(pinned, self.floor, free / total * free_mass)

            newly_below = (~pinned) & (floored < self.floor)
            if not newly_below.any():
                return floored
            pinned = pinned | newly_below

    def __repr__(self) -> str:
        return (
            f"MirrorDescentPreferences(K={self.num_objectives}, "
            f"eta=[{self.eta_min}, {self.eta_max}], "
            f"w={np.round(self._weights, 4).tolist()})"
        )
