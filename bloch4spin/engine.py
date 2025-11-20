"""
bloch4spin: Evolution–Measurement Simulation Engine
---------------------------------------------------
Provides a simple `run` function that evolves an initial state under a fixed
Liouvillian and performs measurements at specified times, returning the joint
probability distribution over outcomes as an ndarray.

Public API
----------
- `run`: Simulate a sequence of measurements with intervening unitary evolution.

Notes
-----
- Joint probabilities. Let measurement times be ``t_1 < \dots < t_K`` and let
    each step use outcome superoperators ``\{L^{(r)}_i\}_i`` (one per outcome at
    step ``r``). With evolution ``E(\Delta t) = \exp(L\,\Delta t)``, the
    (unnormalized) post-state for a particular outcome path
    ``\mathbf{i}=(i_1,\dots,i_K)`` is

    .. math::
         r_{\mathbf{i}} = L^{(K)}_{i_K} E(t_K{-}t_{K-1}) \cdots
                                            L^{(2)}_{i_2} E(t_2{-}t_1) L^{(1)}_{i_1} E(t_1{-}t_0)\, r_0.

    Using the COBITO convention ``T_0^{(0)}=I/\sqrt{d}``, the probability of this
    path is ``p(\mathbf{i}) = \sqrt{d}\, [r_{\mathbf{i}}]_0`` (the ``(0,0)``
    coordinate), and the returned ndarray stores these joint probabilities ordered
    by outcome indices.

- Scalability. Designed for small to moderate branching. For large schedules or
    high-dimensional systems, prefer sparse path dictionaries and pruning.
"""

from typing import List, Tuple
import warnings
import numpy as np

from .evolution import GeneralizedBlochEvolutionMatrix, GeneralizedBlochState
from .observable import GeneralizedBlochObservable, apply_measurement

__all__ = ["run"]

def _evolve_state(state: GeneralizedBlochState,
                  L: GeneralizedBlochEvolutionMatrix,
                  dt: float) -> GeneralizedBlochState:
    if dt == 0.0:
        return state.copy()
    s = state.copy()
    s.evolve(L, dt)
    return s

def run(L: GeneralizedBlochEvolutionMatrix,
        init_state: GeneralizedBlochState,
        schedule: List[Tuple[float, List[GeneralizedBlochObservable]]],
        *,
        norm_mode: str = "normalized",
        t0: float = 0.0) -> np.ndarray:
    """Run an evolution–measurement protocol and return joint probabilities.

    Parameters
    ----------
    L : GeneralizedBlochEvolutionMatrix
        Fixed Liouvillian driving the unitary evolution between measurements.
    init_state : GeneralizedBlochState
        Initial state at time ``t0``.
    schedule : list of (time, list of GeneralizedBlochObservable)
        Measurement steps; each entry holds the measurement time and the list
        of outcome superoperators for that step.
    norm_mode : {"normalized", "renormalized", "unnormalized"}, optional
        Passed to :func:`apply_measurement` at each step.
    t0 : float, optional
        Initial time (default 0.0). Steps earlier than ``t0`` are ignored with
        a warning.

    Returns
    -------
    numpy.ndarray
        Joint probability ndarray with shape ``(M1, M2, ..., Mk)`` where ``Mi``
        is the number of outcomes at measurement step ``i`` (after filtering and
        sorting by time, and ignoring steps earlier than ``t0``). If no steps
        remain, returns ``np.array([1.0])``.

        Notes
        -----
        - Branching is handled explicitly by duplicating state objects per outcome.
        - Joint probability for a path ``(i_1,\dots,i_K)`` equals
            ``\sqrt{d}`` times the ``r_{00}`` coordinate of the unnormalized state
            after composing interleaved evolutions and measurement superoperators.
        - For large branching factors, consider a sparse dictionary over paths for
            efficiency.
    """
    if not schedule:
        return np.array([1.0], dtype=float)

    # Filter and sort schedule by time
    steps = [(t, obs) for (t, obs) in schedule if t >= t0]
    dropped = len(schedule) - len(steps)
    if dropped > 0:
        warnings.warn(f"Ignored {dropped} measurement steps with time < t0.")
    if not steps:
        return np.array([1.0], dtype=float)
    steps.sort(key=lambda x: x[0])

    # Prepare shapes and timeline
    times = [t for (t, _) in steps]
    outcomes_per_step = [len(obs) for (_, obs) in steps]

    # Initialize branch list with initial state
    branches: List[Tuple[tuple, float, GeneralizedBlochState]] = [
        (tuple(), 1.0, init_state.copy())
    ]

    # Iterate steps
    last_time = t0
    for step_idx, (t, obs_list) in enumerate(steps):
        dt = float(t - last_time)
        last_time = t
        new_branches: List[Tuple[tuple, float, GeneralizedBlochState]] = []
        for path, prob, st in branches:
            # Evolve to current time
            st_t = _evolve_state(st, L, dt)
            # Measure -> returns list of (p_i, post_state_i)
            results = apply_measurement(obs_list, st_t, norm_mode=norm_mode)
            for i, (p_i, st_i) in enumerate(results):
                if p_i <= 0.0:
                    continue  # prune zero-probability branches
                new_path = path + (i,)
                new_prob = prob * p_i
                new_branches.append((new_path, new_prob, st_i))
        branches = new_branches
        if not branches:
            # All branches pruned
            shape = tuple(outcomes_per_step[: step_idx + 1])
            return np.zeros(shape, dtype=float)

    # Assemble joint probability ndarray
    shape = tuple(outcomes_per_step)
    joint = np.zeros(shape, dtype=float)
    for path, prob, _ in branches:
        joint[path] = joint[path] + prob

    # Numerical cleanup: ensure total sums to ~1 for normalized/renormalized
    total = float(np.sum(joint))
    if norm_mode in ("normalized", "renormalized") and total > 0:
        joint = joint / total
    return joint