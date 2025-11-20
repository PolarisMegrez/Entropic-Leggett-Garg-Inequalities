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
from scipy.sparse import issparse, hstack as sparse_hstack

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
        is the number of outcomes at measurement step ``i``.

    Notes
    -----
    - Uses vectorized batch evolution for performance.
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
    outcomes_per_step = [len(obs) for (_, obs) in steps]

    # Initialize batch: single branch
    # current_state: GeneralizedBlochState with shape (d^2, N_branches)
    # current_probs: (N_branches,)
    # current_paths: list of tuples (length N_branches)
    
    # Ensure init_state is a copy and 1D-ish (d^2,) or (d^2, 1)
    current_state = init_state.copy()
    if current_state.data.ndim == 1:
        current_state.data = current_state.data[:, np.newaxis]
    
    current_probs = np.array([1.0], dtype=float)
    current_paths = [()]

    last_time = t0
    
    for step_idx, (t, obs_list) in enumerate(steps):
        dt = float(t - last_time)
        last_time = t
        
        # 1. Evolve all branches
        if dt > 0:
            current_state.evolve(L, dt)
        
        # 2. Apply measurements -> returns list of (probs_vec, state_obj)
        # Each element i corresponds to outcome i
        results = apply_measurement(obs_list, current_state, norm_mode=norm_mode)
        
        next_state_cols = []
        next_probs = []
        next_paths = []
        
        # 3. Expand branches
        for outcome_idx, (p_vec, s_obj) in enumerate(results):
            # p_vec: (N_branches,)
            # s_obj: (d^2, N_branches)
            
            # Calculate new path probabilities
            new_p = current_probs * p_vec
            
            # Prune zero probability branches
            mask = new_p > 0
            if not np.any(mask):
                continue
                
            # Append surviving columns
            if isinstance(s_obj.data, np.matrix):
                # Convert matrix to array to avoid indexing issues
                s_obj.data = np.asarray(s_obj.data)

            next_state_cols.append(s_obj.data[:, mask])
            next_probs.append(new_p[mask])
            
            # Update paths
            # We need to map back to which parent path generated this
            parent_indices = np.flatnonzero(mask)
            for pid in parent_indices:
                next_paths.append(current_paths[pid] + (outcome_idx,))
        
        if not next_state_cols:
            # All branches died
            shape = tuple(outcomes_per_step[: step_idx + 1])
            return np.zeros(shape, dtype=float)
            
        # 4. Reassemble batch
        # Optimization: Use list comprehension and hstack which is generally efficient.
        # Memory warning: For very deep trees or high branching, this can grow large.
        # Pruning (mask) above is essential.
        
        # Check if we are dealing with sparse matrices
        if issparse(next_state_cols[0]):
            current_state = GeneralizedBlochState(sparse_hstack(next_state_cols))
        else:
            current_state = GeneralizedBlochState(np.hstack(next_state_cols))
            
        current_probs = np.concatenate(next_probs)
        current_paths = next_paths

    # Assemble joint probability ndarray
    shape = tuple(outcomes_per_step)
    joint = np.zeros(shape, dtype=float)
    for path, prob in zip(current_paths, current_probs):
        joint[path] += prob

    # Numerical cleanup
    total = float(np.sum(joint))
    if norm_mode in ("normalized", "renormalized") and total > 0:
        joint = joint / total
    return joint