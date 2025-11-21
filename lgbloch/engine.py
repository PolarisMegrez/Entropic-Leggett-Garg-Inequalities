from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from bloch4spin.basis import bloch_init, _idx_from_kq, GeneralizedBlochVector
from bloch4spin.evolution import GeneralizedBlochEvolutionMatrix, GeneralizedBlochState
from bloch4spin.observable import GeneralizedBlochObservable
from bloch4spin.engine import run
from scipy.sparse import dok_matrix

__all__ = [
    "run_case",
    "JointProbabilitySet",
    "distributions_from_times",
    "distributions_from_deltas",
]

def spin_ops(d: int):
    """Construct spin operators Jx, Jy, Jz, Jp, Jm for dimension d.

    Parameters
    ----------
    d : int
        Hilbert space dimension (d = 2j + 1).

    Returns
    -------
    tuple
        (Jx, Jy, Jz, Jp, Jm) as GeneralizedBlochVector objects.
    """
    bloch_init(d)
    j = (d - 1) / 2.0
    c = np.sqrt((j*(j+1)*(2*j+1))/3)
    
    # Use dok_matrix for efficient single-element assignment
    # Jz
    Jz_data = dok_matrix((d*d, 1), dtype=complex)
    Jz_data[_idx_from_kq(1, 0), 0] = c
    Jz = GeneralizedBlochVector(Jz_data.tocsc())
    
    # Jp
    Jp_data = dok_matrix((d*d, 1), dtype=complex)
    Jp_data[_idx_from_kq(1, 1), 0] = -np.sqrt(2) * c
    Jp = GeneralizedBlochVector(Jp_data.tocsc())
    
    # Jm
    Jm_data = dok_matrix((d*d, 1), dtype=complex)
    Jm_data[_idx_from_kq(1, -1), 0] = np.sqrt(2) * c
    Jm = GeneralizedBlochVector(Jm_data.tocsc())
    
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)
    return Jx, Jy, Jz, Jp, Jm

def projectors_Jz(d: int, indices: Tuple[int, ...] | None = None):
    """Construct projectors onto Jz eigenstates for dimension d.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    indices : tuple[int, ...], optional
        Indices of the eigenstates to project onto. If None, returns all d projectors.

    Returns
    -------
    list
        List of projectors (as numpy arrays), ordered by eigenvalue.
    """
    Ps = []
    rng = indices if indices is not None else range(d)
    for i in rng:
        P = np.zeros((d, d), dtype=complex)
        P[i, i] = 1.0
        Ps.append(P)
    return Ps

# Cache for (L, r0, obs_list) per dimension d to avoid recomputation
_ENGINE_CACHE: dict[int, tuple] = {}

def _get_engine_env(d: int):
    if d not in _ENGINE_CACHE:
        bloch_init(d)
        _, Jy, Jz, _, _ = spin_ops(d)
        L_mat = GeneralizedBlochEvolutionMatrix.from_Hamiltonian(Jy)
        rho0 = np.eye(d, dtype=complex) / d
        r0 = GeneralizedBlochState.from_matrix(rho0)
        obs_list = [GeneralizedBlochObservable.from_projector(P) for P in projectors_Jz(d)]
        _ENGINE_CACHE[d] = (L_mat, r0 + 0.0 * Jz, obs_list)
    return _ENGINE_CACHE[d]

# NOTE: No caching of JointProbabilitySet to respect user's request.

def run_case(
    d: int,
    times: list[float],
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: list[GeneralizedBlochObservable],
):
    """Return joint probability tensor for given ``d`` and measurement ``times``.

    Parameters
    - d: Hilbert-space dimension (d = 2j + 1)
    - times: Measurement times (monotonic list of floats)
    - L_mat: Evolution matrix (Liouvillian) acting on Bloch vectors
    - r0: Initial Bloch-state
    - obs_list: List of observables (projectors) to measure at each time

    Returns
    - joint_prob: ndarray shaped (d, d, ..., d) with one axis per measurement time
    """
    # Ensure Bloch basis dimension matches requested d before running
    bloch_init(d)
    schedule = [(float(t), obs_list) for t in times]
    return run(L_mat, r0, schedule, norm_mode="renormalized")

# ---- Probability distributions helpers ----

def _accumulate_times(deltas: List[float]) -> List[float]:
    """Given time gaps [t12, t23, ...], return absolute times [t1=0, t2, t3, ...]."""
    t = 0.0
    times = [t]
    for dt in deltas:
        t += float(dt)
        times.append(t)
    return times

def _powerset_indices(n: int) -> List[Tuple[int, ...]]:
    """All non-empty subsets of {1..n} as sorted tuples, ordered by order then lexicographically."""
    from itertools import combinations
    out: List[Tuple[int, ...]] = []
    for k in range(1, n + 1):
        out.extend(combinations(range(1, n + 1), k))
    return out

# ---- Probability distributions for all time subsets ----

@dataclass(slots=True)
class JointProbabilitySet:
    """Container for joint probability tensors over all non-empty time subsets.

    Attributes
    ----------
    distributions : Dict[Tuple[int, ...], np.ndarray]
        Mapping from subset index tuple S (1-based) to joint probability tensor
        with shape (outcome_num,)*|S|.
    outcome_num : int
        Number of distinct measurement outcomes (equals Hilbert dimension `d`).
    time_point_num : int
        Total number of time points in the original input list.

    Notes
    -----
    - Invasive semantics: each subset S is computed independently (no marginalization).
    - Use `validate(expected_outcomes, expected_time_points)` before passing to LGI
      boolean test functions to ensure compatibility.
    """
    distributions: Dict[Tuple[int, ...], np.ndarray]
    outcome_num: int
    time_point_num: int

    def validate(self, expected_time_points: int, expected_outcomes: int | None = None) -> None:
        if expected_outcomes is not None and self.outcome_num != expected_outcomes:
            raise ValueError(
                f"JointProbabilitySet outcome_num={self.outcome_num} != expected {expected_outcomes}."
            )
        if self.time_point_num != expected_time_points:
            raise ValueError(
                f"JointProbabilitySet time_point_num={self.time_point_num} != expected {expected_time_points}."
            )
        # Structural sanity: ensure required subset keys exist
        required_subsets = _powerset_indices(expected_time_points)
        for S in required_subsets:
            if S not in self.distributions:
                raise ValueError(f"Missing subset {S} in distributions.")

def distributions_from_times(
    times: List[float],
    d: int,
    *,
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: List[GeneralizedBlochObservable],
) -> JointProbabilitySet:
    """Compute joint probability tensors for all non-empty time subsets.

    Requires explicit ``L_mat``, ``r0``, and ``obs_list`` (no internal cache).
    """
    M = len(times)
    if M == 0:
        return JointProbabilitySet({}, d, 0)
    # Compute invasively for each subset (no marginalization)
    dist: Dict[Tuple[int, ...], np.ndarray] = {}
    for S in _powerset_indices(M):
        sub_times = [times[i - 1] for i in S]
        jp = run_case(d, sub_times, L_mat, r0, obs_list)
        dist[S] = jp
    return JointProbabilitySet(dist, len(obs_list), M)

def distributions_from_deltas(
    deltas: List[float],
    d: int,
    *,
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: List[GeneralizedBlochObservable],
) -> JointProbabilitySet:
    """As ``distributions_from_times`` but accepts time gaps ``[t12, t23, ...]``.

    Requires explicit ``L_mat``, ``r0``, and ``obs_list`` (no internal cache).
    """
    times = _accumulate_times(deltas)
    return distributions_from_times(times, d, L_mat=L_mat, r0=r0, obs_list=obs_list)