r"""lgbloch.engine: Spin System Operators and Probability Computation
================================================================
Provides utilities for constructing spin operators, measurement projectors, and computing joint probability distributions for Leggett-Garg inequality testing. This module builds on bloch4spin's generalized Bloch representation to provide spin-specific functionality for quantum macrorealism studies.

Public API
----------
- ``spin_ops``: Construct spin operators Jx, Jy, Jz, Jp, Jm.
- ``projectors_Jz``: Construct projectors onto Jz eigenstates.
- ``run_case``: Compute joint probability tensor for given measurement times.
- ``distributions_from_times``: Compute joint probabilities for all non-empty time subsets.
- ``distributions_from_deltas``: Convenience wrapper using time gaps between measurements.
- ``JointProbabilitySet``: Container class for joint probability distributions.

Notes
-----
- Caching: Uses internal cache to store environment for each dimension.
- Computation: Computes probabilities invasively for each subset (no marginalization).
- Time input: Supports both absolute times and relative time gaps.

"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import dok_matrix

from bloch4spin.basis import GeneralizedBlochVector, _idx_from_kq, bloch_init
from bloch4spin.engine import run
from bloch4spin.evolution import GeneralizedBlochEvolutionMatrix, GeneralizedBlochState
from bloch4spin.observable import GeneralizedBlochObservable

__all__ = [
    "JointProbabilitySet",
    "distributions_from_deltas",
    "distributions_from_times",
    "run_case",
    "spin_ops",
    "projectors_Jz",
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
    c = np.sqrt((j * (j + 1) * (2 * j + 1)) / 3)

    Jz_data = dok_matrix((d * d, 1), dtype=complex)
    Jz_data[_idx_from_kq(1, 0), 0] = c
    Jz = GeneralizedBlochVector(Jz_data.tocsc())
    Jp_data = dok_matrix((d * d, 1), dtype=complex)
    Jp_data[_idx_from_kq(1, 1), 0] = -np.sqrt(2) * c
    Jp = GeneralizedBlochVector(Jp_data.tocsc())
    Jm_data = dok_matrix((d * d, 1), dtype=complex)
    Jm_data[_idx_from_kq(1, -1), 0] = np.sqrt(2) * c
    Jm = GeneralizedBlochVector(Jm_data.tocsc())

    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)
    return Jx, Jy, Jz, Jp, Jm


def projectors_Jz(d: int, indices: tuple[int, ...] | None = None):
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


_ENGINE_CACHE: dict[int, tuple] = {}


def _get_engine_env(d: int):
    """Get cached environment for dimension d.

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Returns
    -------
    tuple
        (L_mat, r0, obs_list) tuple cached for dimension d.

    """
    if d not in _ENGINE_CACHE:
        bloch_init(d)
        _, Jy, Jz, _, _ = spin_ops(d)
        L_mat = GeneralizedBlochEvolutionMatrix.from_Hamiltonian(Jy)
        rho0 = np.eye(d, dtype=complex) / d
        r0 = GeneralizedBlochState.from_matrix(rho0)
        obs_list = [
            GeneralizedBlochObservable.from_projector(P) for P in projectors_Jz(d)
        ]
        _ENGINE_CACHE[d] = (L_mat, r0 + 0.0 * Jz, obs_list)
    return _ENGINE_CACHE[d]


def run_case(
    d: int,
    times: list[float],
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: list[GeneralizedBlochObservable],
):
    """Return joint probability tensor for given ``d`` and measurement ``times``.

    Parameters
    ----------
    d : int
        Hilbert-space dimension (``d = 2j + 1``).
    times : list of float
        Measurement times (monotonic list of floats).
    L_mat : GeneralizedBlochEvolutionMatrix
        Evolution matrix (Liouvillian) acting on Bloch vectors.
    r0 : GeneralizedBlochState
        Initial Bloch-state.
    obs_list : list of GeneralizedBlochObservable
        List of observables (projectors) to measure at each time.

    Returns
    -------
    numpy.ndarray
        Joint probability tensor shaped ``(d, d, ..., d)`` with one axis per
        measurement time.

    """
    bloch_init(d)
    schedule = [(float(t), obs_list) for t in times]
    return run(L_mat, r0, schedule, norm_mode="renormalized")


def _accumulate_times(deltas: list[float]) -> list[float]:
    """Convert time gaps to absolute times.

    Given time gaps ``[t12, t23, ...]``, return absolute times
    ``[t1=0, t2, t3, ...]``.

    Parameters
    ----------
    deltas : list of float
        Time gaps between consecutive measurements.

    Returns
    -------
    list of float
        Absolute times starting from 0.

    """
    t = 0.0
    times = [t]
    for dt in deltas:
        t += float(dt)
        times.append(t)
    return times


def _powerset_indices(n: int) -> list[tuple[int, ...]]:
    """All non-empty subsets of {1..n} as sorted tuples.

    Returns subsets ordered by size (k) then lexicographically.

    Parameters
    ----------
    n : int
        Size of the set (number of time points).

    Returns
    -------
    list of tuple
        List of tuples representing all non-empty subsets.

    """
    from itertools import combinations

    out: list[tuple[int, ...]] = []
    for k in range(1, n + 1):
        out.extend(combinations(range(1, n + 1), k))
    return out


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

    distributions: dict[tuple[int, ...], np.ndarray]
    outcome_num: int
    time_point_num: int

    def validate(
        self, expected_time_points: int, expected_outcomes: int | None = None
    ) -> None:
        """Validate the JointProbabilitySet structure.

        Parameters
        ----------
        expected_time_points : int
            Expected number of time points.
        expected_outcomes : int, optional
            Expected number of measurement outcomes.

        Raises
        ------
        ValueError
            If validation fails.

        """
        if expected_outcomes is not None and self.outcome_num != expected_outcomes:
            raise ValueError(
                f"JointProbabilitySet outcome_num={self.outcome_num} != "
                f"expected {expected_outcomes}."
            )
        if self.time_point_num != expected_time_points:
            raise ValueError(
                f"JointProbabilitySet time_point_num={self.time_point_num} != "
                f"expected {expected_time_points}."
            )
        # Structural sanity: ensure required subset keys exist
        required_subsets = _powerset_indices(expected_time_points)
        for S in required_subsets:
            if S not in self.distributions:
                raise ValueError(f"Missing subset {S} in distributions.")


def distributions_from_times(
    times: list[float],
    d: int,
    *,
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: list[GeneralizedBlochObservable],
) -> JointProbabilitySet:
    """Compute joint probability tensors for all non-empty time subsets.

    Parameters
    ----------
    times : list of float
        Absolute measurement times.
    d : int
        Hilbert space dimension.
    L_mat : GeneralizedBlochEvolutionMatrix
        Evolution matrix.
    r0 : GeneralizedBlochState
        Initial state.
    obs_list : list of GeneralizedBlochObservable
        Measurement observables.

    Returns
    -------
    JointProbabilitySet
        Container with all joint probability distributions.

    Notes
    -----
    Requires explicit ``L_mat``, ``r0``, and ``obs_list`` (no internal cache).
    Computes invasively for each subset (no marginalization).

    """
    M = len(times)
    if M == 0:
        return JointProbabilitySet({}, d, 0)
    # Compute invasively for each subset (no marginalization)
    dist: dict[tuple[int, ...], np.ndarray] = {}
    for S in _powerset_indices(M):
        sub_times = [times[i - 1] for i in S]
        jp = run_case(d, sub_times, L_mat, r0, obs_list)
        dist[S] = jp
    return JointProbabilitySet(dist, len(obs_list), M)


def distributions_from_deltas(
    deltas: list[float],
    d: int,
    *,
    L_mat: GeneralizedBlochEvolutionMatrix,
    r0: GeneralizedBlochState,
    obs_list: list[GeneralizedBlochObservable],
) -> JointProbabilitySet:
    """As ``distributions_from_times`` but accepts time gaps.

    Parameters
    ----------
    deltas : list of float
        Time gaps ``[t12, t23, ...]`` between consecutive measurements.
    d : int
        Hilbert space dimension.
    L_mat : GeneralizedBlochEvolutionMatrix
        Evolution matrix.
    r0 : GeneralizedBlochState
        Initial state.
    obs_list : list of GeneralizedBlochObservable
        Measurement observables.

    Returns
    -------
    JointProbabilitySet
        Container with all joint probability distributions.

    Notes
    -----
    This is a convenience wrapper around ``distributions_from_times`` that
    converts time gaps to absolute times internally.
    Requires explicit ``L_mat``, ``r0``, and ``obs_list`` (no internal cache).

    """
    times = _accumulate_times(deltas)
    return distributions_from_times(times, d, L_mat=L_mat, r0=r0, obs_list=obs_list)
