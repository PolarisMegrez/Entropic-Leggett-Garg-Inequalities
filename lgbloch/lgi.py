"""LGI boolean test functions.

This module exposes boolean-valued Leggett–Garg inequality (LGI) tests that
operate on a `JointProbabilitySet` produced by the engine. Each test embeds
the expected number of outcomes and time points and validates the input before
performing its calculation. Implementations are NumPy-only for clarity and
maintainability; no optional JIT/parallel paths are used.
"""

from typing import Tuple
from itertools import combinations
import numpy as np
from .engine import JointProbabilitySet

__all__ = [
    "entropic_LGI",
    "entropic_LGI_three_point",
    "entropic_LGI_four_point",
    "standard_LGI_dichotomic_three_point",
    "standard_LGI_dichotomic_four_point",
    "wigner_LGI_dichotomic_three_point",
    "wigner_LGI_dichotomic_four_point",
]

def _entropy_flat(arr: np.ndarray) -> float:
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = np.asarray(arr, dtype=float) / total
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))

def _shannon_entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(_entropy_flat(p.ravel()))

def _jz_values_for_d(d: int) -> np.ndarray:
    j = (d - 1) / 2.0
    return j - np.arange(d, dtype=float)

def entropic_LGI(n: int, jps: JointProbabilitySet) -> float:
    r"""General entropic LGI test for n time points using Shannon cone basis elements.

    Iterates over all subsets of measurement points and computes elementary inequalities:
    1. Conditional Entropy: H(S) - H(S \ {i}) >= 0
    2. Conditional Mutual Information: H(S \ {i}) + H(S \ {j}) - H(S) - H(S \ {i, j}) >= 0

    Parameters
    ----------
    n : int
        Number of time points.
    jps : JointProbabilitySet
        Joint probability tensors.

    Returns
    -------
    float
        Minimum value of the inequalities.
    """
    jps.validate(expected_time_points=n)
    dist = jps.distributions

    def H(subset):
        if not subset:
            return 0.0
        return _shannon_entropy_bits(dist[tuple(sorted(subset))])

    values = []
    # Iterate over all subset sizes k from 2 to n
    for k in range(2, n + 1):
        for subset in combinations(range(1, n + 1), k):
            subset_set = set(subset)

            # Type 1: D_i = H(S) - H(S \ {i})
            for i in subset_set:
                val = H(subset_set) - H(subset_set - {i})
                values.append(val)

            # Type 2: D_{i,j} (only for k >= 3)
            if k >= 3:
                for i, j in combinations(subset_set, 2):
                    val = H(subset_set - {i}) + H(subset_set - {j}) - H(subset_set) - H(subset_set - {i, j})
                    values.append(val)

    return float(np.min(values)) if values else 0.0


def standard_LGI_dichotomic_three_point(jps: JointProbabilitySet) -> float:
    """Three-point standard LGI positivity test (dichotomic outcomes).

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of three time points
        with two measurement outcomes (dichotomic).
    atol : float, optional
        Numerical tolerance for non-negativity of reconstructed probabilities.

    Returns
    -------
    float
        The minimum reconstructed probability among the 8 assignments.

    Notes
    -----
    - Interprets the two outcomes directly as ``s ∈ {+1, -1}`` per time.
    - Uses the 3-time expansion with ``Q_i`` (means), ``C_{ij}`` (pair correlators),
      and ``D_{123}`` (triple correlator); probabilities are reconstructed and
      checked for non-negativity without post scaling factors.

    References
    ----------
    - J. J. Halliwell, "Leggett-Garg tests of macrorealism: Checks for noninvasiveness and generalizations to higher-order correlators," Phys. Rev. A 99, 022119 (2019).
    - J. J. Halliwell, "Necessary and sufficient conditions for macrorealism using two- and three-time Leggett-Garg inequalities," J. Phys.: Conf. Ser. 1275, 012008 (2019).
    """
    # Enforce dichotomic two outcomes and 3 time points
    jps.validate(expected_time_points=3, expected_outcomes=2)
    dist = jps.distributions
    s = np.array([1.0, -1.0])
    Q1 = float(np.sum(dist[(1,)] * s))
    Q2 = float(np.sum(dist[(2,)] * s))
    Q3 = float(np.sum(dist[(3,)] * s))
    C12 = float(np.sum(dist[(1, 2)] * (s[:, None] * s[None, :])))
    C23 = float(np.sum(dist[(2, 3)] * (s[:, None] * s[None, :])))
    C13 = float(np.sum(dist[(1, 3)] * (s[:, None] * s[None, :])))
    D123 = float(np.sum(dist[(1, 2, 3)] * (s[:, None, None] * s[None, :, None] * s[None, None, :])))
    min_val = 1e300
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            for s3 in (-1.0, 1.0):
                p = (1.0 / 8.0) * (
                    1.0
                    + s1 * Q1 + s2 * Q2 + s3 * Q3
                    + s1 * s2 * C12 + s2 * s3 * C23 + s1 * s3 * C13
                    + s1 * s2 * s3 * D123
                )
                if p < min_val:
                    min_val = p
    return float(min_val)

def standard_LGI_dichotomic_four_point(jps: JointProbabilitySet) -> float:
    """Four-point standard LGI positivity test (dichotomic outcomes).

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of four time points
        with two measurement outcomes (dichotomic).
    atol : float, optional
        Numerical tolerance for non-negativity of reconstructed probabilities.

    Returns
    -------
    float
        The minimum reconstructed probability among the 16 assignments.

    Notes
    -----
    - Interprets the two outcomes directly as ``s ∈ {+1, -1}`` per time.
    - Reconstructs ``p(s_1, s_2, s_3, s_4)`` via the dichotomic expansion
      ``(1/16) * (1 + Σ_i s_i Q_i + Σ_{i<j} s_i s_j C_{ij} + Σ_{i<j<k} s_i s_j s_k D_{ijk} + s_1 s_2 s_3 s_4 E_{1234})``
      and checks non-negativity for all 16 assignments with ``s_i ∈ {±1}``.

    References
    ----------
    - J. J. Halliwell, "Leggett-Garg tests of macrorealism: Checks for noninvasiveness and generalizations to higher-order correlators," Phys. Rev. A 99, 022119 (2019).
    - J. J. Halliwell, "Necessary and sufficient conditions for macrorealism using two- and three-time Leggett-Garg inequalities," J. Phys.: Conf. Ser. 1275, 012008 (2019).
    """
    jps.validate(expected_time_points=4, expected_outcomes=2)
    dist = jps.distributions
    s = np.array([1.0, -1.0])
    Q1 = float(np.sum(dist[(1,)] * s))
    Q2 = float(np.sum(dist[(2,)] * s))
    Q3 = float(np.sum(dist[(3,)] * s))
    Q4 = float(np.sum(dist[(4,)] * s))
    C12 = float(np.sum(dist[(1, 2)] * (s[:, None] * s[None, :])))
    C13 = float(np.sum(dist[(1, 3)] * (s[:, None] * s[None, :])))
    C14 = float(np.sum(dist[(1, 4)] * (s[:, None] * s[None, :])))
    C23 = float(np.sum(dist[(2, 3)] * (s[:, None] * s[None, :])))
    C24 = float(np.sum(dist[(2, 4)] * (s[:, None] * s[None, :])))
    C34 = float(np.sum(dist[(3, 4)] * (s[:, None] * s[None, :])))
    D123 = float(np.sum(dist[(1, 2, 3)] * (s[:, None, None] * s[None, :, None] * s[None, None, :])))
    D124 = float(np.sum(dist[(1, 2, 4)] * (s[:, None, None] * s[None, :, None] * s[None, None, :])))
    D134 = float(np.sum(dist[(1, 3, 4)] * (s[:, None, None] * s[None, :, None] * s[None, None, :])))
    D234 = float(np.sum(dist[(2, 3, 4)] * (s[:, None, None] * s[None, :, None] * s[None, None, :])))
    E1234 = float(np.sum(dist[(1, 2, 3, 4)] * (s[:, None, None, None] * s[None, :, None, None] * s[None, None, :, None] * s[None, None, None, :])))
    min_val = 1e300
    for s1v in (-1.0, 1.0):
        for s2v in (-1.0, 1.0):
            for s3v in (-1.0, 1.0):
                for s4v in (-1.0, 1.0):
                    p = (1.0 / 16.0) * (
                        1.0
                        + s1v * Q1 + s2v * Q2 + s3v * Q3 + s4v * Q4
                        + s1v * s2v * C12 + s1v * s3v * C13 + s1v * s4v * C14
                        + s2v * s3v * C23 + s2v * s4v * C24 + s3v * s4v * C34
                        + s1v * s2v * s3v * D123 + s1v * s2v * s4v * D124
                        + s1v * s3v * s4v * D134 + s2v * s3v * s4v * D234
                        + s1v * s2v * s3v * s4v * E1234
                    )
                    if p < min_val:
                        min_val = p
    return float(min_val)

def wigner_LGI_dichotomic_three_point(jps: JointProbabilitySet) -> float:
    """Three-point Wigner-form LGI (dichotomic outcomes).

    Evaluates inequalities of the form::

        P(A x, B y) - P(A x, C z) - P(B y, C z') <= 0

    where A, B, C are time points (permutations of {1,2,3}); outcomes are
    dichotomic with index mapping 0 → '+', 1 → '-'. To remove the global sign
    redundancy, fix x_idx=0 (i.e., '+'), while y_idx and z_idx range over {0,1},
    with z'_idx = 1 - z_idx. This yields 6 * 4 = 24 inequalities.

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of three time points
        with two dichotomic outcomes.

    Returns
    -------
    float
        The minimum value among the 24 Wigner inequalities.

    Notes
    -----
    - Outcome index convention: 0 → '+', 1 → '-' (descending m-order).
    - If expansion to full 48 inequalities is desired, include x = '-' cases.
    """
    jps.validate(expected_time_points=3, expected_outcomes=2)
    dist = jps.distributions
    from itertools import permutations
    values = []
    for A, B, C in permutations([1, 2, 3], 3):
        for x_idx in (0, 1):
            for y_idx in (0, 1):
                for z_idx in (0, 1):
                    z_op_idx = 1 - z_idx
                    
                    # Fix: Check time order to access correct array axes
                    if A < B:
                        p_Ax_By = dist[(A, B)][x_idx, y_idx]
                    else:
                        p_Ax_By = dist[(B, A)][y_idx, x_idx]

                    if A < C:
                        p_Ax_Cz = dist[(A, C)][x_idx, z_idx]
                    else:
                        p_Ax_Cz = dist[(C, A)][z_idx, x_idx]

                    if B < C:
                        p_By_Czop = dist[(B, C)][y_idx, z_op_idx]
                    else:
                        p_By_Czop = dist[(C, B)][z_op_idx, y_idx]

                    expr = p_Ax_Cz + p_By_Czop - p_Ax_By
                    values.append(float(expr))
    return float(min(values))

def wigner_LGI_dichotomic_four_point(jps: JointProbabilitySet) -> float:
    """Four-point Wigner-form LGI (dichotomic outcomes).

    Evaluates inequalities of the form::

        P(A x, B y) - P(A x, C z) - P(B y, D z') - P(C z', D z) <= 0

    where A, B, C, D are permutations of {1,2,3,4}; outcomes use index mapping
    0 → '+', 1 → '-'. To reduce global sign redundancy, fix x_idx=0, while
    y_idx, z_idx ∈ {0,1}, with z'_idx = 1 - z_idx. This yields 24 * 2 * 2 / 2 = 48
    inequalities after symmetry reduction.

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of four time points
        with two dichotomic outcomes.

    Returns
    -------
    float
        The minimum value among the 48 Wigner inequalities.

    Notes
    -----
    - Outcome index convention: 0 → '+', 1 → '-' (descending m-order).
    - Adjust enumeration if a different symmetry reduction is preferred.
    """
    jps.validate(expected_time_points=4, expected_outcomes=2)
    dist = jps.distributions
    from itertools import permutations
    values = []
    for A, B, C, D in permutations([1, 2, 3, 4], 4):
        for x_idx in (0, 1):
            for y_idx in (0, 1):
                for i_idx in (0, 1):
                    for j_idx in (0, 1):
                        i_op_idx = 1 - i_idx
                        j_op_idx = 1 - j_idx
                        
                        if A < B:
                            p_Ax_By = dist[(A, B)][x_idx, y_idx]
                        else:
                            p_Ax_By = dist[(B, A)][y_idx, x_idx]

                        if A < C:
                            p_Ax_Ci = dist[(A, C)][x_idx, i_idx]
                        else:
                            p_Ax_Ci = dist[(C, A)][i_idx, x_idx]

                        if B < D:
                            p_By_Djop = dist[(B, D)][y_idx, j_op_idx]
                        else:
                            p_By_Djop = dist[(D, B)][j_op_idx, y_idx]

                        if C < D:
                            p_Ciop_Dj = dist[(C, D)][i_op_idx, j_idx]
                        else:
                            p_Ciop_Dj = dist[(D, C)][j_idx, i_op_idx]

                        expr = p_Ax_Ci + p_By_Djop + p_Ciop_Dj - p_Ax_By
                        values.append(float(expr))
    return float(min(values))