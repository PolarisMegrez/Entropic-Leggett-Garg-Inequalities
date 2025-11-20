"""LGI boolean test functions.

This module exposes boolean-valued Leggett–Garg inequality (LGI) tests that
operate on a `JointProbabilitySet` produced by the engine. Each test embeds
the expected number of outcomes and time points and validates the input before
performing its calculation. Implementations are NumPy-only for clarity and
maintainability; no optional JIT/parallel paths are used.
"""

from typing import Tuple
import numpy as np
from .engine import JointProbabilitySet

__all__ = [
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

def entropic_LGI_three_point(jps: JointProbabilitySet) -> float:
    """Three-point entropic LGI test.

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of three time points.
        Outcome count is unconstrained; only the number of time points must be 3.
    atol : float, optional
        Numerical tolerance. An inequality is considered satisfied if its value is
        greater than or equal to ``-atol``. Default is ``1e-10``.

    Returns
    -------
    float
        The minimum value among the checked entropic combinations. Values ≥ 0
        indicate satisfaction (use ``atol`` outside to set a tolerance threshold).

    Notes
    -----
    - Invasive semantics: each subset distribution is computed independently (no marginalization).
    - The following linear combinations of Shannon entropies are checked (all must be ≥ 0 up to ``atol``):

    References
    ----------
    - Q.-H. Cai, X.-H. Yu, et al., "Conditions for Quantum Violation of Macrorealism in Large-Spin Limit," arXiv:2505.13162 (2025).
    """
    # Entropic test does not constrain outcome_num; require 3 time points
    jps.validate(expected_time_points=3)
    dist = jps.distributions
    # 1-point entropies
    h1   = _shannon_entropy_bits(dist[(1,)])
    h2   = _shannon_entropy_bits(dist[(2,)])
    h3   = _shannon_entropy_bits(dist[(3,)])
    # 2-point entropies
    h12  = _shannon_entropy_bits(dist[(1, 2)])
    h23  = _shannon_entropy_bits(dist[(2, 3)])
    h13  = _shannon_entropy_bits(dist[(1, 3)])
    # 3-point entropies
    h123 = _shannon_entropy_bits(dist[(1, 2, 3)])
    
    combos = []
    # order-3 type
    combos.extend([
        h123 - h23,
        h123 - h13,
        h13 + h23 - h123 - h3,
        h12 + h13 - h123 - h1,
        h12 + h23 - h123 - h2
    ])
    # order-2 type
    combos.extend([
        h12 - h2, h1 + h2 - h12,
        h23 - h3, h2 + h3 - h23,
        h13 - h3, h1 + h3 - h13,
        h23 + h12 - h2 - h13,
        h13 + h23 - h3 - h12,
        h13 + h12 - h1 - h23,
    ])
    return float(np.min(combos))

def entropic_LGI_four_point(jps: JointProbabilitySet) -> float:
    """Four-point entropic LGI test.

    Parameters
    ----------
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of four time points.
        Outcome count is unconstrained; only the number of time points must be 4.
    atol : float, optional
        Numerical tolerance. An inequality is considered satisfied if its value is
        greater than or equal to ``-atol``. Default is ``1e-10``.

    Returns
    -------
    float
        The minimum value among the checked entropic combinations. Values ≥ 0
        indicate satisfaction (use ``atol`` outside to set a tolerance threshold).

    Notes
    -----
    - Invasive semantics: each subset distribution is computed independently (no marginalization).
    - Checks the union of linear combinations across categories (order-2, pseudo-order-3, order-3,
      pseudo-order-4, order-4); all must be ≥ 0 up to ``atol``.

    References
    ----------
    - Q.-H. Cai, X.-H. Yu, et al., "Conditions for Quantum Violation of Macrorealism in Large-Spin Limit," arXiv:2505.13162 (2025).
    """
    jps.validate(expected_time_points=4)
    dist = jps.distributions
    # 1-point entropies
    h1 = _shannon_entropy_bits(dist[(1,)])
    h2 = _shannon_entropy_bits(dist[(2,)])
    h3 = _shannon_entropy_bits(dist[(3,)])
    h4 = _shannon_entropy_bits(dist[(4,)])
    # 2-point entropies
    h12 = _shannon_entropy_bits(dist[(1, 2)])
    h13 = _shannon_entropy_bits(dist[(1, 3)])
    h14 = _shannon_entropy_bits(dist[(1, 4)])
    h23 = _shannon_entropy_bits(dist[(2, 3)])
    h24 = _shannon_entropy_bits(dist[(2, 4)])
    h34 = _shannon_entropy_bits(dist[(3, 4)])
    # 3-point entropies
    h123 = _shannon_entropy_bits(dist[(1, 2, 3)])
    h124 = _shannon_entropy_bits(dist[(1, 2, 4)])
    h134 = _shannon_entropy_bits(dist[(1, 3, 4)])
    h234 = _shannon_entropy_bits(dist[(2, 3, 4)])
    # 4-point entropy
    h1234 = _shannon_entropy_bits(dist[(1, 2, 3, 4)])

    combos = []
    # order-2 type
    combos.extend([
        h12 - h2, h1 + h2 - h12,
        h13 - h3, h1 + h3 - h13,
        h14 - h4, h1 + h4 - h14,
        h23 - h3, h2 + h3 - h23,
        h24 - h4, h2 + h4 - h24,
        h34 - h4, h3 + h4 - h34,
    ])
    # pseudo order-3 type
    combos.extend([
        h23 + h12 - h2 - h13,
        h13 + h23 - h3 - h12,
        h13 + h12 - h1 - h23,
        h24 + h12 - h2 - h14,
        h14 + h24 - h4 - h12,
        h14 + h12 - h1 - h24,
        h34 + h13 - h3 - h14,
        h14 + h34 - h4 - h13,
        h14 + h13 - h1 - h34,
        h34 + h23 - h3 - h24,
        h24 + h34 - h4 - h23,
        h24 + h23 - h2 - h34,
    ])
    # order-3 type
    combos.extend([
        h123 - h23, h123 - h13,
        h124 - h12, h124 - h14,
        h134 - h13, h134 - h34,
        h234 - h23, h234 - h24,
        h13 + h23 - h123 - h3,
        h12 + h13 - h123 - h1,
        h12 + h23 - h123 - h2,
        h14 + h12 - h124 - h1,
        h14 + h24 - h124 - h4,
        h14 + h12 - h124 - h2,
        h13 + h34 - h134 - h3,
        h14 + h34 - h134 - h4,
        h14 + h13 - h134 - h1,
        h23 + h34 - h234 - h3,
        h24 + h34 - h234 - h4,
        h24 + h23 - h234 - h2,
    ])
    # pseudo order-4 type
    combos.extend([
        h134 + h234 - h123 - h34,
        h124 + h134 - h123 - h14,
        h124 + h234 - h123 - h24,
        h134 + h123 - h124 - h13,
        h134 + h234 - h124 - h34,
        h134 + h123 - h124 - h23,
        h123 + h234 - h134 - h23,
        h124 + h234 - h134 - h24,
        h124 + h123 - h134 - h12,
        h123 + h134 - h234 - h13,
        h124 + h134 - h234 - h14,
        h124 + h123 - h234 - h12,
    ])
    # order-4 type
    combos.extend([
        h1234 - h124, h1234 - h134, h1234 - h234,
        h123 + h124 - h12 - h1234,
        h134 + h123 - h13 - h1234,
        h134 + h124 - h14 - h1234,
        h234 + h123 - h23 - h1234,
        h124 + h234 - h24 - h1234,
        h134 + h234 - h34 - h1234,
    ])
    return float(np.min(combos))
    

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