r"""lgbloch.lgi: Leggett-Garg Inequality Testing Functions
======================================================
Provides various forms of Leggett-Garg inequality (LGI) tests for quantum macrorealism verification, including entropic formulations, standard dichotomic tests, and Wigner-form inequalities. Supports both general n-time cases and specific order analyses.

References
----------
- Q.-H. Cai, X.-H. Yu, M.-C. Yang, A.-X. Liu, C.-F. Qiao, "Conditions for Quantum Violation of Macrorealism in Large-spin Limit," arXiv:2505.13162, 2025.
- A. R. U. Devi, H. S. Karthik, Sudha, A. K. Rajagopal, "Macrorealism from Entropic Leggett-Garg Inequalities," *Phys. Rev. A* 87, 052103 (2013).
- J. J. Halliwell, "Necessary and Sufficient Conditions for Macrorealism Using Two- and Three-Time Leggett-Garg Inequalities," *J. Phys.: Conf. Ser.* 1275, 012008 (2019).
- J. J. Halliwell, "Leggett-Garg Tests of Macrorealism: Checks for Noninvasiveness and Generalizations to Higher-Order Correlators," *Phys. Rev. A* 99, 022119 (2019).
- D. Saha, S. Mal, P. K. Panigrahi, D. Home, "Wigner's Form of the Leggett-Garg Inequality, the No-Signaling-in-Time Condition, and Unsharp Measurements," *Phys. Rev. A* 91, 032117 (2015).

Public API
----------
- ``shannon_entropy``: Compute Shannon entropy of probability distributions.
- ``entropic_LGI``: General entropic LGI test using Shannon cone basis elements.
- ``entropic_LGI_for_order_k``: Entropic LGI test for specific order k inequalities.
- ``entropic_LGI_from_chain_rule``: Chain rule-based entropic LGI test (Devi et al.).
- ``print_entropic_LGI_forms_for_order_k``: Print symbolic forms of LGI inequalities.
- ``standard_LGI_dichotomic``: Standard-form LGI for dichotomic outcomes (Halliwell).
- ``wigner_LGI_dichotomic``: Wigner-form LGI using chain inequalities.

Notes
-----
- Return values: All LGI tests return violation values (negative for quantum violations).
- Entropy: Entropic tests use Shannon entropy with base-2 logarithm.
- Measurement: Standard and Wigner forms are specifically for dichotomic (two-outcome) measurements.
- Computation: Iterates over all non-empty subsets of measurement times.

"""

from itertools import combinations, permutations, product

import numpy as np

from .engine import JointProbabilitySet

__all__ = [
    "shannon_entropy",
    "entropic_LGI",
    "entropic_LGI_for_order_k",
    "entropic_LGI_from_chain_rule",
    "print_entropic_LGI_forms_for_order_k",
    "standard_LGI_dichotomic",
    "wigner_LGI_dichotomic",
]


def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Parameters
    ----------
    p : numpy.ndarray
        Probability distribution (will be normalized automatically).

    Returns
    -------
    float
        Shannon entropy in bits.

    Notes
    -----
    The function automatically normalizes the input distribution and uses
    base-2 logarithm. Zero probabilities are handled safely.

    """
    p = np.asarray(p, dtype=float).ravel()
    total = float(np.sum(p))
    if total <= 0.0:
        return 0.0
    p = p / total
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def entropic_LGI(n: int, jps: JointProbabilitySet) -> float:
    r"""General entropic LGI test for n time points using Shannon cone basis elements.

    Iterates over all subsets of measurement points and computes elementary
    inequalities:
    1. Conditional Entropy: 
       ``H(S) - H(S \ {i}) >= 0``
    2. Conditional Mutual Information: 
       ``H(S \ {i}) + H(S \ {j}) - H(S) - H(S \ {i, j}) >= 0``

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

    References
    ----------
    - Q.H. Cai, X.H. Yu, M.C. Yang, A.X. Liu, C.F. Qiao.
      Conditions for Quantum Violation of Macrorealism in Large-spin Limit.
      http://arxiv.org/abs/2505.13162 (2025).

    """
    jps.validate(expected_time_points=n)
    dist = jps.distributions

    def H(subset):
        if not subset:
            return 0.0
        return shannon_entropy(dist[tuple(sorted(subset))])

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
                    val = (
                        H(subset_set - {i})
                        + H(subset_set - {j})
                        - H(subset_set)
                        - H(subset_set - {i, j})
                    )
                    values.append(val)

    return float(np.min(values)) if values else 0.0


def entropic_LGI_for_order_k(k: int, n: int, jps: JointProbabilitySet) -> float:
    r"""Entropic LGI test for a specific order k.

    Computes the minimum value of inequalities belonging to order k:
    1. Type 1 & 2 inequalities from subsets of size k.
    2. Type 3 (derivative) inequalities from subsets of size k+1.

    Parameters
    ----------
    k : int
        Order of the inequality.
    n : int
        Total number of time points available.
    jps : JointProbabilitySet
        Joint probability tensors.

    Returns
    -------
    float
        Minimum value of the inequalities for order k.

    References
    ----------
    - Q.H. Cai, X.H. Yu, M.C. Yang, A.X. Liu, C.F. Qiao.
      Conditions for Quantum Violation of Macrorealism in Large-spin Limit.
      http://arxiv.org/abs/2505.13162 (2025).

    """
    jps.validate(expected_time_points=n)
    dist = jps.distributions
    values = []

    def H(subset):
        if not subset:
            return 0.0
        return shannon_entropy(dist[tuple(sorted(subset))])

    # 1. Inequalities from subsets of size k (Type 1 & Type 2)
    for subset in combinations(range(1, n + 1), k):
        subset_set = set(subset)

        # Type 1: D_i^(S) = H(S) - H(S \ {i}) >= 0
        for i in subset_set:
            val = H(subset_set) - H(subset_set - {i})
            values.append(val)

        # Type 2: D_{i,j}^(S) = H(S \ {i}) + H(S \ {j}) - H(S) - H(S \ {i, j}) >= 0
        for i, j in combinations(subset_set, 2):
            val = (
                H(subset_set - {i})
                + H(subset_set - {j})
                - H(subset_set)
                - H(subset_set - {i, j})
            )
            values.append(val)

    # 2. Inequalities from subsets of size k+1 (Type 3)
    # D_{i,j|k}^(S) = D_k^(S) + D_{i,j}^(S)
    # This belongs to order k (one order lower than S)
    if k + 1 <= n:
        for subset in combinations(range(1, n + 1), k + 1):
            subset_set = set(subset)
            # Iterate over distinct i, j, k_idx in S
            for k_idx in subset_set:
                # Remaining set for i, j
                remaining = subset_set - {k_idx}
                for i, j in combinations(remaining, 2):
                    # D_{i,j|k}^(S) = H(S\{i}) + H(S\{j}) - H(S\{k}) - H(S\{i,j})
                    val = (
                        H(subset_set - {i})
                        + H(subset_set - {j})
                        - H(subset_set - {k_idx})
                        - H(subset_set - {i, j})
                    )
                    values.append(val)

    return float(np.min(values)) if values else 0.0


def entropic_LGI_from_chain_rule(n: int, jps: JointProbabilitySet) -> float:
    r"""Entropic LGI test based on the chain rule inequality (Devi et al.).

    Iterates over all subsets of size k (3 <= k <= n) and computes the chain inequality:

    ``\sum_{m=1}^{k-1} H(Q_{i_m}, Q_{i_{m+1}}) - \sum_{m=2}^{k-1} H(Q_{i_m}) - H(Q_{i_1}, Q_{i_k}) >= 0``

    where indices are sorted time points of the subset.

    Parameters
    ----------
    n : int
        Number of time points.
    jps : JointProbabilitySet
        Joint probability tensors.

    Returns
    -------
    float
        Minimum value of the chain inequalities over all subsets.

    References
    ----------
    - A.R.U. Devi, H.S. Karthik, Sudha, A.K. Rajagopal.
      Macrorealism from entropic Leggett-Garg inequalities.
      Physical Review A 87(5), 052103 (2013).

    """
    jps.validate(expected_time_points=n)
    dist = jps.distributions

    def H(subset):
        return shannon_entropy(dist[tuple(sorted(subset))])

    values = []
    # Iterate over subset sizes k from 3 to n
    for k in range(3, n + 1):
        for subset in combinations(range(1, n + 1), k):
            indices = sorted(list(subset))

            term1 = 0.0
            for m in range(k - 1):
                term1 += H({indices[m], indices[m + 1]})

            term2 = 0.0
            for m in range(1, k - 1):
                term2 += H({indices[m]})

            term3 = H({indices[0], indices[k - 1]})

            values.append(float(term1 - term2 - term3))

    return float(np.min(values)) if values else 0.0


def print_entropic_LGI_forms_for_order_k(k: int, n: int) -> None:
    r"""Print the symbolic forms of entropic LGI inequalities for a specific order k.

    This function generates and prints the inequalities that would be computed
    by `entropic_LGI_for_order_k`, useful for verification.

    Parameters
    ----------
    k : int
        Order of the inequality.
    n : int
        Total number of time points available.

    """
    print(f"--- Entropic LGI Forms for Order k={k} (n={n}) ---")

    def fmt_H(subset):
        if not subset:
            return "0"
        return f"H({{{', '.join(map(str, sorted(subset)))}}})"

    # 1. Inequalities from subsets of size k (Type 1 & Type 2)
    print(f"\n[Subsets of size {k}]")
    for subset in combinations(range(1, n + 1), k):
        subset_set = set(subset)
        S_str = fmt_H(subset_set)

        # Type 1
        for i in subset_set:
            term1 = S_str
            term2 = fmt_H(subset_set - {i})
            print(f"Type 1 (i={i}): {term1} - {term2} >= 0")

        # Type 2
        for i, j in combinations(subset_set, 2):
            term1 = fmt_H(subset_set - {i})
            term2 = fmt_H(subset_set - {j})
            term3 = S_str
            term4 = fmt_H(subset_set - {i, j})
            print(f"Type 2 (i={i}, j={j}): {term1} + {term2} - {term3} - {term4} >= 0")

    # 2. Inequalities from subsets of size k+1 (Type 3)
    if k + 1 <= n:
        print(f"\n[Subsets of size {k+1}] (Derivative Type 3)")
        for subset in combinations(range(1, n + 1), k + 1):
            subset_set = set(subset)
            for k_idx in subset_set:
                remaining = subset_set - {k_idx}
                for i, j in combinations(remaining, 2):
                    term1 = fmt_H(subset_set - {i})
                    term2 = fmt_H(subset_set - {j})
                    term3 = fmt_H(subset_set - {k_idx})
                    term4 = fmt_H(subset_set - {i, j})
                    print(
                        f"Type 3 (k={k_idx} | i={i}, j={j}): {term1} + {term2} - {term3} - {term4} >= 0"
                    )
    else:
        print(f"\n[Subsets of size {k+1}] None (k+1 > n)")
    print("-" * 60)


def standard_LGI_dichotomic(n: int, jps: JointProbabilitySet) -> float:
    """Standard-form LGI for n time points (dichotomic outcomes).

    Parameters
    ----------
    n : int
        Number of time points.
    jps : JointProbabilitySet
        Joint probability tensors for all non-empty subsets of n time points
        with two measurement outcomes (dichotomic).

    Returns
    -------
    float
        The minimum reconstructed probability among the 2^n assignments.

    Notes
    -----
    - Interprets the two outcomes directly as ``s ∈ {+1, -1}`` per time.
    - Reconstructs ``p(s_1, ..., s_n)`` via the dichotomic expansion
      and checks non-negativity for all 2^n assignments.

    References
    ----------
    - J. J. Halliwell, "Leggett-Garg tests of macrorealism: Checks for
      noninvasiveness and generalizations to higher-order correlators,"
      Phys. Rev. A 99, 022119 (2019).
    - J. J. Halliwell, "Necessary and sufficient conditions for macrorealism
      using two- and three-time Leggett-Garg inequalities,"
      J. Phys.: Conf. Ser. 1275, 012008 (2019).

    """
    jps.validate(expected_time_points=n, expected_outcomes=2)
    dist = jps.distributions
    s_vec = np.array([1.0, -1.0])

    # Precompute correlations for all subsets
    correlations = {}
    for k in range(1, n + 1):
        for subset in combinations(range(1, n + 1), k):
            subset = tuple(sorted(subset))
            d = dist[subset]
            term = d
            for axis in range(k):
                shape = [1] * k
                shape[axis] = 2
                s_reshaped = s_vec.reshape(shape)
                term = term * s_reshaped
            correlations[subset] = float(np.sum(term))

    min_val = 1e300

    # Iterate all 2^n assignments
    for s_vals in product([-1.0, 1.0], repeat=n):
        total = 1.0
        for k in range(1, n + 1):
            for subset in combinations(range(1, n + 1), k):
                subset = tuple(sorted(subset))
                corr = correlations[subset]
                sign_prod = 1.0
                for time_idx in subset:
                    sign_prod *= s_vals[time_idx - 1]
                total += corr * sign_prod

        p = total / (2**n)
        if p < min_val:
            min_val = p

    return float(min_val)


def wigner_LGI_dichotomic(n: int, jps: JointProbabilitySet) -> float:
    """Wigner-form LGI for n time points (dichotomic outcomes).

    Evaluates chain inequalities of the form:
    P(Start, End) <= P(Start, M1) + P(M1', M2) + ... + P(Mk', End)

    Parameters
    ----------
    n : int
        Number of time points.
    jps : JointProbabilitySet
        Joint probability tensors.

    Returns
    -------
    float
        The minimum value among the Wigner inequalities (should be >= 0 for
        macrorealism). Returns negative value if violated.

    References
    ----------
    - D. Saha, S. Mal, P.K. Panigrahi, D. Home. "Wigner's form of the Leggett-
      Garg inequality, the no-signaling-in-time condition, and unsharp
      measurements," Physical Review A 91, 032117 (2015).

    """
    jps.validate(expected_time_points=n, expected_outcomes=2)
    dist = jps.distributions

    if n < 3:
        return 0.0

    values = []

    for p in permutations(range(1, n + 1), n):
        A = p[0]
        B = p[1]
        M = p[2:]
        num_intermediates = len(M)

        for x in (0, 1):
            for y in (0, 1):
                for z in product((0, 1), repeat=num_intermediates):
                    chain_sum = 0.0

                    # 1. P(A=x, M_1=z[0])
                    t1, t2 = A, M[0]
                    o1, o2 = x, z[0]
                    if t1 < t2:
                        prob = dist[(t1, t2)][o1, o2]
                    else:
                        prob = dist[(t2, t1)][o2, o1]
                    chain_sum += prob

                    # 2. Middle links
                    for k in range(num_intermediates - 1):
                        t1, t2 = M[k], M[k + 1]
                        o1, o2 = 1 - z[k], z[k + 1]
                        if t1 < t2:
                            prob = dist[(t1, t2)][o1, o2]
                        else:
                            prob = dist[(t2, t1)][o2, o1]
                        chain_sum += prob

                    # 3. Last link
                    t1, t2 = M[-1], B
                    o1, o2 = 1 - z[-1], y
                    if t1 < t2:
                        prob = dist[(t1, t2)][o1, o2]
                    else:
                        prob = dist[(t2, t1)][o2, o1]
                    chain_sum += prob

                    # Direct term
                    t1, t2 = A, B
                    o1, o2 = x, y
                    if t1 < t2:
                        direct = dist[(t1, t2)][o1, o2]
                    else:
                        direct = dist[(t2, t1)][o2, o1]

                    values.append(float(chain_sum - direct))

    return float(min(values))
