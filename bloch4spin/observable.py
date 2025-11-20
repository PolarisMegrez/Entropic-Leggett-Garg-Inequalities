"""
bloch4spin: Measurement Superoperators in Bloch Representation
--------------------------------------------------------------
Defines a compact wrapper for measurement-induced superoperators acting on
Bloch vectors. For a Kraus operator ``K``, the action on a density operator is
``ρ -> K ρ K^\dagger``. In Bloch coordinates ``r`` this becomes a linear map
``r' = L_K r`` with components

.. math:: r'_a = \langle k_{a}, r \rangle, \quad k_a \equiv \text{Bloch}(K^{\dagger} T_a K),

so the superoperator matrix is assembled as ``(L_K)_{a b} = \overline{k_a[b]}``.

Public API
----------
- ``GeneralizedBlochObservable``: Stores the superoperator matrix ``L`` of shape ``(d**2, d**2)``.
- ``GeneralizedBlochObservable.from_Kraus``: Build ``L`` from a single Kraus operator ``K``.
- ``GeneralizedBlochObservable.from_POVM_elem``: Build ``L`` from one POVM element ``M = K^{\dagger}K`` using ``K = \sqrt{M}``.
- ``GeneralizedBlochObservable.from_projector``: Rank-1 projector shortcut without tensor products.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

from .basis import (
    GeneralizedBlochVector,
    bloch_dim,
    bloch_tensor_product,
    bloch_hermitian_transpose,
)

__all__ = [
    "GeneralizedBlochObservable",
    "apply_measurement",
]

@dataclass
class GeneralizedBlochObservable:
    """Measurement superoperator acting on Bloch vectors.

    Stores the linear map ``r -> L r`` where ``L`` is a complex array of shape
    ``(d**2, d**2)`` for the currently initialized Bloch dimension ``d``.

    Parameters
    ----------
    data : scipy.sparse.csr_matrix
        Sparse complex matrix ``(d**2, d**2)`` representing the superoperator.

    Raises
    ------
    ValueError
        If ``data`` is not square or cannot be interpreted as complex.
    """

    data: csr_matrix

    def __post_init__(self) -> None:
        d = bloch_dim()
        expected = d * d
        arr = self.data
        if not isspmatrix_csr(arr):
            arr = csr_matrix(np.asarray(arr, dtype=np.complex128))
        if arr.shape != (expected, expected):
            raise ValueError("Observable superoperator must have shape (d^2, d^2).")
        if arr.dtype != np.complex128:
            arr = arr.astype(np.complex128)
        self.data = arr

    @staticmethod
    def from_Kraus(K: Union[np.ndarray, GeneralizedBlochVector]) -> "GeneralizedBlochObservable":
        """Construct the superoperator ``L_K`` for ``ρ -> K ρ K^\dagger``.

        Parameters
        ----------
        K : numpy.ndarray or GeneralizedBlochVector
            Kraus operator. If a matrix is provided, it is converted to Bloch
            coordinates via ``GeneralizedBlochVector.from_matrix``.

        Returns
        -------
        GeneralizedBlochObservable
            The superoperator matrix ``L_K`` satisfying ``r' = L_K r``.

        Notes
        -----
        For each Bloch basis element ``T_a``, defines ``k_a = Bloch(K^\dagger T_a K)``.
        Since ``r'_a = <k_a, r>``, the row ``a`` of ``L_K`` equals ``conj(k_a)``.
        """
        d = bloch_dim()
        D = d * d

        if isinstance(K, GeneralizedBlochVector):
            rK = K
        else:
            rK = GeneralizedBlochVector.from_matrix(np.asarray(K))
        rK_dag = bloch_hermitian_transpose(rK)

        L = np.zeros((D, D), dtype=np.complex128)

        # Unit vectors e_a for each basis index a
        for a in range(D):
            e_a = np.zeros((D,), dtype=complex)
            e_a[a] = 1.0
            rTa = GeneralizedBlochVector(e_a)
            # k_a = Bloch(K^\dagger T_a K)
            k_a = rK_dag * rTa * rK
            # Row a is conj(k_a)
            L[a, :] = np.conj(k_a.data)

        return GeneralizedBlochObservable(csr_matrix(L))

    @staticmethod
    def from_POVM_elem(M: Union[np.ndarray, GeneralizedBlochVector], *, atol: float = 1e-12) -> "GeneralizedBlochObservable":
        """Construct ``L`` from a single POVM element ``M = K^{\dagger}K``.

        Chooses the Kraus operator as the positive square root ``K = \sqrt{M}``.

        Parameters
        ----------
        M : numpy.ndarray or GeneralizedBlochVector
            POVM element. If a Bloch vector is provided, it is converted to a
            matrix via ``to_matrix`` prior to the square root.
        atol : float, optional
            Numerical tolerance for Hermiticity and PSD checks (default ``1e-12``).

        Returns
        -------
        GeneralizedBlochObservable
            The superoperator matrix corresponding to ``ρ -> K ρ K^\dagger`` with ``K=\sqrt{M}``.

        Raises
        ------
        ValueError
            If ``M`` is not Hermitian or has significantly negative eigenvalues.
        """
        if isinstance(M, GeneralizedBlochVector):
            mat = M.to_matrix()
        else:
            mat = np.asarray(M)

        # Hermiticity check
        if not np.allclose(mat, mat.conj().T, atol=atol):
            raise ValueError("POVM element must be Hermitian for square root Kraus.")

        # Eigen decomposition and PSD check
        w, V = np.linalg.eigh(mat)
        if np.any(w < -atol):
            raise ValueError("POVM element is not positive semidefinite.")
        w_clipped = np.clip(w, 0.0, None)
        K = (V * np.sqrt(w_clipped)) @ V.conj().T

        return GeneralizedBlochObservable.from_Kraus(K)

    @staticmethod
    def from_projector(P: Union[np.ndarray, GeneralizedBlochVector]) -> "GeneralizedBlochObservable":
        """Rank-1 projector shortcut ``ρ -> P ρ P``.

        For a one-dimensional projector ``P = |ψ\rangle\langleψ|``, the post-measurement
        (unnormalized) state is ``ρ' = (\langleψ|ρ|ψ\rangle) P``. In Bloch coordinates
        this is ``r' = (\langle r_P, r \rangle) r_P``, so the superoperator is the
        outer product ``L = r_P \; r_P^{\dagger}``.

        Parameters
        ----------
        P : numpy.ndarray or GeneralizedBlochVector
            Projector (no rank check performed).

        Returns
        -------
        GeneralizedBlochObservable
            Superoperator ``L = r_P r_P^{\dagger}``.

        Notes
        -----
        No rank verification is performed. For non–rank-1 inputs, this shortcut
        does not equal ``ρ -> P ρ P``; use :meth:`from_Kraus` with ``K=P`` instead.
        """
        if isinstance(P, GeneralizedBlochVector):
            rP = P
        else:
            rP = GeneralizedBlochVector.from_matrix(np.asarray(P))
        v = rP.data
        L = np.outer(v, np.conj(v))
        return GeneralizedBlochObservable(csr_matrix(L))

def apply_measurement(observables: list[GeneralizedBlochObservable],
                      state,  # GeneralizedBlochState (forward-declared)
                      *,
                      norm_mode: str = "normalized",
                      clip_negative: bool = True,
                      atol: float = 1e-12):
    """Apply a list of measurement superoperators to a Bloch state.

    Each ``GeneralizedBlochObservable`` represents a single outcome's
    (unnormalized) update ``r' = L r``. This function returns the outcome
    probabilities and the corresponding normalized post-measurement states.

    Parameters
    ----------
    observables : list of GeneralizedBlochObservable
        Measurement outcome superoperators ``L_i`` applied independently.
    state : GeneralizedBlochState
        Input Bloch state ``r``.
        norm_mode : {"normalized", "renormalized", "unnormalized"}, optional
                Controls how the observable set normalization is handled.
                - "normalized": enforce completeness by checking ``sum_i L_i[0,0] == 1``;
                    raises ``ValueError`` if violated.
                - "renormalized": skip completeness check and renormalize the output
                    probabilities to sum to 1 (if the total is positive).
                - "unnormalized": skip completeness check and return raw probabilities.
                Default is "normalized".
        clip_negative : bool, optional
        If ``True``, small negative probabilities in ``[-atol, 0)`` are clipped
        to zero for numerical stability.
    atol : float, optional
        Absolute tolerance for treating probabilities as zero.

    Returns
    -------
    list of tuple
        For each observable ``L_i`` a tuple ``(p_i, post_state_i)`` where
        ``p_i`` is the outcome probability and ``post_state_i`` the normalized
        Bloch state after the measurement.

    Notes
    -----
    Probability extraction uses ``p_i = \mathrm{Tr}(ρ'_i)`` with ``ρ'_i`` the
    unnormalized post-measurement operator corresponding to ``r'_i = L_i r``;
    with ``T_0^{(0)} = I/\sqrt{d}``, ``p_i = \sqrt{d} \cdot r'_{00}``.
    Normalization of each post state enforces ``r_{00} = 1/\sqrt{d}``.
    """
    from .evolution import GeneralizedBlochState  # local import to avoid cycle
    d = bloch_dim()
    out: list[tuple[float, GeneralizedBlochState]] = []
    r_in = state.data
    sqrt_d = np.sqrt(d)
    total_prob = 0.0
    probs_tmp: list[float] = []
    states_tmp: list[GeneralizedBlochState] = []

    # Optional completeness check using sum_i L_i[0,0] == 1
    if norm_mode not in ("normalized", "renormalized", "unnormalized"):
        raise ValueError("norm_mode must be 'normalized', 'renormalized', or 'unnormalized'.")
    if norm_mode == "normalized":
        csum = 0.0
        for obs in observables:
            csum += float(np.real(obs.data[0, 0]))
        if not np.isclose(csum, 1.0, atol=10*atol):
            raise ValueError(f"Observable set not complete: sum L_i[0,0] = {csum} ≠ 1.")
    for obs in observables:
        r_prime = obs.data @ r_in  # unnormalized Bloch vector (numpy array)
        p = (sqrt_d * r_prime[0]).real
        if clip_negative and p < 0 and p > -atol:
            p = 0.0
        probs_tmp.append(p)
        total_prob += p
        if p > atol:
            post = GeneralizedBlochState(r_prime)
            # Enforce exact normalization (numerically) via method
            post.normalization()
        else:
            # Zero probability branch: keep vector but attempt normalization for consistency
            post = GeneralizedBlochState(r_prime.copy())
            # If zero vector or near-zero, leave as is; else try normalization guarded
            try:
                if np.abs(r_prime[0]) > atol:
                    post.normalization()
            except Exception:
                pass
        states_tmp.append(post)
    # Optional global renormalization depending on norm_mode
    if norm_mode == "renormalized" and total_prob > 0:
        scale = 1.0 / total_prob
        for i in range(len(probs_tmp)):
            probs_tmp[i] *= scale
    for p, st in zip(probs_tmp, states_tmp):
        out.append((p, st))
    return out