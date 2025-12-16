r"""bloch4spin: Evolution of operators and states in Bloch representation
---------------------------------------------------------------------
Provides dataclass wrappers for Hamiltonians, Liouvillian evolution matrices,
and density operators expressed in the canonical orthonormal Bloch basis
(COBITO) defined in ``bloch4spin.basis``. The Bloch vector ``r`` obeys

.. math:: \frac{dr}{dt} = L r,

where ``L`` collects both unitary and dissipative contributions.

Unitary (Hamiltonian) part
-------------------------
For a Hermitian Hamiltonian ``H`` the Liouville–von Neumann equation
``\dot{\rho} = -i[H,\rho]`` yields

.. math:: (L_H)_{ab} = -i\,\mathrm{Tr}\big(T_a^{\dagger}[H, T_b]\big)
          = -i\,\sum_c h_c\, f_{a b}^{\;c},

with Bloch coefficients ``h_c`` and structure constants ``f_{ab}^{\;c}``.

Dissipative (Lindblad) part
---------------------------
For a single Lindblad jump operator ``K`` the dissipator
``\mathcal{L}_K[\rho] = K\rho K^{\dagger} - \tfrac12\{K^{\dagger}K,\rho\}`` has

.. math:: (L_K)_{ab} = \mathrm{Tr}\!\left[T_a^{\dagger}\!\left(K T_b K^{\dagger}
          - \tfrac12 T_b K^{\dagger}K - \tfrac12 K^{\dagger}K T_b\right)\right].

Public API
----------
- ``GeneralizedBlochHamiltonian``: Hermitian Hamiltonian in Bloch space.
- ``GeneralizedBlochEvolutionMatrix``: Liouvillian / evolution matrix acting on Bloch vectors.
- ``GeneralizedBlochState``: Hermitian density operator represented as a Bloch vector.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
from scipy.linalg import expm as dense_expm
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr
from scipy.sparse.linalg import expm_multiply

from .basis import (
    GeneralizedBlochVector,
    _kq_from_idx,
    bloch_dim,
    bloch_hermitian_transpose,
    bloch_inner_product,
    structure_const,
)

__all__ = [
    "GeneralizedBlochHamiltonian",
    "GeneralizedBlochEvolutionMatrix",
    "GeneralizedBlochState",
]


@dataclass
class GeneralizedBlochHamiltonian(GeneralizedBlochVector):
    """Hermitian Hamiltonian in Bloch space.

    Represents a Hermitian operator ``H`` through its Bloch (COBITO) expansion
    coefficients ``h_a``. Hermiticity is verified at construction time.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional complex array of length ``d**2`` containing the Bloch
        coefficients of the Hamiltonian.

    Raises
    ------
    ValueError
        If the supplied Bloch coefficients do not satisfy Hermiticity under
        the transformation implemented by ``bloch_hermitian_transpose``.

    Notes
    -----
    Hermiticity test uses ``np.allclose`` with ``rtol=1e-10`` and
    ``atol=1e-12``; adjust at call site by post-validating if tighter
    tolerances are required.

    """

    def __post_init__(self) -> None:
        """Initialize GeneralizedBlochHamiltonian after dataclass construction.

        Verifies Hermiticity of the Hamiltonian and raises ValueError if
        validation fails.

        """
        super().__post_init__()
        # Hermiticity check
        s = bloch_hermitian_transpose(self)

        d1 = self.data
        d2 = s.data

        if issparse(d1):
            d1 = d1.toarray()
        if issparse(d2):
            d2 = d2.toarray()

        if not np.allclose(d1, d2, rtol=1e-10, atol=1e-12):
            raise ValueError("Hamiltonian must be Hermitian in operator space.")

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "GeneralizedBlochHamiltonian":
        """Create a Bloch-space Hamiltonian from a matrix.

        Converts a Hermitian matrix ``H`` (shape ``(d,d)``) into Bloch
        coefficients via the basis inner products.

        Parameters
        ----------
        mat : numpy.ndarray
            Square ``(d,d)`` complex Hermitian matrix representing ``H``.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Bloch-space Hamiltonian with validated Hermiticity.

        Raises
        ------
        ValueError
            If ``mat`` is not Hermitian within numerical tolerance.

        """
        gv = GeneralizedBlochVector.from_matrix(mat)
        return GeneralizedBlochHamiltonian(gv.data)

    # Restrict scaling to real scalars only; disallow scalar/vector reversed
    # division
    def __mul__(self, other):
        """Multiply by real scalar.

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Scaled Hamiltonian.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(self.data * float(other))
        return NotImplemented

    def __rmul__(self, other):
        """Right-side multiplication by real scalar (commutative).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Scaled Hamiltonian.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(float(other) * self.data)
        return NotImplemented

    def __truediv__(self, other):
        """Divide by real scalar.

        Parameters
        ----------
        other : scalar
            Real scalar divisor.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Scaled Hamiltonian.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(self.data / float(other))
        return NotImplemented

    def __imul__(self, other):
        """In-place multiplication by real scalar (modifies self).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Self (modified in-place).

        """
        if np.isscalar(other) and np.isrealobj(other):
            self.data *= float(other)
            # Hermiticity check happens on demand (construction already ensured)
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """In-place division by real scalar (modifies self).

        Parameters
        ----------
        other : scalar
            Real scalar divisor.

        Returns
        -------
        GeneralizedBlochHamiltonian
            Self (modified in-place).

        """
        if np.isscalar(other) and np.isrealobj(other):
            self.data /= float(other)
            return self
        return NotImplemented


@dataclass
class GeneralizedBlochEvolutionMatrix:
    """Liouvillian evolution matrix acting on Bloch vectors.

    Encodes the linear map ``r -> L r`` for both unitary and dissipative
    (Lindblad) dynamics in Bloch coordinates. For dimension ``d`` the shape is
    ``(d**2, d**2)``.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional complex array of shape ``(d**2, d**2)`` giving the
        evolution matrix elements.

    Raises
    ------
    ValueError
        If ``data`` is not square or cannot be cast to complex type.

    See Also
    --------
    GeneralizedBlochHamiltonian : Source for constructing unitary evolution.
    GeneralizedBlochState : Target objects evolved by this matrix.
    GeneralizedBlochObservable : Construction patterns for Kraus/superoperators.

    Notes
    -----
    In operator space, unitary dynamics generate an anti-Hermitian commutator.
    In Bloch coordinates, ``L`` need not be anti-Hermitian but its spectrum
    reflects trace preservation and Hermiticity constraints.

    """

    data: csr_matrix
    # Internal cache for small dense expm(t*L) matrices
    _dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _expm_cache: "OrderedDict[float, np.ndarray]" = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _expm_cache_maxsize: int = field(default=512, init=False, repr=False)
    _expm_cache_round: int = field(default=12, init=False, repr=False)
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        """Initialize GeneralizedBlochEvolutionMatrix after dataclass construction.

        Accepts dense or sparse matrices and stores as CSR complex128 format.
        Validates that the matrix is square.

        """
        # Accept dense or sparse; store as CSR complex128
        arr = self.data
        if not isspmatrix_csr(arr):
            arr = csr_matrix(np.asarray(arr, dtype=np.complex128))
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Evolution matrix must be square (d^2 x d^2).")
        # Ensure complex128 dtype
        if arr.dtype != np.complex128:
            arr = arr.astype(np.complex128)
        self.data = arr

    # Small-dimension helper: cached dense expm(t*L)
    def _get_dense(self) -> np.ndarray:
        """Convert sparse matrix to dense and cache the result.

        Returns
        -------
        numpy.ndarray
            Dense representation of the evolution matrix.

        """
        if self._dense is None:
            self._dense = self.data.toarray()
        return self._dense

    def _dense_expm_cached(self, t: float) -> np.ndarray:
        """Compute and cache matrix exponential for small systems.

        Parameters
        ----------
        t : float
            Evolution time.

        Returns
        -------
        numpy.ndarray
            Matrix exponential ``exp(t*L)``.

        Notes
        -----
        Uses LRU cache with configurable maximum size and rounding precision
        to avoid recomputing frequently-used time points.

        """
        key = round(float(t), self._expm_cache_round)
        cache = self._expm_cache

        with self._cache_lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

        Ld = self._get_dense()
        mat = dense_expm(Ld * key)

        with self._cache_lock:
            cache[key] = mat
            if len(cache) > self._expm_cache_maxsize:
                cache.popitem(last=False)
        return mat

    # NumPy interop
    def __array__(self, dtype=None) -> np.ndarray:
        """Convert to NumPy array (optionally cast).

        Parameters
        ----------
        dtype : data-type, optional
            If specified, cast the result to this type.

        Returns
        -------
        numpy.ndarray
            Dense array representation.

        """
        # Provide dense view for NumPy interop when needed
        arr = self.data.toarray()
        return arr.astype(dtype) if dtype is not None else arr

    # Helpers
    @staticmethod
    def _is_scalar(x: Any) -> bool:
        """Check if ``x`` is a scalar value.

        Parameters
        ----------
        x : Any
            Value to check.

        Returns
        -------
        bool
            True if ``x`` is a NumPy or Python scalar.

        """
        return np.isscalar(x) or isinstance(x, np.generic)

    @staticmethod
    def _as_2d_array(x: Any) -> np.ndarray | None:
        """Convert ``x`` to a 2D array if possible.

        Parameters
        ----------
        x : Any
            Input to convert.

        Returns
        -------
        numpy.ndarray or None
            2D array if conversion succeeds, None otherwise.

        """
        try:
            arr = np.asarray(x)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        return arr

    @classmethod
    def _wrap_result(cls, arr: np.ndarray) -> "GeneralizedBlochEvolutionMatrix":
        """Wrap an array as a GeneralizedBlochEvolutionMatrix.

        Parameters
        ----------
        arr : numpy.ndarray
            Matrix data.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            New evolution matrix instance.

        """
        return cls(arr)

    def copy(self) -> "GeneralizedBlochEvolutionMatrix":
        """Return a deep copy of the evolution matrix."""
        return self.__class__(self.data.copy())

    def __repr__(self) -> str:
        """Return string representation of the evolution matrix.

        Returns
        -------
        str
            String representation showing shape and dtype.

        """
        n = self.data.shape[0] if self.data.ndim == 2 else 0
        return f"GeneralizedBlochEvolutionMatrix(shape=({n},{n}), dtype=complex)"

    # Arithmetic operators (elementwise for matrices)
    def _ensure_same_shape(self, other_arr: np.ndarray) -> bool:
        """Check if other_arr has the same shape as self.

        Parameters
        ----------
        other_arr : numpy.ndarray
            Array to compare.

        Returns
        -------
        bool
            True if other_arr is 2D and matches self.data.shape.

        """
        return other_arr.ndim == 2 and other_arr.shape == self.data.shape

    def __add__(self, other):
        """Add two evolution matrices (elementwise).

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to add.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Elementwise sum.

        Raises
        ------
        ValueError
            If matrices have incompatible shapes.

        """
        if isinstance(other, GeneralizedBlochEvolutionMatrix):
            if self.data.shape != other.data.shape:
                raise ValueError(
                    f"Inconsistent shapes in addition: {self.data.shape} vs "
                    f"{other.data.shape}"
                )
            return self._wrap_result(self.data + other.data)
        return NotImplemented

    def __radd__(self, other):
        """Right-side addition (commutative).

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to add.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Elementwise sum.

        """
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract two evolution matrices (elementwise).

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to subtract.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Elementwise difference.

        """
        if isinstance(other, GeneralizedBlochEvolutionMatrix):
            return self._wrap_result(self.data - other.data)
        return NotImplemented

    def __rsub__(self, other):
        """Right-side subtraction.

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to subtract from.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Elementwise difference.

        """
        if isinstance(other, GeneralizedBlochEvolutionMatrix):
            return self._wrap_result(other.data - self.data)
        return NotImplemented

    def __mul__(self, other):
        """Multiply by real scalar (elementwise scaling).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Scaled matrix.

        """
        # Only scalar scaling is supported
        if self._is_scalar(other) and np.isrealobj(other):
            return self._wrap_result(self.data * float(other))
        return NotImplemented

    def __rmul__(self, other):
        """Right-side multiplication by real scalar (commutative).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Scaled matrix.

        """
        if self._is_scalar(other) and np.isrealobj(other):
            return self._wrap_result(float(other) * self.data)
        return NotImplemented

    def __truediv__(self, other):
        """Division not supported for sparse matrices.

        Parameters
        ----------
        other : scalar
            Divisor.

        Returns
        -------
        NotImplemented
            Division is not supported.

        """
        # Sparse matrices do not support scalar division efficiently
        return NotImplemented

    def __neg__(self):
        """Negate the evolution matrix.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Negated matrix.

        """
        return self._wrap_result(-self.data)

    def __iadd__(self, other):
        """In-place addition (modifies self).

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to add.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Self (modified in-place).

        """
        if isinstance(other, GeneralizedBlochEvolutionMatrix):
            self.data = (self.data + other.data).tocsr()
            return self
        return NotImplemented

    def __isub__(self, other):
        """In-place subtraction (modifies self).

        Parameters
        ----------
        other : GeneralizedBlochEvolutionMatrix
            Matrix to subtract.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Self (modified in-place).

        """
        if isinstance(other, GeneralizedBlochEvolutionMatrix):
            self.data = (self.data - other.data).tocsr()
            return self
        return NotImplemented

    def __imul__(self, other):
        """In-place multiplication by real scalar (modifies self).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Self (modified in-place).

        """
        if self._is_scalar(other) and np.isrealobj(other):
            self.data = (self.data * float(other)).tocsr()
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """Division not supported for sparse matrices.

        Parameters
        ----------
        other : scalar
            Divisor.

        Returns
        -------
        NotImplemented
            Division is not supported.

        """
        # Not supported for sparse efficiently
        return NotImplemented

    @staticmethod
    def from_Hamiltonian(Ham: Any) -> "GeneralizedBlochEvolutionMatrix":
        r"""Construct the Liouvillian ``L`` for unitary evolution.

        Accepts a Hamiltonian specified either as a Bloch-space object or as
        a matrix and guarantees Hermiticity by first instantiating a
        ``GeneralizedBlochHamiltonian``.

        Parameters
        ----------
        Ham : GeneralizedBlochHamiltonian or GeneralizedBlochVector or numpy.ndarray
            Hamiltonian specification. If an ``ndarray`` is supplied it must be
            a square Hermitian matrix; if a raw ``GeneralizedBlochVector`` is
            supplied it must represent a Hermitian operator (checked during
            ``GeneralizedBlochHamiltonian`` construction).

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Evolution matrix implementing ``r -> L r``.

        Raises
        ------
        TypeError
            If ``Ham`` is not one of the accepted types.
        ValueError
            If Hermiticity validation fails when constructing the Hamiltonian.

        Notes
        -----
        Using ``[T_a, T_b] = \sum_c f_{ab}^{\;c} T_c`` and
        ``[H, T_b] = \sum_c h_c f_{c b}^{\;d} T_d`` gives

        .. math:: (L_H)_{ab} = -i\, \mathrm{Tr}\big(T_a^{\dagger}[H, T_b]\big)
              = -i \sum_c h_c f_{a b}^{\;c}.

        Structure constants are retrieved through ``structure_const``; phase
        factors use ``T_q^{(k)\dagger} = (-1)^q T_{-q}^{(k)}``.

        """
        # Canonicalize to a validated GeneralizedBlochHamiltonian
        if isinstance(Ham, GeneralizedBlochHamiltonian):
            H = Ham
        elif isinstance(Ham, GeneralizedBlochVector):
            H = GeneralizedBlochHamiltonian(Ham.data)
        elif isinstance(Ham, np.ndarray):
            H = GeneralizedBlochHamiltonian.from_matrix(Ham)
        else:
            raise TypeError(
                "Ham must be ndarray, GeneralizedBlochVector, or GeneralizedBlochHamiltonian."
            )

        d_hilbert = bloch_dim()
        D = d_hilbert * d_hilbert
        mat = np.zeros((D, D), dtype=complex)
        # Map indices to (k,q)
        kq = [_kq_from_idx(n) for n in range(D)]
        # Compute each matrix element via c-vector of [T_a, T_b]
        for a in range(D):
            ka, qa = kq[a]
            for b in range(D):
                kb, qb = kq[b]
                phase = (-1) ** qb
                # c-vector of structure constants f_{ab}^c
                c_vec = structure_const((ka, qa), (kb, -qb))
                val = -1j * phase * bloch_inner_product(c_vec, H)
                # Ensure scalar
                if issparse(val):
                    val = val.toarray().item()
                elif isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                mat[a, b] = val
        return GeneralizedBlochEvolutionMatrix(csr_matrix(mat, dtype=np.complex128))

    @staticmethod
    def from_Lindblad(
        K: Any, *, atol: float = 1e-12
    ) -> "GeneralizedBlochEvolutionMatrix":
        r"""Construct the dissipative Liouvillian row-wise from a Lindblad operator.

        Builds the matrix ``L_K`` corresponding to the superoperator
        ``\mathcal{L}_K[\rho] = K\rho K^{\dagger} - \tfrac12\{K^{\dagger}K,\rho\}``.
        In Bloch coordinates, rows are

        .. math:: (L_K)_{ab} = \mathrm{Tr}\!\left[T_a^{\dagger}\!\left(K T_b K^{\dagger}
            - \tfrac12 T_b K^{\dagger}K - \tfrac12 K^{\dagger}K T_b\right)\right],

        which we assemble via the adjoint map as

        .. math:: k_a = \mathrm{Bloch}\!\left(K^{\dagger} T_a K
            - \tfrac12 K^{\dagger}K\, T_a - \tfrac12 T_a \, K^{\dagger}K\right),\quad
            L[a,:] = \overline{k_a}^T.

        Parameters
        ----------
        K : numpy.ndarray or GeneralizedBlochVector
            Lindblad jump operator. If a matrix is supplied it is converted to
            Bloch coordinates via ``GeneralizedBlochVector.from_matrix``.
        atol : float, optional
            Numerical tolerance used internally when needed (default ``1e-12``).

        Returns
        -------
        GeneralizedBlochEvolutionMatrix
            Evolution matrix implementing the single-operator dissipator.

        Notes
        -----
        Row construction mirrors ``from_Kraus`` in ``observable.py`` but applies
        the anti-commutator subtraction. All products are computed via the
        cached Bloch-space tensor product.

        """
        # Canonicalize K into a Bloch vector
        if isinstance(K, GeneralizedBlochVector):
            rK = K
        else:
            rK = GeneralizedBlochVector.from_matrix(np.asarray(K))
        rK_dag = bloch_hermitian_transpose(rK)
        rMM = rK_dag * rK  # Bloch(K^\dagger K)

        d = bloch_dim()
        D = d * d
        L = np.zeros((D, D), dtype=np.complex128)

        # Unit vectors e_a for each basis index a
        for a in range(D):
            e_a = np.zeros((D,), dtype=complex)
            e_a[a] = 1.0
            rTa = GeneralizedBlochVector(e_a)
            # K^\dagger T_a K
            jump_term = rK_dag * rTa * rK
            # (1/2) K^\dagger K T_a and (1/2) T_a K^\dagger K
            left = rMM * rTa
            right = rTa * rMM
            k_a = jump_term - 0.5 * (left + right)
            if issparse(k_a.data):
                L[a, :] = np.conj(k_a.data.toarray().flatten())
            else:
                L[a, :] = np.conj(k_a.data)

        return GeneralizedBlochEvolutionMatrix(csr_matrix(L))


@dataclass
class GeneralizedBlochState(GeneralizedBlochVector):
    """Density operator represented as a Bloch vector.

    Stores Bloch coefficients ``r_a`` of a Hermitian density matrix ``ρ``.
    Hermiticity is verified on construction; normalization can be enforced via
    ``normalization``.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional complex array of length ``d**2`` containing Bloch
        coefficients of the density operator.

    Raises
    ------
    ValueError
        If Hermiticity validation fails.

    Notes
    -----
    The trace condition ``Tr(ρ)=1`` corresponds (in the chosen COBITO
    convention) to a fixed value of the ``r_{00}`` component; see
    ``normalization`` for enforcement.

    """

    def __post_init__(self) -> None:
        """Initialize GeneralizedBlochState after dataclass construction.

        Verifies Hermiticity of the density operator.

        """
        super().__post_init__()
        # Hermiticity check
        s = bloch_hermitian_transpose(self)

        d1 = self.data
        d2 = s.data

        if issparse(d1):
            d1 = d1.toarray()
        if issparse(d2):
            d2 = d2.toarray()

        if not np.allclose(d1, d2, rtol=1e-10, atol=1e-10):
            raise ValueError("Density Matrix must be Hermitian in operator space.")

    def normalization(self) -> complex | np.ndarray:
        r"""Normalize the Bloch state so that ``Tr(ρ)=1``.

        Ensures the density operator trace equals unity by rescaling the Bloch
        vector with a complex factor determined from ``r_{00}``.

        Parameters
        ----------
        None

        Returns
        -------
        complex or numpy.ndarray
            Scaling factor applied in-place to ``self.data``.

        Raises
        ------
        ValueError
            If ``r_{00}`` is zero (normalization impossible).

        Notes
        -----
        With the COBITO choice ``T_0^{(0)} = I/\sqrt{d}``, a normalized state
        satisfies ``r_{00} = 1/\sqrt{d}``.

        """
        d = bloch_dim()
        target = 1.0 / np.sqrt(d)

        if issparse(self.data):
            # Sparse batch normalization
            # r00 is the first row (index 0)
            r00 = (
                self.data[0, :].toarray().flatten()
            )  # Convert just the first row to dense for calculation

            # Avoid division by zero
            scale = np.zeros_like(r00)
            mask = r00 != 0
            scale[mask] = target / r00[mask]

            # Apply scaling: self.data = self.data * diag(scale)
            # self.data is (d^2, N), scale is (N,)
            # We want to multiply each column j by scale[j].
            # This is equivalent to right-multiplying by a diagonal matrix.
            from scipy.sparse import diags

            S = diags(scale)
            self.data = self.data @ S
            return scale

        if self.data.ndim == 1:
            r00 = self.data[0]
            if r00 == 0:
                raise ValueError("Cannot normalize: r_{00} is zero.")
            scale = target / r00
            self.data *= scale
            return scale
        else:
            # Batch normalization (dense)
            r00 = self.data[0, :]
            scale = np.zeros_like(r00)
            mask = r00 != 0
            scale[mask] = target / r00[mask]
            self.data *= scale[np.newaxis, :]
            return scale

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "GeneralizedBlochState":
        """Create a Bloch-state from an operator matrix.

        Parameters
        ----------
        mat : numpy.ndarray
            Square complex Hermitian matrix representing a density operator or
            general Hermitian operator (will be interpreted as a state).

        Returns
        -------
        GeneralizedBlochState
            Bloch-space state with Hermiticity validated.

        Raises
        ------
        ValueError
            If ``mat`` fails Hermiticity validation.

        """
        gbvec = GeneralizedBlochVector.from_matrix(mat)
        return GeneralizedBlochState(gbvec.data)

    def evolve(self, L: GeneralizedBlochEvolutionMatrix, time: float) -> None:
        """Evolve the Bloch state in place: ``r ← exp(L t) r``.

        Applies Bloch-space dynamics (unitary and/or dissipative) via a
        numerically stable exponential–vector product. Prefers
        ``scipy.sparse.linalg.expm_multiply``; for very small systems it uses a
        cached dense matrix exponential for speed.

        Parameters
        ----------
        L : GeneralizedBlochEvolutionMatrix
            Evolution (Liouvillian) matrix acting on the state.
        time : float
            Real evolution time ``t``.

        Returns
        -------
        None
            The state is modified in place.

        Raises
        ------
        TypeError
            If ``L`` is not a ``GeneralizedBlochEvolutionMatrix`` or ``time``
            is not a scalar.

        Notes
        -----
                - The eigen-decomposition approach can be inaccurate/unstable for
                    non-normal generators due to ill-conditioned eigenvectors. Using
                    ``expm_multiply`` (Padé with scaling/squaring / Krylov) improves
                    robustness with negligible overhead for the small sizes used here.

        """
        if not isinstance(L, GeneralizedBlochEvolutionMatrix):
            raise TypeError("L must be a GeneralizedBlochEvolutionMatrix")
        if not np.isscalar(time):
            raise TypeError("time must be a scalar")
        t = float(time)
        # For small systems, use cached dense expm(t*L) for speed; else Krylov
        n = L.data.shape[0]
        if n <= 16:
            mat = L._dense_expm_cached(t)
            self.data = mat @ self.data
        else:
            # Use SciPy's stable expm_multiply (required dependency)
            # expm_multiply supports B as a matrix (columns are vectors)
            self.data = expm_multiply(L.data * t, self.data)
        # Light Hermiticity symmetrization to curb roundoff drift
        try:
            s = bloch_hermitian_transpose(self)
            self.data = 0.5 * (self.data + s.data)
        except Exception:
            pass

    # Restrict scaling to real scalars only; disallow scalar/vector reversed
    # division
    def __mul__(self, other):
        """Multiply by real scalar.

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochState
            Scaled state.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(self.data * float(other))
        return NotImplemented

    def __rmul__(self, other):
        """Right-side multiplication by real scalar (commutative).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochState
            Scaled state.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(float(other) * self.data)
        return NotImplemented

    def __truediv__(self, other):
        """Divide by real scalar.

        Parameters
        ----------
        other : scalar
            Real scalar divisor.

        Returns
        -------
        GeneralizedBlochState
            Scaled state.

        """
        if np.isscalar(other) and np.isrealobj(other):
            return self._wrap_result(self.data / float(other))
        return NotImplemented

    def __imul__(self, other):
        """In-place multiplication by real scalar (modifies self).

        Parameters
        ----------
        other : scalar
            Real scalar multiplier.

        Returns
        -------
        GeneralizedBlochState
            Self (modified in-place).

        """
        if np.isscalar(other) and np.isrealobj(other):
            self.data *= float(other)
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """In-place division by real scalar (modifies self).

        Parameters
        ----------
        other : scalar
            Real scalar divisor.

        Returns
        -------
        GeneralizedBlochState
            Self (modified in-place).

        """
        if np.isscalar(other) and np.isrealobj(other):
            self.data /= float(other)
            return self
        return NotImplemented
