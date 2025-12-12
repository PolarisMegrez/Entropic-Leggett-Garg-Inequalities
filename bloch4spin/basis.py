r"""
bloch4spin: COBITO Bloch Basis and Operations for Spin-j Systems
----------------------------------------------------------------
Provides utilities for the Canonical Orthonormal Basis of Irreducible Tensor
Operators (COBITO) T_q^{(k)} for Hilbert space dimension d=2j+1. Includes
efficient Bloch-space vector wrappers, basis construction, tensor product
expansion, and commutator structure constants using Wigner symbols.

Public API
----------
- ``bloch_init``, ``bloch_dim``, ``bloch_basis``, ``structure_const``
- ``GeneralizedBlochVector``, ``bloch_inner_product``
- ``bloch_hermitian_transpose``

Notes
-----
- Normalization: The scalar component is ``T_{0}^{(0)} = I/\sqrt{d}`` ensuring
    orthonormality under the Hilbert–Schmidt inner product.
- Indexing: Basis coordinates are linearized via ``n = k(k+1) + q`` with
    inverse mapping recovered by minimal ``k`` such that ``k(k+2) \ge n``.
- Tensor product: The expansion uses Wigner 3j/6j symbols (Condon–Shortley
    convention). Reversed factor ordering introduces a phase ``(-1)^{k1+k2+k3}``
    due to 3j symmetry.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union
import numpy as np
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j
from scipy.sparse import issparse, csr_matrix, csc_matrix, spmatrix, coo_matrix, dok_matrix, vstack as sparse_vstack

__all__ = [
    "bloch_init",
    "bloch_dim",
    "bloch_basis",
    "structure_const",
    "GeneralizedBlochVector",
    "bloch_inner_product",
    "bloch_hermitian_transpose",
]

_bloch_dim: int | None = None
"""Current Hilbert space dimension ``d`` (set by :func:`bloch_init`)."""

_bloch_basis: Dict[Tuple[int, int], np.ndarray] = {}
"""Cache of COBITO basis matrices keyed by ``(k, q)``.

For ``k > 0``, only nonnegative ``q`` are stored. Negative ``q`` are generated
on demand via the relation ``T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^T`` under the
adopted convention where COBITO are real matrices.
"""

_bloch_product_coeff: Dict[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray] = {}
"""Compact tensor product decomposition coefficients for COBITO basis ``((k1,q1),(k2,q2))``.

For each pair with ``index(k1,q1) < index(k2,q2)``, stores a contiguous vector
over allowed ``k3`` in ``[max(|k1-k2|, |q1+q2|), min(k1+k2, d-1)]``. Full vectors
are expanded by :func:`get_product_coeff` when needed.
"""

# --------- Indexing helpers ---------
def _ensure_initialized() -> None:
    """Ensure :func:`bloch_init` has been called.

    Raises
    ------
    RuntimeError
        If the module has not been initialized.
    """
    if _bloch_dim is None:
        raise RuntimeError("Bloch basis not initialized. Call bloch_init(d) first.")

def _idx_from_kq(k: int, q: int) -> int:
    """Map ``(k, q)`` to linear index ``n = k*(k+1) + q``."""
    return k * (k + 1) + q

def _kq_from_idx(n: int) -> Tuple[int, int]:
    """Inverse mapping from linear index ``n`` to ``(k, q)``.

    Finds the smallest nonnegative integer ``k`` such that ``k*(k+2) >= n``;
    then returns ``q = n - k*(k+1)``.
    """
    k = 0
    while k * (k + 2) < n:
        k += 1
    q = n - k * (k + 1)
    return k, q

# --------- COBITO construction ---------
def _generate_tensor_basis(d: int, k: int, q: int,
                           w3_cache: Dict[Tuple[int, int, int, int, int, int], float]) -> np.ndarray:
    """Construct real COBITO matrix ``T_q^(k)`` for dimension ``d``.

    Parameters
    ----------
    d : int
        Hilbert space dimension, with ``d = 2j + 1``.
    k, q : int
        Tensor rank and component, with ``0 <= k <= d-1`` and ``-k <= q <= k``.

    Returns
    -------
    numpy.ndarray
        The real matrix representation of ``T_q^(k)``.
    """
    ITO = np.zeros((d, d), dtype=float)
    pref = np.sqrt(2 * k + 1)
    if q >= 0:
        m1_start, m1_end = q, d
    else:
        m1_start, m1_end = 0, d + q
    # Precomputed 3j cache key uses doubled integers: (K3,K2,K1,-Q3,Q2,Q1)
    # For basis generation: k3 = jR (use -1),     k2 = k, k1 = jR (use -1);
    #                       q3 = m2 (use m2_idx), q2 = q, q1 = m1 (use m1_idx).
    jR = Rational(d - 1, 2)
    for m1_idx in range(m1_start, m1_end):
        m1 = jR - m1_idx
        m2 = m1 + q
        three_j = float(wigner_3j(jR, k, jR, -m2, q, m1))
        phase = (-1) ** (m1_idx - q)
        m2_idx = m1_idx - q
        ITO[m2_idx, m1_idx] += phase * pref * three_j
    return ITO                          # real matrix

def _derive_tensor_product_coeff(d: int, k1: int, q1: int, k2: int, q2: int, k3: int, q3: int,
                                 w6_cache: Dict[Tuple[int, int, int], float],
                                 w3_cache: Dict[Tuple[int, int, int, int, int, int], float]) -> float:
    r"""Return a single tensor-product coefficient.

    Coefficient ``c_{k1 k2 k3}^{q1 q2 q3}`` in the expansion
    ``T_{q1}^{(k1)} T_{q2}^{(k2)} = \sum_{k3,q3} c_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Notes
    -----
    Condon–Shortley convention:

    ``c = \sqrt{(2k1+1)(2k2+1)(2k3+1)} * (-1)^(2j+q3) * {k3 k2 k1; j j j} * (k3 k2 k1; -q3 q2 q1)``,
    with ``j=(d-1)/2``, where ``{...}`` is the Wigner 6j and ``(...)`` the Wigner 3j symbol.
    """
    pref = np.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1))
    phase = (-1) ** (d - 1 + q3)
    # 3j original argument order: (k3, k2, k1; -q3, q2, q1)
    j_trip = [k3, k2, k1]
    m_trip = [-q3, q2, q1]
    # Canonicalize: sort j descending; apply same permutation to m; compute parity
    idx = sorted(range(3), key=lambda i: j_trip[i], reverse=True)
    j_sorted = [j_trip[i] for i in idx]
    m_sorted = [m_trip[i] for i in idx]
    # Count inversions for permutation parity
    swaps = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if idx[i] > idx[j]:
                swaps += 1
    three_j = w3_cache.get((j_sorted[0], j_sorted[1], j_sorted[2],
                            m_sorted[0], m_sorted[1], m_sorted[2]))
    six_j = w6_cache[(j_sorted[0], j_sorted[1], j_sorted[2])]
    if swaps % 2 == 1:
        # Odd permutation: multiply by (-1)^{k1+k2+k3} for 3j only
        three_j *= (-1) ** (k1 + k2 + k3)
    return float(pref * phase * six_j * three_j)

def _init_tensor_basis(d: int,
                       w3_cache: Dict[Tuple[int, int, int, int, int, int], float]) -> None:
    """Populate basis cache for dimension ``d``."""
    global _bloch_basis
    _bloch_basis.clear()
    # Build and store only q>=0 for k>0
    for k in range(1, d):               # since k runs 0...2j (integer)
        for q in range(0, k + 1):
            _bloch_basis[(k, q)] = _generate_tensor_basis(d, k, q, w3_cache)

def _init_tensor_product_coeff(d: int,
                               w6_cache: Dict[Tuple[int, int, int], float],
                               w3_cache: Dict[Tuple[int, int, int, int, int, int], float]) -> None:
    """Populate product coefficient cache (optimized bounds version).

    Constrains ``q2`` so the linear index ordering ``n(k1,q1) <= n(k2,q2)`` holds,
    avoiding post-filtering and cutting roughly half the pair iterations for large ``d``.

    See Also
    --------
    _init_tensor_product_coeff_old : Simpler reference implementation (slower).
    """
    global  _bloch_product_coeff
    _bloch_product_coeff.clear()
    # Precompute product coefficients for ordered pairs; store compact k3-ranges.
    for k1 in range(1, d):
        for q1 in range(-k1, k1 + 1):
            for k2 in range(1, d):
                q2min = max(-k2, 1 - d - q1, k1 * (k1 + 1) + q1 - k2 * (k2 + 1))  # n(k1, q1) <= n(k2, q2)
                q2max = min(k2, d - 1 - q1)
                for q2 in range(q2min, q2max + 1):
                    q3 = q1 + q2
                    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
                    coeffs = []
                    for k3 in range(k3_min, k3_max + 1):
                        c_val = _derive_tensor_product_coeff(d, k1, q1, k2, q2, k3, q3, w6_cache, w3_cache)
                        coeffs.append(c_val)
                    _bloch_product_coeff[((k1, q1), (k2, q2))] = np.array(coeffs, dtype=float)

def _init_tensor_product_coeff_old(d: int) -> None:
    """Legacy product coefficient population (reference / slower).

    Enumerates full ``q2`` range then discards pairs violating ordering. Retained
    only for clarity and comparison; not invoked by ``bloch_init``.
    """
    global  _bloch_product_coeff
    _bloch_product_coeff.clear()
    # Precompute commutator coefficients for ordered pairs; store compact k3-ranges.
    for k1 in range(1, d):
        for q1 in range(-k1, k1 + 1):
            n1 = _idx_from_kq(k1, q1)
            for k2 in range(1, d):
                for q2 in range(-k2, k2 + 1):
                    n2 = _idx_from_kq(k2, q2)
                    if n1 > n2:
                        continue
                    q3 = q1 + q2
                    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
                    if abs(q3) > d - 1:
                        continue
                    # allocate vector over k3 range; we will map later when expanding
                    coeffs = []
                    for k3 in range(k3_min, k3_max + 1):
                        c_val = _derive_tensor_product_coeff(d, k1, q1, k2, q2, k3, q3)
                        coeffs.append(c_val)
                    _bloch_product_coeff[((k1, q1), (k2, q2))] = np.array(coeffs, dtype=float)

# --------- Core vector class and operations ---------
@dataclass
class GeneralizedBlochVector:
    """Bloch-space vector wrapper over a 1D complex array of length ``d**2``.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.spmatrix
        One-dimensional complex array of length ``d**2``, or a sparse matrix
        of shape ``(d**2, batch_size)``.

    Notes
    -----
    - ``data`` is a mutable view; direct mutation can improve performance but use with care.
    - Supports elementwise arithmetic and scalar broadcasting via operator overloads.
    - Use :meth:`to_matrix` / :meth:`from_matrix` to convert between operators and Bloch vectors.
    """
    data: Union[np.ndarray, spmatrix]  # shape ((2j+1)^2,), complex

    # Prefer our arithmetic when mixed with NumPy arrays
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        """Validate shape and dtype of the underlying array."""
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        expected = d * d
        
        if issparse(self.data):
            if self.data.shape[0] != expected:
                raise ValueError(f"Expected sparse vector with {expected} rows, got {self.data.shape}.")
            # Sparse matrices handle dtype differently, usually fixed at creation.
            # We can check if it's complex.
            if not np.issubdtype(self.data.dtype, np.complexfloating):
                 self.data = self.data.astype(complex)
        else:
            # Ensure dense data is ndarray, not matrix
            if isinstance(self.data, np.matrix):
                self.data = np.asarray(self.data)
                
            if self.data.ndim == 1:
                if self.data.shape != (expected,):
                    raise ValueError(f"Expected vector of length {expected}, got {self.data.shape}.")
                # Enforce column vector shape (N, 1) for consistency with sparse
                self.data = self.data.reshape(-1, 1)
            elif self.data.ndim == 2:
                if self.data.shape[0] != expected:
                    raise ValueError(f"Expected vector with {expected} rows, got {self.data.shape}.")
            else:
                raise ValueError(f"Expected 1D or 2D array, got {self.data.ndim}D.")

            if self.data.dtype != complex:
                self.data = self.data.astype(complex)

    # ---- NumPy interop ----
    def __array__(self, dtype=None) -> np.ndarray:
        """Return the underlying data as a NumPy array (optionally cast)."""
        if issparse(self.data):
            arr = self.data.toarray()
        else:
            arr = self.data
        return arr.astype(dtype) if dtype is not None else arr

    # ---- Helpers ----
    @staticmethod
    def _is_scalar(x) -> bool:
        """Return True if ``x`` is a scalar (Python or NumPy scalar)."""
        return np.isscalar(x) or isinstance(x, (np.generic,))

    @staticmethod
    def _as_1d_array(x) -> Union[np.ndarray, spmatrix, None]:
        """Coerce ``x`` to a 1D array of length ``d**2`` or return ``None``."""
        d = _bloch_dim  # type: ignore
        
        if issparse(x):
            # Accept (d^2, 1) or (1, d^2)
            if x.shape == (d*d, 1) or x.shape == (1, d*d):
                return x
            return None

        try:
            arr = np.asarray(x)
        except Exception:
            return None
        if arr.ndim != 1:
            return None
        if arr.shape[0] != d * d:
            return None
        return arr.reshape(-1, 1)

    @classmethod
    def _wrap_result(cls, arr: Union[np.ndarray, spmatrix]) -> "GeneralizedBlochVector":
        """Wrap a 1D complex array or sparse matrix, preserving subclass type."""
        return cls(arr)

    def copy(self) -> "GeneralizedBlochVector":
        """Return a deep copy, preserving the dynamic subclass type."""
        return self.__class__(self.data.copy())

    def __repr__(self) -> str:
        """Compact string representation with logical shape and dtype."""
        d = _bloch_dim if _bloch_dim is not None else 0
        shape_str = str(self.data.shape) if d else "(0,)"
        return f"GeneralizedBlochVector(shape={shape_str}, dtype=complex)"

    @staticmethod
    def zeros(batch_size: int = 0) -> "GeneralizedBlochVector":
        """Return a zero Bloch vector for the current dimension ``d``.
        
        If batch_size > 0, returns a batch of zero vectors (d^2, batch_size).
        Returns a sparse matrix (CSC) to save memory.
        """
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        # Always return 2D sparse matrix for consistency with sparse backend
        cols = batch_size if batch_size > 0 else 1
        return GeneralizedBlochVector(csc_matrix((d * d, cols), dtype=complex))

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "GeneralizedBlochVector":
        r"""Convert an operator matrix to its Bloch vector.

        Parameters
        ----------
        mat : numpy.ndarray
            Square matrix of shape ``(d, d)``.

        Returns
        -------
        GeneralizedBlochVector
            Bloch coefficients ``r_{kq} = Tr(T_{q}^{(k)\dagger} @ mat)``.

        Raises
        ------
        ValueError
            If the matrix shape does not match current ``d``.
        """
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        if mat.shape != (d, d):
            raise ValueError("Density/operator matrix has incompatible shape.")
        r = np.zeros((d * d, 1), dtype=complex)
        for k in range(0, d):
            for q in range(-k, k + 1):
                T_kq = bloch_basis((k, q))
                r[_idx_from_kq(k, q), 0] = np.trace(T_kq.T @ mat)
        return GeneralizedBlochVector(r)

    def to_matrix(self) -> np.ndarray:
        r"""Reconstruct the operator matrix from Bloch coefficients.

        Returns
        -------
        numpy.ndarray
            Matrix ``A = \sum_{k,q} r_{kq} T_q^(k)`` of shape ``(d, d)``.
        """
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        mat = np.zeros((d, d), dtype=complex)
        for k in range(0, d):
            for q in range(-k, k + 1):
                val = self.data[_idx_from_kq(k, q)]
                # Handle (N, 1) shape indexing
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                mat += val * bloch_basis((k, q))
        return mat

    # ---- Arithmetic operators ----
    def __add__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data + other.data)
        if self._is_scalar(other):
            # Sparse + scalar -> dense (usually, unless scalar is 0)
            # But if scalar is 0, it's no-op.
            if other == 0:
                return self.copy()
            return self._wrap_result(self.data + other)
        arr = self._as_1d_array(other)
        if arr is not None:
            # If self.data is sparse and arr is dense, result is dense
            # If arr is sparse (not handled by _as_1d_array currently), result sparse
            return self._wrap_result(self.data + arr)
        return NotImplemented

    def __radd__(self, other):
        # commutative
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data - other.data)
        if self._is_scalar(other):
            if other == 0:
                return self.copy()
            return self._wrap_result(self.data - other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(self.data - arr)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(other.data - self.data)
        if self._is_scalar(other):
            return self._wrap_result(other - self.data)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(arr - self.data)
        return NotImplemented

    def __mul__(self, other):
        # Scalar multiplication
        if self._is_scalar(other):
            return self._wrap_result(self.data * other)
        # Bloch-space operator product with another vector or array-like (1D, length d^2)
        if isinstance(other, GeneralizedBlochVector):
            return bloch_tensor_product(self, other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return bloch_tensor_product(self, GeneralizedBlochVector(arr.astype(complex)))
        return NotImplemented

    def __rmul__(self, other):
        # Scalar multiplication (commutative)
        if self._is_scalar(other):
            return self._wrap_result(other * self.data)
        # Bloch-space operator product for array-like on the left
        arr = self._as_1d_array(other)
        if arr is not None:
            return bloch_tensor_product(GeneralizedBlochVector(arr.astype(complex)), self)
        return NotImplemented

    def __truediv__(self, other):
        # Only scalar division is supported
        if self._is_scalar(other):
            return self._wrap_result(self.data / other)
        return NotImplemented

    def __neg__(self):
        return self._wrap_result(-self.data)

    # ---- In-place operators (mutating) ----
    def __iadd__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self.data = self.data + other.data
            return self
        if self._is_scalar(other):
            self.data = self.data + other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data = self.data + arr
            return self
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self.data = self.data - other.data
            return self
        if self._is_scalar(other):
            self.data = self.data - other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data = self.data - arr
            return self
        return NotImplemented

    def __imul__(self, other):
        if self._is_scalar(other):
            self.data = self.data * other
            return self
        if isinstance(other, GeneralizedBlochVector):
            prod = bloch_tensor_product(self, other)
            self.data = prod.data
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            prod = bloch_tensor_product(self, GeneralizedBlochVector(arr.astype(complex)))
            self.data = prod.data
            return self
        return NotImplemented

    def __itruediv__(self, other):
        # Only scalar division is supported
        if self._is_scalar(other):
            self.data = self.data / other
            return self
        return NotImplemented

# --------- Public API ---------
def bloch_init(d: int) -> None:
    """Initialize global caches for dimension ``d = 2j + 1``.

    Parameters
    ----------
    d : int
        Hilbert space dimension. Must be a positive integer.

    Raises
    ------
    ValueError
        If ``d <= 0`` or not an integer.
    """
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer.")
    global _bloch_dim
    # If we've already initialized for this dimension and caches exist, skip
    if _bloch_dim == d and _bloch_basis and _bloch_product_coeff:
        return
    _bloch_dim = d
    # --- Build ephemeral Wigner caches (doubled-integer indexing) ---
    jR = Rational(d - 1, 2)
    w6_cache: Dict[Tuple[int, int, int], float] = {}
    w3_cache: Dict[Tuple[int, int, int, int, int, int], float] = {}
    # 6j: iterate k1,k2,k3 with triangle |k1-k2|<=k3<=k1+k2, 0<=k*<=2j (i.e. <= J2)
    for k1 in range(0, d):
        for k2 in range(k1, d):
            k3_max = min(k1 + k2, d - 1)
            for k3 in range(k2, k3_max + 1):
                val = float(wigner_6j(k3, k2, k1, jR, jR, jR))
                w6_cache[(k3, k2, k1)] = val
    # 3j: iterate k1,k2,k3 triangle; q1,q2 and q3=q1+q2 bounds
    for k1 in range(0, d):
        for q1 in range(-k1, k1 + 1):
            for k2 in range(k1, d):
                for q2 in range(-k2, k2 + 1):
                    q3 = q1 + q2
                    k3_max = min(k1 + k2, d - 1)
                    for k3 in range(k2, k3_max + 1):
                        val = float(wigner_3j(k3, k2, k1, -q3, q2, q1))
                        w3_cache[(k3, k2, k1, -q3, q2, q1)] = val
    # --- Populate basis and product coefficient caches using precomputed tables ---
    _init_tensor_basis(d, w3_cache)
    _init_tensor_product_coeff(d, w6_cache, w3_cache)
    # --- Drop ephemeral Wigner caches to free memory ---
    w6_cache.clear()
    w3_cache.clear()

def bloch_dim() -> int:
    """Return the current Hilbert space dimension ``d`` set by ``bloch_init``.

    Returns
    -------
    int
        The dimension ``d``.

    Raises
    ------
    RuntimeError
        If the module has not been initialized via :func:`bloch_init`.
    """
    _ensure_initialized()
    return int(_bloch_dim)  # type: ignore

def bloch_basis(kq: Tuple[int, int]) -> np.ndarray:
    """Return COBITO basis matrix ``T_q^(k)``.

    Parameters
    ----------
    kq : tuple of int
        Pair ``(k, q)`` with ``0 <= k <= d-1`` and ``-k <= q <= k``.

    Returns
    -------
    numpy.ndarray
        The real matrix ``T_q^(k)`` of shape ``(d, d)``.

    Raises
    ------
    ValueError
        If ``(k, q)`` is invalid for the current dimension.

    Notes
    -----
    For ``q < 0``, the matrix is synthesized by ``T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^T``.
    """
    _ensure_initialized()
    k, q = kq
    d = _bloch_dim  # type: ignore
    if k < 0 or k > (d - 1) or q < -k or q > k:
        raise ValueError("Invalid (k,q) for current spin j.")
    if k == 0:
        return np.eye(d, dtype=float) / np.sqrt(d)
    if q >= 0:
        return _bloch_basis[(k, q)]
    # q < 0: derive via transpose relation
    T_pos = _bloch_basis[(k, -q)]
    phase = (-1) ** (q)
    return phase * T_pos.T

def basis_product(kq1: Tuple[int, int], kq2: Tuple[int, int]) -> "GeneralizedBlochVector":
    r"""Return Bloch vector for the product ``T_{q1}^{(k1)} T_{q2}^{(k2)}``.

    Computes ``r = T_{q1}^{(k1)} * T_{q2}^{(k2)}`` in the operator sense.
    Uses the precomputed structure constants.

    Parameters
    ----------
    kq1, kq2 : tuple(int, int)
        Indices ``(k, q)`` for the two basis operators.

    Returns
    -------
    GeneralizedBlochVector
        Bloch vector for the product operator.
    """
    _ensure_initialized()
    d = _bloch_dim   # type: ignore
    k1, q1 = kq1
    k2, q2 = kq2
    if k1 < 0 or k1 > (d - 1) or q1 < -k1 or q1 > k1:
        raise ValueError("Invalid (k1,q1) for current spin j.")
    if k2 < 0 or k2 > (d - 1) or q2 < -k2 or q2 > k2:
        raise ValueError("Invalid (k2,q2) for current spin j.")
    vec = np.zeros((d * d,), dtype=complex)

    # Handle identity cases: T00 = I/sqrt(d) so T00*T = T*T00 = (1/sqrt(d)) T
    if k1 == 0 or k2 == 0:
        n = _idx_from_kq(k1 + k2, q1 + q2)
        # Return sparse vector
        val = 1.0 / np.sqrt(d)
        sp_vec = coo_matrix(([val], ([n], [0])), shape=(d*d, 1), dtype=complex).tocsc()
        return GeneralizedBlochVector(sp_vec)
    
    q3 = q1 + q2
    n1, n2 = _idx_from_kq(k1, q1), _idx_from_kq(k2, q2)
    # Determine sign/order
    if n1 > n2:
        coeffs = _bloch_product_coeff.get(((k2, q2), (k1, q1)))
    else:
        coeffs = _bloch_product_coeff.get(((k1, q1), (k2, q2)))
    if coeffs is None:
        return GeneralizedBlochVector.zeros()
    
    # Expand to full vector; only k3 in allowed range, q3=q1+q2
    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
    
    # Construct sparse vector directly
    rows = []
    data = []
    
    for idx, k3 in enumerate(range(k3_min, k3_max + 1)):
        n = _idx_from_kq(k3, q3)
        if n1 > n2:
            sign = (-1) ** (k1 + k2 + k3)
        else:
            sign = 1.0
        val = sign * coeffs[idx]
        if val != 0:
            rows.append(n)
            data.append(val)
            
    if not rows:
        return GeneralizedBlochVector.zeros()
        
    # Create sparse column vector (d^2, 1)
    # Use COO for construction
    cols = np.zeros(len(rows), dtype=int)
    sp_vec = coo_matrix((data, (rows, cols)), shape=(d*d, 1), dtype=complex).tocsc()
    
    return GeneralizedBlochVector(sp_vec)

def structure_const(kq1: Tuple[int, int], kq2: Tuple[int, int]) -> "GeneralizedBlochVector":
    r"""Return Bloch vector of commutator coefficients.

    Uses ``[A,B] = A B - B A`` with each product expanded by :func:`basis_product`.
    Provides entries ``f_{k1 k2 k3}^{q1 q2 q3}`` in
    ``[T_{q1}^{(k1)}, T_{q2}^{(k2)}] = \sum_{k3,q3} f_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Parameters
    ----------
    kq1, kq2 : tuple(int, int)
        Indices ``(k, q)`` for the two basis operators.

    Returns
    -------
    GeneralizedBlochVector
        Coefficients of the commutator.
    """
    _ensure_initialized()
    return basis_product(kq1, kq2) - basis_product(kq2, kq1)

def bloch_inner_product(u: GeneralizedBlochVector, v: GeneralizedBlochVector) -> Union[complex, np.ndarray]:
    """Complex inner product ``<u, v>`` between two Bloch vectors.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors of length ``d**2``. Can be batched (2D arrays).

    Returns
    -------
    complex or numpy.ndarray
        The value ``np.vdot(u.data, v.data)`` for 1D inputs.
        For batched inputs, returns an array of inner products (element-wise along batch dimension).
        If shapes allow broadcasting (e.g. 1D vs 2D), broadcasting is applied.
    """
    d1 = u.data
    d2 = v.data
    
    # Handle sparse matrices
    is_sparse_1 = issparse(d1)
    is_sparse_2 = issparse(d2)
    
    if is_sparse_1 or is_sparse_2:
        # Ensure both are sparse for efficient operation, or handle mixed
        # scipy.sparse multiply is elementwise
        # We want sum(conj(d1) * d2, axis=0)
        
        # Convert 1D arrays to column vectors for consistency if mixed
        if not is_sparse_1 and d1.ndim == 1:
            d1 = d1[:, np.newaxis]
        if not is_sparse_2 and d2.ndim == 1:
            d2 = d2[:, np.newaxis]
            
        # Conjugate
        d1_conj = d1.conj()
        
        # Elementwise multiplication
        # If both are sparse, .multiply() returns sparse
        # If one is dense, * usually returns dense (numpy broadcasting)
        if is_sparse_1 and is_sparse_2:
            prod = d1_conj.multiply(d2)
        elif is_sparse_1:
            # sparse.multiply(dense) -> sparse (usually) or dense?
            # CSR multiply dense -> dense usually
            prod = d1_conj.multiply(d2) 
        else:
            # dense * sparse -> dense
            prod = d1_conj * d2
            
        # Sum along axis 0
        res = np.sum(prod, axis=0)
        
        # Result might be matrix (1, N) or (1, 1) if sparse sum used
        if issparse(res):
            res = res.toarray().flatten()
        else:
            res = np.asarray(res).flatten()
            
        if res.size == 1:
            return res.item()
        return res

    # Fast path for simple 1D vectors (dense)
    if d1.ndim == 1 and d2.ndim == 1:
        return np.vdot(d1, d2)
        
    # Handle batching/broadcasting (dense)
    # Ensure at least 2D for consistent axis logic: (features, batch)
    if d1.ndim == 1:
        d1 = d1[:, np.newaxis]
    if d2.ndim == 1:
        d2 = d2[:, np.newaxis]
        
    # Check feature dimension match
    if d1.shape[0] != d2.shape[0]:
        raise ValueError(f"Feature dimension mismatch: {d1.shape[0]} vs {d2.shape[0]}")
        
    # Compute element-wise inner product along axis 0 (features)
    # <u, v> = sum(conj(u_i) * v_i)
    res = np.sum(np.conj(d1) * d2, axis=0)
    
    # If result is effectively a scalar (1-element array), return scalar?
    # np.vdot returns scalar. To be consistent:
    if res.size == 1:
        return res.item()
        
    return res

def bloch_tensor_product(u: GeneralizedBlochVector,
                     v: GeneralizedBlochVector,
                     tol: float = 0.0) -> GeneralizedBlochVector:
    """Compute operator product ``A * B`` in Bloch space.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors representing operators ``A`` and ``B``.
        Must be 1D (single operators), batching is not currently supported.
    tol : float, optional
        Components with absolute value ``<= tol`` are treated as zero.

    Returns
    -------
    GeneralizedBlochVector
        Bloch vector of the product ``A*B``.

    Notes
    -----
    Accumulates over nonzero coordinates only. Rough complexity ``O(p*q)`` for
    ``p,q`` nonzero terms. Coefficients drawn from :func:`basis_product` cache.
    """
    _ensure_initialized()
    
    if u.data.ndim > 1 and u.data.shape[1] > 1:
         raise NotImplementedError("bloch_tensor_product does not support batched vectors.")
    if v.data.ndim > 1 and v.data.shape[1] > 1:
         raise NotImplementedError("bloch_tensor_product does not support batched vectors.")

    # Extract non-zero elements efficiently
    if issparse(u.data):
        u_coo = u.data.tocoo()
        # Filter by tol
        mask = np.abs(u_coo.data) > tol
        idx_r = u_coo.row[mask]
        ru = u_coo.data[mask]
    else:
        idx_r = np.flatnonzero(np.abs(u.data) > tol)
        ru = u.data.ravel()[idx_r]

    if issparse(v.data):
        v_coo = v.data.tocoo()
        mask = np.abs(v_coo.data) > tol
        idx_s = v_coo.row[mask]
        sv = v_coo.data[mask]
    else:
        idx_s = np.flatnonzero(np.abs(v.data) > tol)
        sv = v.data.ravel()[idx_s]

    if idx_r.size == 0 or idx_s.size == 0:
        return GeneralizedBlochVector.zeros()

    kq_r = [_kq_from_idx(int(n)) for n in idx_r]
    kq_s = [_kq_from_idx(int(n)) for n in idx_s]

    # Cache product coefficient vectors to avoid repeated expansion
    bloch_vec_cache: dict[tuple[int,int,int,int], Union[np.ndarray, spmatrix]] = {}
    bloch_vecs = []
    weights = []

    for i1, (k1, q1) in enumerate(kq_r):
        r = ru[i1]
        for i2, (k2, q2) in enumerate(kq_s):
            s = sv[i2]
            key = (k1, q1, k2, q2)
            # Get or compute the bloch vector
            bloch_vec = bloch_vec_cache.get(key)
            if bloch_vec is None:
                # basis_product now returns GeneralizedBlochVector wrapping sparse
                bloch_vec = basis_product((k1, q1), (k2, q2)).data
                bloch_vec_cache[key] = bloch_vec
            
            # Check if zero (sparse or dense)
            is_nonzero = False
            if issparse(bloch_vec):
                if bloch_vec.nnz > 0:
                    is_nonzero = True
            elif np.any(bloch_vec):
                is_nonzero = True
                
            if is_nonzero:
                bloch_vecs.append(bloch_vec)
                weights.append(r * s)

    if not weights:  # all products zero
        return GeneralizedBlochVector.zeros()

    # Stack vectors
    # bloch_vecs contains a mix of sparse and dense? 
    # basis_product returns sparse now.
    # If we have sparse vectors, use sparse_vstack (which stacks vertically, i.e. rows)
    # But we want to sum: sum(weight_i * vec_i)
    # This is equivalent to: Matrix * Weights
    # If we stack vectors as columns: (D, P) @ (P, 1) -> (D, 1)
    # sparse_hstack stacks columns.
    
    from scipy.sparse import hstack as sparse_hstack
    
    # Ensure all are sparse for efficient stacking
    # basis_product returns sparse, so they should be sparse.
    
    # Check first element
    if issparse(bloch_vecs[0]):
        coeff_mat = sparse_hstack(bloch_vecs) # (D, P)
        weights_vec = csc_matrix(np.asarray(weights)[:, np.newaxis]) # (P, 1)
        out = coeff_mat @ weights_vec # (D, 1) sparse
        return GeneralizedBlochVector(out)
    else:
        # Fallback for dense
        coeff_mat = np.vstack(bloch_vecs).T # (D, P) - vstack stacks as rows, so transpose
        # Wait, original code:
        # coeff_mat = np.vstack(bloch_vecs)      # shape: (P, D)
        # weights_vec = np.asarray(weights)      # shape: (P,)
        # out = weights_vec @ coeff_mat          # (D,)
        # This was summing rows weighted by weights.
        
        # If we use sparse_hstack, we get (D, P).
        # We want sum(w_i * col_i).
        # This is Mat @ w.
        
        coeff_mat = np.vstack(bloch_vecs) # (P, D)
        weights_vec = np.asarray(weights) # (P,)
        out = weights_vec @ coeff_mat # (D,)
        return GeneralizedBlochVector(out)

def bloch_commutator(u: GeneralizedBlochVector,
                     v: GeneralizedBlochVector,
                     tol: float = 0.0) -> GeneralizedBlochVector:
    """Compute the Bloch-space commutator ``[A, B]``.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors representing operators ``A`` and ``B``.
        Must be 1D (single operators), batching is not currently supported.
    tol : float, optional
        Components with absolute value ``<= tol`` are treated as zero.

    Returns
    -------
    GeneralizedBlochVector
        Bloch vector of the commutator ``[A, B]``.

    Notes
    -----
    Uses precomputed compact structure constants via :func:`structure_const` and
    vectorizes the accumulation over nonzero components.
    """
    _ensure_initialized()
    
    if u.data.ndim > 1 and u.data.shape[1] > 1:
        raise NotImplementedError("bloch_commutator does not support batched vectors.")

    # Extract non-zero elements efficiently
    if issparse(u.data):
        u_coo = u.data.tocoo()
        mask = np.abs(u_coo.data) > tol
        idx_u = u_coo.row[mask]
        ru = u_coo.data[mask]
    else:
        idx_u = np.flatnonzero(np.abs(u.data) > tol)
        ru = u.data[idx_u]

    if issparse(v.data):
        v_coo = v.data.tocoo()
        mask = np.abs(v_coo.data) > tol
        idx_v = v_coo.row[mask]
        sv = v_coo.data[mask]
    else:
        idx_v = np.flatnonzero(np.abs(v.data) > tol)
        sv = v.data[idx_v]

    if idx_u.size == 0 or idx_v.size == 0:
        return GeneralizedBlochVector.zeros()

    kq_u = [_kq_from_idx(int(n)) for n in idx_u]
    kq_v = [_kq_from_idx(int(n)) for n in idx_v]

    # Cache commutator coefficients to avoid repeated expansion
    coeff_cache: dict[tuple[int,int,int,int], Union[np.ndarray, spmatrix]] = {}
    coeff_rows = []
    weights = []

    for i1, (k1, q1) in enumerate(kq_u):
        r = ru[i1]
        for i2, (k2, q2) in enumerate(kq_v):
            s = sv[i2]
            key = (k1, q1, k2, q2)
            # Get or compute the commutator coefficient vector
            cvec = coeff_cache.get(key)
            if cvec is None:
                cvec = structure_const((k1, q1), (k2, q2)).data
                coeff_cache[key] = cvec
            
            # Skip zero vectors
            is_nonzero = False
            if issparse(cvec):
                if cvec.nnz > 0:
                    is_nonzero = True
            elif np.any(cvec):
                is_nonzero = True
                
            if is_nonzero:
                coeff_rows.append(cvec)
                weights.append(r * s)

    if not weights:  # all commutators zero
        return GeneralizedBlochVector.zeros()

    # Stack vectors
    from scipy.sparse import hstack as sparse_hstack
    
    if issparse(coeff_rows[0]):
        coeff_mat = sparse_hstack(coeff_rows) # (D, P)
        # Convert weights to sparse to ensure sparse result
        weights_vec = csc_matrix(np.asarray(weights)[:, np.newaxis]) # (P, 1)
        out = coeff_mat @ weights_vec # (D, 1) sparse
        return GeneralizedBlochVector(out)
    else:
        coeff_mat = np.vstack(coeff_rows).T      # shape: (D, P)
        weights_vec = np.asarray(weights)      # shape: (P,)
        out = coeff_mat @ weights_vec          # (D,)
        return GeneralizedBlochVector(out)

def bloch_hermitian_transpose(r: GeneralizedBlochVector) -> GeneralizedBlochVector:
    """Hermitian transpose in Bloch space.

    Parameters
    ----------
    r : GeneralizedBlochVector
        Bloch vector with components ``r_{kq}``.

    Returns
    -------
    GeneralizedBlochVector
        The vector ``s`` with components ``s_{kq} = (-1)^q conj(r_{k,-q})``.
    """
    _ensure_initialized()
    d = _bloch_dim  # type: ignore
    
    if issparse(r.data):
        # Sparse implementation
        # We need to permute rows: row for (k, q) <-> row for (k, -q)
        # And apply phases and conjugation.
        # Constructing a permutation matrix might be cleanest but maybe slow?
        # Or just constructing COO data directly.
        
        # For now, convert to dense if small? No, user said "inherently sparse".
        # Let's use row slicing and stacking.
        
        # Optimization: Precompute permutation indices and phases once per dimension?
        # For now, just do the loop.
        
        # We need to construct a new sparse matrix.
        # LIL format is good for construction.
        s = r.data.tolil() # Copy structure
        # But we are swapping rows, so in-place modification of LIL is okay-ish?
        # Actually, we are mapping r -> s.
        
        # Let's use a list of rows and vstack?
        # Or better: compute the permutation vector P and phase vector Ph.
        # s = Ph * (P @ r.data.conj())
        
        # Let's build P and Ph on the fly (or cache them in future).
        rows = []
        cols = []
        vals = []
        phases = []
        
        # Map: old_row -> new_row
        # s_new = phase * conj(r_old)
        # s[new_idx] = phase * conj(r[old_idx])
        # So we want to move data FROM old_idx TO new_idx.
        # P[new, old] = 1.
        
        perm_indices = np.zeros(d*d, dtype=int)
        phase_vec = np.zeros(d*d, dtype=complex)
        
        for k in range(0, d):
            # q=0
            idx0 = _idx_from_kq(k, 0)
            perm_indices[idx0] = idx0
            phase_vec[idx0] = 1.0
            
            for q in range(1, k + 1):
                idx_pos = _idx_from_kq(k, q)
                idx_neg = _idx_from_kq(k, -q)
                phase = (-1) ** q
                
                # s[idx_pos] comes from r[idx_neg]
                perm_indices[idx_pos] = idx_neg
                phase_vec[idx_pos] = phase
                
                # s[idx_neg] comes from r[idx_pos]
                perm_indices[idx_neg] = idx_pos
                phase_vec[idx_neg] = phase
                
        # Apply permutation and phase
        # s = diag(phase) @ r.data[perm_indices, :].conj()
        # Sparse indexing r.data[perm_indices, :] works
        
        s_data = r.data[perm_indices, :].conj()
        # Apply phases (row-wise multiplication)
        # sparse.diags(phase_vec) @ s_data
        from scipy.sparse import diags
        P_mat = diags(phase_vec)
        s_final = P_mat @ s_data
        
        return GeneralizedBlochVector(s_final)

    if r.data.ndim == 1:
        s = np.zeros((d * d,), dtype=complex)
        # q = 0 terms
        for k in range(0, d):
            s[_idx_from_kq(k, 0)] = np.conj(r.data[_idx_from_kq(k, 0)])
        # q != 0 terms
        for k in range(0, d):
            for q in range(1, k + 1):
                phase = (-1) ** q
                s[_idx_from_kq(k, q)] = phase * np.conj(r.data[_idx_from_kq(k, -q)])
                s[_idx_from_kq(k, -q)] = phase * np.conj(r.data[_idx_from_kq(k, q)])
    else:
        # Vectorized for batch dimension
        N = r.data.shape[1]
        s = np.zeros((d * d, N), dtype=complex)
        # q = 0 terms
        for k in range(0, d):
            idx = _idx_from_kq(k, 0)
            s[idx, :] = np.conj(r.data[idx, :])
        # q != 0 terms
        for k in range(0, d):
            for q in range(1, k + 1):
                phase = (-1) ** q
                idx_pos = _idx_from_kq(k, q)
                idx_neg = _idx_from_kq(k, -q)
                s[idx_pos, :] = phase * np.conj(r.data[idx_neg, :])
                s[idx_neg, :] = phase * np.conj(r.data[idx_pos, :])
                
    return GeneralizedBlochVector(s)