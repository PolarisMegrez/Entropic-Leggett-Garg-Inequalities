"""COBITO Bloch basis and operations for spin-j systems.

Provides utilities to work with the Canonical Orthonormal Basis of Irreducible
Tensor Operators (COBITO) ``T_q^{(k)}`` for Hilbert space dimension ``d=2j+1``.
Includes an efficient Bloch-space vector wrapper, basis construction, tensor
product expansion, and commutator structure constants using Wigner symbols.

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
from typing import Dict, Tuple
import numpy as np
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

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
    """Return a single tensor-product coefficient.

    Coefficient ``c_{k1 k2 k3}^{q1 q2 q3}`` in the expansion
    ``T_{q1}^{(k1)} T_{q2}^{(k2)} = \sum_{k3,q3} c_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Notes
    -----
    Condon–Shortley convention:

    ``c = sqrt((2k1+1)(2k2+1)(2k3+1)) * (-1)^(2j+q3) * {k3 k2 k1; j j j} * (k3 k2 k1; -q3 q2 q1)``,
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
    data : numpy.ndarray
        One-dimensional complex array of length ``d**2``.

    Notes
    -----
    - ``data`` is a mutable view; direct mutation can improve performance but use with care.
    - Supports elementwise arithmetic and scalar broadcasting via operator overloads.
    - Use :meth:`to_matrix` / :meth:`from_matrix` to convert between operators and Bloch vectors.
    """
    data: np.ndarray  # shape ((2j+1)^2,), complex

    # Prefer our arithmetic when mixed with NumPy arrays
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        """Validate shape and dtype of the underlying array."""
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        expected = d * d
        if self.data.shape != (expected,):
            raise ValueError(f"Expected vector of length {expected}, got {self.data.shape}.")
        if self.data.dtype != complex:
            self.data = self.data.astype(complex)

    # ---- NumPy interop ----
    def __array__(self, dtype=None) -> np.ndarray:
        """Return the underlying data as a NumPy array (optionally cast)."""
        arr = self.data
        return arr.astype(dtype) if dtype is not None else arr

    # ---- Helpers ----
    @staticmethod
    def _is_scalar(x) -> bool:
        """Return True if ``x`` is a scalar (Python or NumPy scalar)."""
        return np.isscalar(x) or isinstance(x, (np.generic,))

    @staticmethod
    def _as_1d_array(x) -> np.ndarray | None:
        """Coerce ``x`` to a 1D array of length ``d**2`` or return ``None``."""
        try:
            arr = np.asarray(x)
        except Exception:
            return None
        if arr.ndim != 1:
            return None
        d = _bloch_dim  # type: ignore
        if arr.shape[0] != d * d:
            return None
        return arr

    @classmethod
    def _wrap_result(cls, arr: np.ndarray) -> "GeneralizedBlochVector":
        """Wrap a 1D complex array, preserving subclass type (validates length)."""
        return cls(arr)

    def copy(self) -> "GeneralizedBlochVector":
        """Return a deep copy, preserving the dynamic subclass type."""
        return self.__class__(self.data.copy())

    def __repr__(self) -> str:
        """Compact string representation with logical shape and dtype."""
        d = _bloch_dim if _bloch_dim is not None else 0
        return f"GeneralizedBlochVector(shape=({d*d if d else 0},), dtype=complex)"

    @staticmethod
    def zeros() -> "GeneralizedBlochVector":
        """Return a zero Bloch vector for the current dimension ``d``."""
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        return GeneralizedBlochVector(np.zeros((d * d,), dtype=complex))

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "GeneralizedBlochVector":
        """Convert an operator matrix to its Bloch vector.

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
        r = np.zeros((d * d,), dtype=complex)
        for k in range(0, d):
            for q in range(-k, k + 1):
                T_kq = bloch_basis((k, q))
                r[_idx_from_kq(k, q)] = np.trace(T_kq.T @ mat)
        return GeneralizedBlochVector(r)

    def to_matrix(self) -> np.ndarray:
        """Reconstruct the operator matrix from Bloch coefficients.

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
                mat += self.data[_idx_from_kq(k, q)] * bloch_basis((k, q))
        return mat

    # ---- Arithmetic operators ----
    def __add__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data + other.data)
        if self._is_scalar(other):
            return self._wrap_result(self.data + other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(self.data + arr)
        return NotImplemented

    def __radd__(self, other):
        # commutative
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data - other.data)
        if self._is_scalar(other):
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
            self.data += other.data
            return self
        if self._is_scalar(other):
            self.data += other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data += arr
            return self
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self.data -= other.data
            return self
        if self._is_scalar(other):
            self.data -= other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data -= arr
            return self
        return NotImplemented

    def __imul__(self, other):
        if self._is_scalar(other):
            self.data *= other
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
            self.data /= other
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
    """Return Bloch vector for the product ``T_{q1}^{(k1)} T_{q2}^{(k2)}``.

    Implements the expansion
    ``T_{q1}^{(k1)} T_{q2}^{(k2)} = \sum_{k3,q3} c_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Parameters
    ----------
    kq1, kq2 : tuple(int, int)
        Basis labels ``(k, q)``.

    Returns
    -------
    GeneralizedBlochVector
        Length ``d**2`` vector with nonzeros at allowed ``(k3,q3)``.

    Notes
    -----
    - Identity: ``T_{00} = I/\sqrt{d}`` so products with ``T_{00}`` scale by ``1/\sqrt{d}``.
    - Ordering: Reversed order applies phase ``(-1)^{k1+k2+k3}`` from 3j symmetry.
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
        vec[n] = 1.0 / np.sqrt(d)
        return GeneralizedBlochVector(vec)
    
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
    for idx, k3 in enumerate(range(k3_min, k3_max + 1)):
        n = _idx_from_kq(k3, q3)
        if n1 > n2:
            sign = (-1) ** (k1 + k2 + k3)
        else:
            sign = 1.0
        vec[n] = sign * coeffs[idx]
    return GeneralizedBlochVector(vec)

def structure_const(kq1: Tuple[int, int], kq2: Tuple[int, int]) -> "GeneralizedBlochVector":
    """Return Bloch vector of commutator coefficients.

    Uses ``[A,B] = A B - B A`` with each product expanded by :func:`basis_product`.
    Provides entries ``f_{k1 k2 k3}^{q1 q2 q3}`` in
    ``[T_{q1}^{(k1)}, T_{q2}^{(k2)}] = \sum_{k3,q3} f_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.
    """
    _ensure_initialized()
    return basis_product(kq1, kq2) - basis_product(kq2, kq1)

def bloch_inner_product(u: GeneralizedBlochVector, v: GeneralizedBlochVector) -> complex:
    """Complex inner product ``<u, v>`` between two Bloch vectors.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors of length ``d**2``.

    Returns
    -------
    complex
        The value ``np.vdot(u.data, v.data)``.
    """
    return np.vdot(u.data, v.data)

def bloch_tensor_product(u: GeneralizedBlochVector,
                     v: GeneralizedBlochVector,
                     tol: float = 0.0) -> GeneralizedBlochVector:
    """Compute operator product ``A * B`` in Bloch space.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors representing operators ``A`` and ``B``.
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

    idx_r = np.flatnonzero(np.abs(u.data) > tol)   # non-zero indices
    idx_s = np.flatnonzero(np.abs(v.data) > tol)   # non-zero indices
    if idx_r.size == 0 or idx_s.size == 0:
        return GeneralizedBlochVector.zeros()

    ru = u.data[idx_r]
    sv = v.data[idx_s]

    kq_r = [_kq_from_idx(int(n)) for n in idx_r]
    kq_s = [_kq_from_idx(int(n)) for n in idx_s]

    # Cache product coefficient vectors to avoid repeated expansion
    bloch_vec_cache: dict[tuple[int,int,int,int], np.ndarray] = {}
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
                bloch_vec = basis_product((k1, q1), (k2, q2)).data
                bloch_vec_cache[key] = bloch_vec
            # Skip zero vectors
            if np.any(bloch_vec):
                bloch_vecs.append(bloch_vec)
                weights.append(r * s)

    if not weights:  # all commutators zero
        return GeneralizedBlochVector.zeros()

    coeff_mat = np.vstack(bloch_vecs)      # shape: (P, D)
    weights_vec = np.asarray(weights)      # shape: (P,)
    out = weights_vec @ coeff_mat          # (D,)

    return GeneralizedBlochVector(out)

def bloch_commutator(u: GeneralizedBlochVector,
                     v: GeneralizedBlochVector,
                     tol: float = 0.0) -> GeneralizedBlochVector:
    """Compute the Bloch-space commutator ``[A, B]``.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors representing operators ``A`` and ``B``.
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

    idx_u = np.flatnonzero(np.abs(u.data) > tol)
    idx_v = np.flatnonzero(np.abs(v.data) > tol)
    if idx_u.size == 0 or idx_v.size == 0:
        return GeneralizedBlochVector.zeros()

    ru = u.data[idx_u]
    sv = v.data[idx_v]

    kq_u = [_kq_from_idx(int(n)) for n in idx_u]
    kq_v = [_kq_from_idx(int(n)) for n in idx_v]

    # Cache commutator coefficients to avoid repeated expansion
    coeff_cache: dict[tuple[int,int,int,int], np.ndarray] = {}
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
            if np.any(cvec):
                coeff_rows.append(cvec)
                weights.append(r * s)

    if not weights:  # all commutators zero
        return GeneralizedBlochVector.zeros()

    coeff_mat = np.vstack(coeff_rows)      # shape: (P, D)
    weights_vec = np.asarray(weights)      # shape: (P,)
    out = weights_vec @ coeff_mat          # (D,)

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
    s = np.zeros((d * d,), dtype=complex)
    # q = 0 terms
    for k in range(0, d):
        s[_idx_from_kq(k, 0)] = np.conj(r.data[_idx_from_kq(k, 0)])
    # q != 0 terms: fill positive and negative q simultaneously
    for k in range(0, d):
        for q in range(1, k + 1):
            phase = (-1) ** q
            s[_idx_from_kq(k, q)] = phase * np.conj(r.data[_idx_from_kq(k, -q)])
            s[_idx_from_kq(k, -q)] = phase * np.conj(r.data[_idx_from_kq(k, q)])
    return GeneralizedBlochVector(s)