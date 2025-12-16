"""bloch4spin - Generalized Bloch representation utilities for spin-j systems
==========================================================================
`bloch4spin` provides a generalized Bloch representation for finite spin systems
(`d = 2j + 1`) using irreducible tensor operators (ITO). The ITO-based Bloch
vector is typically sparse, allowing fast Hamiltonian-to-Liouvillian conversion,
efficient time evolution, and lightweight measurement operations.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 0.1.2 (Dec 2025)
"""

from .basis import (
    GeneralizedBlochVector,
    bloch_basis,
    bloch_commutator,
    bloch_dim,
    bloch_hermitian_transpose,
    bloch_init,
    bloch_inner_product,
    bloch_tensor_product,
    structure_const,
)
from .engine import (
    run,
)
from .evolution import (
    GeneralizedBlochEvolutionMatrix,
    GeneralizedBlochHamiltonian,
    GeneralizedBlochState,
)
from .observable import (
    GeneralizedBlochObservable,
)

__all__ = [
    "GeneralizedBlochHamiltonian",
    "GeneralizedBlochEvolutionMatrix",
    "GeneralizedBlochState",
    "bloch_init",
    "bloch_dim",
    "bloch_basis",
    "structure_const",
    "GeneralizedBlochVector",
    "bloch_inner_product",
    "bloch_hermitian_transpose",
    "bloch_commutator",
    "bloch_tensor_product",
    "GeneralizedBlochObservable",
    "run",
]
