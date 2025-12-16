"""lgbloch - Leggett-Garg Inequality Testing for Spin-j Systems
================================================================
`lgbloch` provides tools for testing Leggett-Garg inequalities (LGI) in quantum
mechanical systems using generalized Bloch representation. It combines spin
operator construction, joint probability computation, and various LGI test
forms (entropic, standard, and Wigner) with visualization capabilities.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 0.1.1 (Dec 2025)
"""

from .engine import (
    JointProbabilitySet,
    distributions_from_deltas,
    distributions_from_times,
    projectors_Jz,
    run_case,
    spin_ops,
)
from .lgi import (
    entropic_LGI,
    entropic_LGI_for_order_k,
    entropic_LGI_from_chain_rule,
    print_entropic_LGI_forms_for_order_k,
    shannon_entropy,
    standard_LGI_dichotomic,
    wigner_LGI_dichotomic,
)
from .viz import (
    boolean_grid,
    plot_boolean_region,
    plot_multioutput_curves,
    replot_boolean_region,
)

__all__ = [
    # engine
    "JointProbabilitySet",
    "distributions_from_deltas",
    "distributions_from_times",
    "run_case",
    "spin_ops",
    "projectors_Jz",
    # lgi
    "shannon_entropy",
    "entropic_LGI",
    "entropic_LGI_for_order_k",
    "print_entropic_LGI_forms_for_order_k",
    "entropic_LGI_from_chain_rule",
    "standard_LGI_dichotomic",
    "wigner_LGI_dichotomic",
    # viz
    "boolean_grid",
    "plot_boolean_region",
    "replot_boolean_region",
    "plot_multioutput_curves",
]
