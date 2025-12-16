"""lgbloch - Leggett-Garg Inequality Testing for Spin-j Systems
================================================================
`lgbloch` provides efficient numerical testing of Leggett-Garg inequalities
(LGI) in quantum systems of arbitrary spin-$j$. Built on the generalized
Bloch representation, this package offers a unified framework for computing
multi-time joint probabilities and evaluating various macrorealism criteria,
including Entropic, Standard, and Wigner-form inequalities.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 0.2.1 (Dec 2025)
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
