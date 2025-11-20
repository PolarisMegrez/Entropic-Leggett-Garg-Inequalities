# Leggett-Garg Inequalities Numerical Toolkit

This repository contains the code used in the manuscript **“Conditions for Quantum Violation of Macrorealism in Large-spin Limit” (arXiv:2505.13162 [quant-ph])**. It provides a reproducible environment to generate multi‑time spin‑j measurement statistics and to evaluate Leggett–Garg–type inequalities (entropic, standard, and Wigner form).

**Authors: Yu Xue-Hao and Qiao Cong-Feng (University of Chinese Academy of Sciences, UCAS)**

- Exact joint probability construction.
- Unified evaluation of ELGI, SLGI, and WLGI.
- High‑resolution violation region and minimum‑value plotting utilities.
- Spin‑j generalized Bloch evolution (Hamiltonian + Lindblad) via vendored `bloch4spin`.
- Symmetry‑reduced Wigner 3j/6j caching for performance and memory efficiency.

## Installation

Requirements:
- Python >= 3.10
- numpy >= 1.24, scipy >= 1.11, sympy >= 1.10, matplotlib >= 3.7, jupyter

Clone and install (editable for development):
```powershell
git clone https://github.com/PolarisMegrez/Entropic-Leggett-Garg-Inequalities.git
cd Entropic-Leggett-Garg-Inequalities
pip install -e .
```

Non‑editable install:
```powershell
pip install .
```

Quick Start:
```powershell
jupyter lab
```
Open `demo.ipynb` and run all cells to reproduce:
- 3‑point / 4‑point violation region overlays.
- Minimum value curves vs rotation angle / time.
- Decoherence robustness comparison among ELGI / SLGI / WLGI.

## Repository Layout

```
.
├─ lgbloch/            # Core package: distributions, LGI evaluators, visualization helpers
├─ bloch4spin/         # Auxiliary (standalone) Bloch-space engine: basis, evolution, observables
├─ demo.ipynb          # Main demonstration: reproduces violation plots / comparisons
├─ pyproject.toml      # Packaging (temporary composite release of lgbloch + bloch4spin)
├─ LICENSE             # License placeholder (adjust before public release)
└─ old/                # Historical notebooks / scratch materials (not part of active API)
```

> Note: `bloch4spin` is an independent project (general SU(d) Bloch formalism and Lindblad / Hamiltonian evolution). It is vendored here only to simplify installation for review.

## Detailed Components

The short feature list above is expanded here for technical readers:
- **Inequality modules**: `lgbloch.lgi` implements entropy‑based, correlator, and Wigner LGIs with numerically stable operations (pure NumPy).
- **Probability engine**: `lgbloch.engine.distributions_from_deltas` couples to Bloch evolution to obtain all required multi‑time joint distributions explicitly.
- **Bloch formalism**: `bloch4spin` supplies SU(d) irreducible tensor operator (ITO) basis generation, evolution matrices, and observable liftings.
- **Caching strategy**: Wigner symbols stored only in canonical descending order with parity handling for 3j retrieval to minimize duplicates.
- **Visualization**: `lgbloch.viz` aggregates multi‑output region logic and curve plotting with consistent color/label management.

## Extending

To add additional inequality variants or noise models:
1. Implement distribution generation or channel in `lgbloch.engine` (or extend `bloch4spin.evolution`).
2. Add inequality evaluator in `lgbloch.lgi` following existing style (pure NumPy, no unnecessary JIT).
3. Add plotting integration via `viz.plot_boolean_region` / `viz.plot_multioutput_curves`.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
