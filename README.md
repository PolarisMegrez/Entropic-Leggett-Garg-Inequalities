# lgbloch — Leggett-Garg Inequality Testing for Spin-j Systems

`lgbloch` provides efficient numerical testing of Leggett-Garg inequalities (LGI) in quantum systems of arbitrary spin-$j$. Built on the generalized Bloch representation, this package offers a unified framework for computing multi-time joint probabilities and evaluating various macrorealism criteria, including Entropic, Standard, and Wigner-form inequalities.

**Author: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)**

- Computes multi-time joint probability distributions for arbitrary spin-$j$ systems using sparse Bloch vectors.
- Detects macrorealism violations using Shannon entropy constraints (conditional entropy and mutual information).
- Supports conventional correlator-based and Wigner-form inequalities for dichotomic measurements.
- Efficiently explores parameter spaces and visualizes violation regions using multi-core processing.
- Optimized implementation leveraging the generalized Bloch representation framework.

This repository serves as the source code release for the following research:

**Conditions for Quantum Violation of Macrorealism in Large-spin Limit** (arXiv:2505.13162 [quant-ph])

## Installation

Requirements:
- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.11
- sympy >= 1.10
- matplotlib >= 3.7
- joblib >= 1.3
- threadpoolctl >= 3.0

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install from source (editable for development):

```powershell
git clone https://github.com/PolarisMegrez/lgbloch.git
cd lgbloch
pip install -r requirements.txt
pip install -e .
```

Or install directly from a local checkout without editable mode:

```powershell
pip install -r requirements.txt
pip install .
```

## Formulations of Leggett-Garg Inequalities

### Entropic LGI (this work)

Tests macrorealism using Shannon entropy inequalities derived from the geometry of the Shannon cone. The implementation evaluates two fundamental positivity conditions over all measurement time subsets:
1. **Conditional Entropy Non-negativity**: $H(S) - H(S \setminus \{i\}) \ge 0$
2. **Conditional Mutual Information Non-negativity**: $H(S \setminus \{i\}) + H(S \setminus \{j\}) - H(S) - H(S \setminus \{i,j\}) \ge 0$

This formulation supports general $n$-time tests and specific order-$k$ correlation analyses.

### Entropic LGI (Devi et al.)

Tests macrorealism via the entropic chain rule inequality. For a sequence of $k$ measurements $\{t_1, \dots, t_k\}$, it evaluates the non-negativity of the sum of pairwise entropies minus marginals:
$\sum_{m=1}^{k-1} H(t_m, t_{m+1}) - \sum_{m=2}^{k-1} H(t_m) - H(t_1, t_k) \ge 0$.
The test iterates over all subsets of size 3 to $n$.

### Standard LGI (Halliwell)

Tests macrorealism for dichotomic (two-outcome) measurements by reconstructing quasi-probability distributions. It assigns outcomes $s \in \{+1, -1\}$ at each time, computes multi-time correlators $\langle s_1 s_2 \dots s_k \rangle$ from the quantum joint probabilities, and reconstructs the history probability $p(s_1, \dots, s_n)$ using the Halliwell expansion formula. A violation is flagged if any reconstructed probability is negative.

### Wigner LGI (Saha et al.)

Tests macrorealism using Wigner-form chain inequalities for dichotomic measurements. It evaluates inequalities of the form:
$P(A=a, B=b) \le P(A=a, M_1=m_1) + P(M_1 \ne m_1, M_2=m_2) + \dots + P(M_k \ne m_k, B=b)$
This checks if the direct two-time probability is bounded by the sum of chain probabilities involving intermediate measurements.

## Module Overview

### Engine (`lgbloch.engine`)

- `spin_ops(d)`: Construct generalized spin operators ($J_x, J_y, J_z, J_+, J_-$) for dimension $d=2j+1$.
- `projectors_Jz(d)`: Generate projectors onto $J_z$ eigenstates for measurement simulation.
- `distributions_from_times(times, d, ...)`: Compute multi-time joint probability distributions for arbitrary time sequences using the generalized Bloch evolution.

### LGI Testing (`lgbloch.lgi`)

- `entropic_LGI(n, jps)`: Evaluates general entropic LGIs, including higher-order constraints derived from the Shannon cone.
- `entropic_LGI_from_chain_rule(n, jps)`: Implements the specific chain-rule based ELGIs (Devi et al.).
- `standard_LGI_dichotomic(n, jps)`: Computes standard-form LGIs for dichotomic outcomes.
- `wigner_LGI_dichotomic(n, jps)`: Computes Wigner-form LGIs for dichotomic outcomes.

### Visualization (`lgbloch.viz`)

- `boolean_grid(func, xs, ys, ...)`: Performs parallelized evaluation of boolean functions (e.g., LGI violations) over 2D parameter grids.
- `plot_boolean_region(func, xs, ys, ...)`: Visualizes violation regions with customizable styling.
- `plot_multioutput_curves(...)`: Plots multiple LGI values against a parameter, useful for comparing different inequality types.

## Parallelization Strategy

`lgbloch` employs a robust parallelization strategy designed for high-performance computing on multi-core systems.

- **Process-based Parallelism**: Uses `joblib` with the `loky` backend to bypass the Python Global Interpreter Lock (GIL), ensuring full utilization of multi-core CPUs for computational tasks.
- **BLAS Thread Control**: Automatically limits BLAS (MKL, OpenBLAS, etc.) to single-threaded mode within each worker process using `threadpoolctl`. This prevents "oversubscription" where nested threading explodes the total thread count and degrades performance.
- **Hardware Awareness**:
  - Automatically detects **physical cores** (via `psutil`) rather than logical processors to avoid contention on Hyper-Threading/SMT systems.
  - Respects OS-specific limits, such as the 64-handle limit on Windows `WaitForMultipleObjects`, capping workers at 61 on Windows to ensure stability.
- **Configuration**:
  - `backend`: Defaults to `'auto'` (process-based). Can be set to `'threading'` for debugging or I/O-bound tasks.
  - `n_jobs`: Defaults to an optimized value based on the hardware detection logic described above. Can be manually overridden.

## References

- Q.-H. Cai, X.-H. Yu, M.-C. Yang, A.-X. Liu, C.-F. Qiao, **Conditions for Quantum Violation of Macrorealism in Large-spin Limit**, arXiv:2505.13162 (2025).
- A.R.U. Devi, H.S. Karthik, Sudha, A.K. Rajagopal, **Macrorealism from Entropic Leggett-Garg Inequalities**, *Phys. Rev. A* **87**, 052103 (2013).
- J.J. Halliwell, **Necessary and Sufficient Conditions for Macrorealism Using Two- and Three-Time Leggett-Garg Inequalities**, *J. Phys.: Conf. Ser.* **1275**, 012008 (2019).
- J.J. Halliwell, **Leggett-Garg Tests of Macrorealism: Checks for Noninvasiveness and Generalizations to Higher-Order Correlators**, *Phys. Rev. A* **99**, 022119 (2019).
- D. Saha, S. Mal, P.K. Panigrahi, D. Home, **Wigner's Form of the Leggett-Garg Inequality, the No-Signaling-in-Time Condition, and Unsharp Measurements**, *Phys. Rev. A* **91**, 032117 (2015).

## Notes

- This package is provided as supplemental material for academic use and reproduction of published results.
- All LGI tests return violation values (negative values indicate quantum violations).
- Visualization functions support automatic thread count detection to avoid BLAS oversubscription.
- Boolean region data can be saved to .npz files for later replotting.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.