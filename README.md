# lgbloch — Leggett-Garg Inequality Testing for Spin-j Systems

`lgbloch` provides numerical tools for testing Leggett-Garg inequalities (LGI) in quantum mechanical systems using generalized Bloch representation. It combines spin operator construction, joint probability computation, and various LGI test forms (entropic, standard, and Wigner) with visualization capabilities for quantum macrorealism studies.

**Author: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)**

- Multi-time joint probability distributions for spin-j quantum systems using sparse Bloch vectors
- Entropic Leggett-Garg inequality tests for macrorealism violation detection
- Standard and Wigner-form Leggett-Garg inequalities with dichotomic measurement support
- Parallel parameter space exploration and violation region visualization
- Efficient implementation built on the generalized Bloch representation framework

This code package provides the numerical tools used in the paper:

**“Conditions for Quantum Violation of Macrorealism in Large-spin Limit” (arXiv:2505.13162 [quant-ph])**

For questions or academic use, please cite the original paper.

## Installation

Requirements:
- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.11
- sympy >= 1.10
- matplotlib >= 3.7

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

## Various Formulations of Leggett-Garg Inequalities

### Entropic LGI (this work)

Tests macrorealism using Shannon entropy inequalities derived from positivity conditions on classical joint probability distributions. Evaluates two elementary inequality types over all measurement time subsets: (1) conditional entropy non-negativity `H(S) - H(S\{i}) >= 0`, and (2) conditional mutual information non-negativity `H(S\{i}) + H(S\{j}) - H(S) - H(S\{i,j}) >= 0`. Supports general n-time tests and specific order-k analyses.

### Entropic LGI (Devi et al.)

Tests macrorealism using the chain rule inequality for Shannon entropy. For k measurement times `{t_1, ..., t_k}`, evaluates: `sum_{m=1}^{k-1} H(t_m, t_{m+1}) - sum_{m=2}^{k-1} H(t_m) - H(t_1, t_k) >= 0`. Iterates over all subsets of size 3 to k.

### Standard LGI (Halliwell)

Tests macrorealism for dichotomic (two-outcome) measurements by reconstructing quasi-probability distributions via dichotomic expansion. Assigns outcomes `s ∈ {+1, -1}` at each time, computes multi-time correlators `<s_1 s_2 ... s_k>` from measured joint probabilities, and reconstructs `p(s_1, ..., s_n)` using the expansion formula. Violation occurs when any reconstructed probability becomes negative.

### Wigner LGI (Saha et al.)

Tests macrorealism using chain inequalities for dichotomic measurements. Evaluates inequalities of the form `P(A=a, B=b) <= P(A=a, M_1=m_1) + P(M_1≠m_1, M_2=m_2) + ... + P(M_k≠m_k, B=b)` over all permutations of measurement times and outcome assignments. Violation occurs when the chain sum becomes smaller than the direct two-time probability.

## Module Overview

### Engine (`lgbloch.engine`)

- `spin_ops(d)`: Construct spin operators Jx, Jy, Jz, Jp, Jm for dimension d.
- `projectors_Jz(d)`: Construct projectors onto Jz eigenstates.
- `distributions_from_times(times, d, ...)`: Compute joint probability distributions for all non-empty time subsets.

### LGI Testing (`lgbloch.lgi`)

- `entropic_LGI(n, jps)`: General entropic LGI test (this work).
- `entropic_LGI_from_chain_rule(n, jps)`: Chain rule entropic LGI test (Devi et al.).
- `standard_LGI_dichotomic(n, jps)`: Standard-form LGI for dichotomic outcomes (Halliwell).
- `wigner_LGI_dichotomic(n, jps)`: Wigner-form LGI for dichotomic outcomes (Saha et al.).

### Visualization (`lgbloch.viz`)

- `boolean_grid(func, xs, ys, ...)`: Evaluate boolean function on parameter grid with parallel execution.
- `plot_boolean_region(func, xs, ys, ...)`: Plot violation regions with customizable styling.

## References

- Q.-H. Cai, X.-H. Yu, M.-C. Yang, A.-X. Liu, C.-F. Qiao, **Conditions for Quantum Violation of Macrorealism in Large-spin Limit**, arXiv:2505.13162, (2025).

- A.R.U. Devi, H.S. Karthik, Sudha, A.K. Rajagopal, **Macrorealism from Entropic Leggett-Garg Inequalities**, *Phys. Rev. A* **87**, 052103 (2013).

- J.J. Halliwell, **Necessary and Sufficient Conditions for Macrorealism Using Two- and Three-Time Leggett-Garg Inequalities**, *J. Phys.: Conf. Ser.* **1275**, 012008 (2019).

- J.J. Halliwell, **Leggett-Garg Tests of Macrorealism: Checks for Noninvasiveness and Generalizations to Higher-Order Correlators**, *Phys. Rev. A* **99**, 022119 (2019).

- D. Saha, S. Mal, P.K. Panigrahi, D. Home, **Wigner's Form of the Leggett-Garg Inequality, the No-Signaling-in-Time Condition, and Unsharp Measurements**, *Phys. Rev. A* **91**, 032117 (2015).

## Notes

- This package is provided as supplemental material for academic use and reproduction of published results.
- All LGI tests return violation values (negative values indicate quantum violations)
- Visualization functions support automatic thread count detection to avoid BLAS oversubscription
- Boolean region data can be saved to .npz files for later replotting

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
