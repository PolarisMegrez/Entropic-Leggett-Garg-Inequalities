# Leggett-Garg Inequalities Numerical Toolkit 

This code package provides the numerical tools used in the paper:

**“Conditions for Quantum Violation of Macrorealism in Large-spin Limit” (arXiv:2505.13162 [quant-ph])**

## Overview

- Computes multi-time joint probability distributions for spin-j quantum systems.
- Evaluates entropic, standard, and Wigner-type Leggett-Garg inequalities (LGIs).
- Includes ready-to-use plotting functions for violation regions and minimum-value curves.

## Quick Start

1. **Requirements:**
   - Python >= 3.10
   - numpy >= 1.24, scipy >= 1.11, sympy >= 1.10, matplotlib >= 3.7, jupyter

2. **Install dependencies:**
   ```powershell
   pip install numpy scipy sympy matplotlib jupyter
   ```

3. **Usage:**
   - Open `demo.ipynb` in Jupyter Lab/Notebook.
   - Run all cells to reproduce the main results and figures from the paper.

4. **Directory structure:**
   - `lgbloch/` : Main code for LGI evaluation and visualization.
   - `bloch4spin/` : Required dependency for Bloch basis and evolution (also available as an independent project: https://github.com/PolarisMegrez/bloch4spin)
   - `demo.ipynb` : Example notebook to reproduce figures.
   - `pic/` : Output figures.
   - `data/` : Precomputed or saved data files.

## Notes
- This package is provided as supplemental material for academic use and reproduction of published results.
- No installation to PyPI is planned. Simply unzip and use locally.
- For questions or academic use, please cite the original paper.