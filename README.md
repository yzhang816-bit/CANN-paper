# Curvature-Aware Neural Networks (CANN) — Paper

This repository contains the LaTeX source for the paper:
“Curvature-Aware Neural Networks for Robust Graph Representation Learning”.

## Contents
- `cann_v7.tex`: Compact standalone LaTeX (compiles directly)
- `cann_v8.tex`: Extended version with detailed sections

## Build
- Install a LaTeX distribution (TeX Live or MiKTeX)
- Compile (recommended):
  - Windows (PowerShell):
    - `pdflatex cann_v7.tex`
    - `pdflatex cann_v7.tex`
  - Linux/macOS:
    - `pdflatex cann_v7.tex && pdflatex cann_v7.tex`
- No BibTeX needed (bibliography is inline where present)
- To compile the extended version: replace `cann_v7.tex` with `cann_v8.tex` in the commands

## Repository Structure
- `cann_v7.tex` — paper source (standalone)
- `cann_v8.tex` — extended paper source
- `.gitignore` — ignores LaTeX build artifacts
- `LICENSE` — CC BY 4.0 license for the paper content

## License
This work is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
You are free to share and adapt the material with appropriate credit.
See the LICENSE file for full terms.