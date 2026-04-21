# Aimo Solver Reproducibility Package

This folder packages the assets for the AIMO Solver (v58) writeup and figure set.

## Contents

- `notebook/aimo-solver-v58.ipynb`: notebook copy used for the v58 writeup evidence.
- `scripts/generate_figures.py`: script used to generate the writeup figures and data tables.
- `log.md`: execution log used for runtime and timeline figures.
- `figures/`: exported PNGs and generated CSV data.
- `data/`: copied data snapshots used by the writeup figure analysis.
- `../aimo-solver-colab.ipynb`: Colab-compatible harness notebook for reproducibility links.

## Figure Mapping Used In writeup-kaggle.md

- Figure 1 -> `figures/fig01_architecture.png`
- Figure 2 -> `figures/fig02_attempt_diagnostics.png`
- Figure 3 -> `figures/fig03_runtime_breakdown.png`
- Figure 4 -> `figures/fig04_entropy_vs_vote_weight.png`
- Figure 5 -> `figures/fig05_execution_timeline.png`
- Figure 6 -> `figures/fig06_entropy_distribution.png`

## Regenerate Figures

Run from this folder:

```bash
python scripts/generate_figures.py
```

The script writes generated outputs to `figures/`.

## Notes

- For external reproducibility links, upload this folder to GitHub and optionally mirror the same workflow in a public Colab notebook.
