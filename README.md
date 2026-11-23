# Analyzing and Optimizing Spectral Filtering in Muon

UC Berkeley CS 182 final project. We study how Muon’s approximate polar update filters momentum in transformer layers, and whether changing the filter during training improves validation loss.

## Results

The [final report](results/final-analysis/report.md) covers the Newton–Schulz and Polar Express comparisons, the architecture experiments, and the effects of regularization on validation loss.

### Spectral Analysis

- [Figure 1 · Attenuation by stage and layer](results/final-analysis/01-stage-and-layer.png) shows how attenuation varies with layer position, training stage, projection type, and model width and depth.

### Optimization Experiments

- [Figure 2 · Fixed regularization](results/final-analysis/02-fixed-filtering.png) shows how fixed λ changes the spectral response, update magnitude, and validation loss.
- [Figure 3 · Scheduled versus fixed regularization](results/final-analysis/03-scheduled-validation.png) compares early-only weak regularization with fixed baselines across three seeds and two validation windows.

To rebuild the three figures (PNG and vector PDF) and their CSV tables from the saved measurements, run `python scripts/plot_final_analysis.py --output-dir /tmp/muon-final-figures` from the repository root with NumPy and Matplotlib installed.
