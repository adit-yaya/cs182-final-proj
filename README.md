# Analyzing and Optimizing Spectral Filtering in Muon

UC Berkeley CS 182 final project. We study how Muon’s approximate polar update filters momentum in transformer layers, and whether changing the filter during training improves validation loss.

## Final results

- [Read the final report](results/final-analysis/report.md) for the setup, evidence, and limitations.
- [Figure 1 · Attenuation by stage and layer](results/final-analysis/01-stage-and-layer.png) shows the early gap between later and earlier layers across three depths and three seeds.
- [Figure 3 · Scheduled versus fixed regularization](results/final-analysis/03-scheduled-validation.png) shows why the lowest mean held-out loss for early-only weak regularization still leaves a different winner for each seed.
- [Exact validation losses and paired differences](results/final-analysis/scheduled-validation.csv) gives the individual seeds and means on both the 2,560-token and 131,072-token validation windows. Negative differences favor the schedule.

To rebuild the three figures (PNG and vector PDF) and their CSV tables from the saved measurements, run `python scripts/plot_final_analysis.py --output-dir /tmp/muon-final-figures` from the repository root with NumPy and Matplotlib installed.
