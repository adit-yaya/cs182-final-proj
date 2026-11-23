"""Plot realized transfer responses from the saved NS and PE training runs."""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.muon import zeropower_via_newtonschulz5
from utils.polar_express import PolarExpress, coeffs_list

RUNS = {
    "NS3": ("train_ns3", "ns", 3),
    "NS5": ("train_ns5_pyarrow24", "ns", 5),
    "NS7": ("train_ns7", "ns", 7),
    "PE3": ("train_pe3", "pe", 3),
    "PE5": ("train_pe5", "pe", 5),
    "PE7": ("train_pe7", "pe", 7),
}
COLORS = dict(zip(RUNS, ("#276FBF", "#F28E2B", "#159D82", "#B279A2", "#E15759", "#59A14F")))


def coefficients(method, steps):
    if method == "ns":
        return [(3.4445, -4.7750, 2.0315)] * steps
    return coeffs_list[:steps] + [coeffs_list[-1]] * max(0, steps - len(coeffs_list))


def ideal_response(x, method, steps):
    y = x / 1.01 if method == "pe" else x.copy()
    for a, b, c in coefficients(method, steps):
        y = a * y + b * y**3 + c * y**5
    return np.abs(y)


def normalize_for_svd(matrix):
    x = matrix.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    norm = x.norm(dim=(-2, -1), keepdim=True)
    x = x / (norm + 1e-7)
    return (x.mT if transposed else x).double()


def apply_filter(matrix, method, steps):
    if method == "ns":
        return zeropower_via_newtonschulz5(matrix, steps)
    return PolarExpress(matrix, steps)


def empirical_points(name, run, method, steps):
    snapshots = torch.load(ROOT / "results/spectral-analysis" / run / "momenta.pt",
                           map_location="cpu", weights_only=True)
    xs, ys, rows, naive_values = [], [], [], []
    for snap in snapshots:
        matrix = snap["input"].cpu()
        normalized = normalize_for_svd(matrix)
        u, s, vh = torch.linalg.svd(normalized, full_matrices=False)
        output = apply_filter(matrix, method, steps).double()
        projection = (u.T @ output @ vh.T).diagonal()
        input_sv = s.numpy()
        signed_projection = projection.numpy()
        xs.extend(input_sv.tolist())
        ys.extend(np.abs(signed_projection).tolist())
        if name == "NS5":
            naive_values.extend(input_sv.tolist())
        for rank, (x, y) in enumerate(zip(input_sv, signed_projection), 1):
            rows.append((snap["step"], snap["layer"], rank, float(x), float(y)))
    return np.asarray(xs), np.asarray(ys), rows, naive_values


def binned_quantiles(x, y, bins):
    indices = np.digitize(x, bins) - 1
    centers, medians, lows, highs = [], [], [], []
    for i in range(len(bins) - 1):
        keep = indices == i
        if keep.any():
            centers.append(np.sqrt(bins[i] * bins[i + 1]))
            lows.append(np.quantile(y[keep], 0.10))
            medians.append(np.median(y[keep]))
            highs.append(np.quantile(y[keep], 0.90))
    return np.asarray(centers), np.asarray(medians), np.asarray(lows), np.asarray(highs)


def main():
    torch.set_num_threads(4)
    # Match the completed PE training runs and avoid the local macOS Inductor crash.
    torch.compiler.set_stance("force_eager")
    out = ROOT / "results/spectral-analysis" / "layer_stage_analysis"
    out.mkdir(parents=True, exist_ok=True)
    grid = np.geomspace(1e-5, 1.0, 600)
    bins = np.geomspace(1e-5, 1.0, 46)
    results = {}
    naive_values = []
    with (out / "empirical_transfer.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "run", "step", "layer", "rank",
                         "normalized_input_singular_value", "signed_directional_projection",
                         "absolute_directional_projection"])
        for name, (run, method, steps) in RUNS.items():
            x, y, rows, method_naive_values = empirical_points(name, run, method, steps)
            writer.writerows([name, run, step, layer, rank, value, projection, abs(projection)]
                             for step, layer, rank, value, projection in rows)
            results[name] = (x, y)
            if name == "NS5":
                naive_values = method_naive_values

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 0.72),
                          left=0.08, right=0.98, bottom=0.10, top=0.88,
                          wspace=0.15, hspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    hist_ax = fig.add_subplot(gs[3, :])
    panel_order = ("NS3", "PE3", "NS5", "PE5", "NS7", "PE7")
    for ax, name in zip(axes, panel_order):
        run, method, steps = RUNS[name]
        x, y = results[name]
        color = COLORS[name]
        ax.scatter(x, y, s=4, alpha=0.075, color=color, rasterized=True,
                   label="realized directions")
        centers, med, lo, hi = binned_quantiles(x, y, bins)
        ax.fill_between(centers, lo, hi, color=color, alpha=0.20, linewidth=0,
                        label="empirical 10–90%")
        ax.plot(centers, med, color=color, linewidth=2.0, label="empirical median")
        ax.plot(grid, ideal_response(grid, method, steps), color="black", linestyle="--",
                linewidth=1.7, label="ideal polynomial")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1.0)
        ax.set_ylim(0, 2.0)
        ax.set_title(f"{name} — native run captures")
        ax.set_xlabel("Input singular value / Frobenius norm")
        ax.set_ylabel(r"$|u_i^T Y v_i|$ (directional output)")
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=4, frameon=False)

    naive = np.asarray(naive_values)
    hist_ax.hist(naive, bins=np.geomspace(max(naive.min(), 1e-6), 1.0, 55),
                 color=COLORS["NS5"], alpha=0.82)
    hist_ax.set_xscale("log")
    hist_ax.set_xlim(1e-5, 1.0)
    hist_ax.set_title("Naive Muon (NS5) empirical input singular values — all captured layers and steps")
    hist_ax.set_xlabel("Input singular value / Frobenius norm")
    hist_ax.set_ylabel("Direction count")
    hist_ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Empirical singular-value transfer across NS and Polar Express runs",
                 fontsize=17, y=0.99)
    fig.text(0.5, 0.025,
             "Points use |uᵢᵀYvᵢ|: realized output projected onto each input singular direction; "
             "each panel uses that method’s own training captures. The bottom histogram shows NS5 inputs.",
             ha="center", fontsize=10)
    fig.savefig(out / "empirical-transfer-curves.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {out / 'empirical-transfer-curves.png'}")
    print(f"Wrote {out / 'empirical_transfer.csv'}")


if __name__ == "__main__":
    main()
