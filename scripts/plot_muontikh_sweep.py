"""Plot losses and empirical spectral attenuation for a MuonTikh lambda sweep."""

import csv
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.muon import tikhonov_via_gram_newtonschulz, zeropower_via_newtonschulz5


def group_name(metadata):
    if metadata["method"] == "ns":
        return "Standard NS5"
    lam = metadata["tikh_lambda"]
    return "Tikh λ=0" if lam == 0 else f"Tikh λ={lam:.3g}"


def empirical_response(snapshots, metadata):
    values, input_singular_values, output_responses = {}, [], []
    for snap in snapshots:
        matrix = snap["input"].float()
        transposed = matrix.size(-2) > matrix.size(-1)
        normalized = matrix.mT if transposed else matrix
        normalized = normalized / (normalized.norm() + 1e-7)
        u, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
        if metadata["method"] == "ns":
            output = zeropower_via_newtonschulz5(matrix, metadata["filter_steps"]).float()
        else:
            output = tikhonov_via_gram_newtonschulz(
                matrix, metadata["filter_steps"], metadata["tikh_lambda"])
        if transposed:
            output = output.mT
        projections = (u.T @ output @ vh.T).diagonal().abs()
        values.setdefault(snap["step"], []).extend(projections.tolist())
        input_singular_values.extend(singular_values.tolist())
        output_responses.extend(projections.tolist())
    return ({step: float(np.median(direction_values))
             for step, direction_values in values.items()},
            np.asarray(input_singular_values), np.asarray(output_responses))


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


def ideal_response(x, name):
    if name.startswith("Tikh"):
        lam = float(name.split("=")[1])
        return np.ones_like(x) if lam == 0 else x / np.sqrt(x**2 + lam)
    y = x.copy()
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        y = a * y + b * y**3 + c * y**5
    return np.abs(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int,
                        help="Plot only this seed, for a matched single-seed comparison")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = ROOT / "results" / "muontikh_lambda_sweep"
    runs = []
    for metadata_path in sorted(root.glob("*/results.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("method") not in ("ns", "tikh"):
            continue
        if args.seed is not None and metadata["seed"] != args.seed:
            continue
        run_dir = metadata_path.parent
        snapshots = torch.load(run_dir / "momenta.pt", map_location="cpu", weights_only=True)
        runs.append((group_name(metadata), metadata, snapshots))
    if not runs:
        raise FileNotFoundError(f"No completed sweep runs found under {root}")

    groups = sorted({name for name, _, _ in runs},
                    key=lambda name: (name != "Standard NS5",
                                      float(name.split("=")[1]) if "=" in name else -1))
    colors = dict(zip(groups, plt.get_cmap("tab10").colors))
    train, validation, response_by_step, transfer = {}, {}, {}, {}
    csv_rows = []
    for name, metadata, snapshots in runs:
        seed = metadata["seed"]
        train.setdefault(name, {})[seed] = np.asarray(metadata["losses"], dtype=float)
        validation.setdefault(name, {})[seed] = {
            item["step"]: item["loss"] for item in metadata["validation_losses"]}
        response_by_step.setdefault(name, {})[seed], xs, ys = empirical_response(snapshots, metadata)
        transfer.setdefault(name, []).append((xs, ys))
        for step, loss in enumerate(metadata["losses"], 1):
            csv_rows.append([name, seed, "training_loss", step, loss])
        for step, loss in validation[name][seed].items():
            csv_rows.append([name, seed, "validation_loss", step, loss])
        for step, value in response_by_step[name][seed].items():
            csv_rows.append([name, seed, "median_directional_output_response", step, value])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for name in groups:
        color = colors[name]
        train_x = np.arange(1, len(next(iter(train[name].values()))) + 1)
        train_values = np.asarray(list(train[name].values()), dtype=float)
        train_mean = train_values.mean(axis=0)
        train_std = train_values.std(axis=0)
        axes[0].plot(train_x, train_mean, color=color, label=name)
        axes[0].fill_between(train_x, train_mean - train_std, train_mean + train_std,
                             color=color, alpha=0.15)
        val_series = validation[name]
        val_steps = sorted(set.intersection(*(set(s) for s in val_series.values())))
        val_values = np.asarray([[s[step] for step in val_steps] for s in val_series.values()])
        axes[1].plot(val_steps, val_values.mean(axis=0), color=color, marker="o", label=name)
        axes[1].fill_between(val_steps, val_values.mean(axis=0) - val_values.std(axis=0),
                             val_values.mean(axis=0) + val_values.std(axis=0),
                             color=color, alpha=0.15)
        att_series = response_by_step[name]
        att_steps = sorted(set.intersection(*(set(s) for s in att_series.values())))
        att_values = np.asarray([[s[step] for step in att_steps] for s in att_series.values()])
        axes[2].errorbar(att_steps, att_values.mean(axis=0), yerr=att_values.std(axis=0),
                         color=color, marker="o", capsize=3, label=name)

    for ax, title, xlabel, ylabel in (
            (axes[0], "Training loss", "Training step", "Cross-entropy loss"),
            (axes[1], "Validation loss", "Training step", "Cross-entropy loss"),
            (axes[2], "Captured singular-value response", "Capture step",
             "Median |uᵢᵀ Y vᵢ| (1 ≈ polar response)")):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MuonTikh λ sweep: optimization and empirical spectral response")
    fig.tight_layout()
    fig.savefig(root / "lambda-sweep-metrics.png", dpi=180)
    plt.close(fig)
    with (root / "lambda-sweep-metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "seed", "metric", "step", "value"])
        writer.writerows(csv_rows)
    print(f"Wrote {root / 'lambda-sweep-metrics.png'}")
    print(f"Wrote {root / 'lambda-sweep-metrics.csv'}")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    grid = np.geomspace(1e-5, 1.0, 500)
    bins = np.geomspace(1e-5, 1.0, 46)
    transfer_rows = []
    for name in groups:
        x = np.concatenate([item[0] for item in transfer[name]])
        y = np.concatenate([item[1] for item in transfer[name]])
        keep = (x >= bins[0]) & np.isfinite(x) & np.isfinite(y)
        centers, median, low, high = binned_quantiles(x[keep], y[keep], bins)
        color = colors[name]
        ax.plot(centers, median, color=color, linewidth=2, label=f"{name} empirical")
        ax.plot(grid, ideal_response(grid, name), color=color, linestyle="--", alpha=0.75)
        transfer_rows.extend((name, "empirical", float(cx), float(m), float(lo), float(hi))
                             for cx, m, lo, hi in zip(centers, median, low, high))
        transfer_rows.extend((name, "ideal", float(cx), float(iy), "", "")
                             for cx, iy in zip(grid, ideal_response(grid, name)))
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Input singular value / Frobenius norm")
    ax.set_ylabel(r"Directional output response $|u_i^T Y v_i|$")
    ax.set_title("Empirical transfer curves across Tikhonov λ settings")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.015,
             "Solid: median response from each run’s captured momentum matrices. "
             "Dashed: target Tikhonov response (NS5 polynomial for baseline).",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(root / "lambda-transfer-curves.png", dpi=180)
    plt.close(fig)
    with (root / "lambda-transfer-curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "curve", "normalized_input_singular_value",
                         "median_output_response", "p10_output_response", "p90_output_response"])
        writer.writerows(transfer_rows)
    print(f"Wrote {root / 'lambda-transfer-curves.png'}")
    print(f"Wrote {root / 'lambda-transfer-curves.csv'}")


if __name__ == "__main__":
    main()
