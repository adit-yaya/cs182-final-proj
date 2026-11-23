"""Rebuild the final report's three figures and three summary tables from saved CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results"
HELD_OUT = "optimization-experiments/dynamic_tikh_weak_pilot/heldout_validation"
SEEDS = (17, 18, 19)
BLUE, ORANGE, TEAL, PURPLE, GREY = "#245C85", "#A44C16", "#087767", "#754987", "#465461"
COLORS = (BLUE, ORANGE, TEAL)
MARKERS = ("o", "s", "^")


def read_csv(path):
    with (RESULTS / path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(output, name, header, rows):
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(header)
        writer.writerows(rows)


def select(rows, **fields):
    return [row for row in rows if all(row[key] == str(value) for key, value in fields.items())]


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def save_figure(figure, output, name):
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{name}.{extension}", bbox_inches="tight", pad_inches=.12,
                       **({"dpi": 250} if extension == "png" else {}))
    plt.close(figure)


def stage_and_layer(output):
    grid = read_csv("spectral-analysis/architecture_spectral_grid/analysis/spectral-summary.csv")
    depths = read_csv("spectral-analysis/architecture_spectral_grid/analysis/depth-seed-replication.csv")
    native = read_csv("optimization-experiments/dynamic_tikh_weak_pilot/replication_analysis/spectral-summary.csv")
    figure, (position, projection, width) = plt.subplots(1, 3, figsize=(12, 3.45), layout="constrained")
    metric = "fraction_sv_le_0_8"
    depth_table = []
    for i, seed in enumerate(SEEDS):
        for j, depth in enumerate((4, 6, 8)):
            gaps = []
            for step in (100, 500):
                row, = select(depths, seed=seed, depth=depth, step=step)
                gaps.append(float(row[f"{metric}_late_minus_early"]) * 100)
            x = j + np.array([-.16, .16]) + (i - 1) * .045
            position.plot(x, gaps, color=COLORS[i], lw=1.2, alpha=.85)
            position.scatter(x, gaps, marker=MARKERS[i], s=28, color=COLORS[i],
                             label=str(seed) if j == 0 else None, zorder=3)
            depth_table.append([depth, seed, *gaps])
    write_csv(output, "depth-contrast",
              ["depth", "seed", "step100_late_minus_early_pp", "step500_late_minus_early_pp"],
              sorted(depth_table))
    position.axhline(0, color=GREY, lw=.8)
    position.set(xticks=range(3), xticklabels=["Depth 4", "Depth 6", "Depth 8"],
                 ylabel="Late − early attenuation (pp)", ylim=(-5, 20),
                 xlabel="Left: step 100    Right: step 500")
    position.set_title("A   Layer position by depth")
    position.legend(ncols=3, fontsize=8, loc="upper center", columnspacing=.7, handlelength=1)

    operators = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    for step, marker, color, offset in [(100, "o", BLUE, -.1), (500, "s", ORANGE, .1)]:
        for j, operator in enumerate(operators):
            fractions = [100 * values(select(native, step=step, setting="fixed_zero", seed=seed,
                                             operator=operator), metric).mean() for seed in SEEDS]
            projection.scatter(j + offset + np.array([-.045, 0, .045]), fractions, s=28, color=color,
                               marker=marker, label=f"Step {step}" if j == 0 else None)
    projection.set(xticks=np.arange(4), xticklabels=["QKV", "Attn out", "MLP up", "MLP down"],
                   ylabel="Attenuated directions (%)", ylim=(20, 62),
                   xlabel="Width 88, depth 6  ·  three seeds")
    projection.set_title("B   Projection type")
    projection.legend(loc="upper right", fontsize=8)

    steps = [1, 10, 50, 100, 250, 500]
    for size, color, marker in zip([64, 88, 128], COLORS, MARKERS):
        fractions = [100 * values(select(grid, width=size, depth=6, seed=17, filter="gram0", step=step),
                                  metric).mean() for step in steps]
        width.plot(steps, fractions, label=str(size), marker=marker, color=color, lw=1.5, ms=4)
    width.set(xlabel="Training step  ·  seed 17", ylabel="Attenuated directions (%)", xticks=[1, 100, 250, 500])
    width.set_title("C   Exploratory width sweep")
    width.legend(title="Width", ncols=3, fontsize=8, title_fontsize=8, columnspacing=.7, handlelength=1)
    for axis in figure.axes:
        axis.grid(axis="y", alpha=.16)
    save_figure(figure, output, "01-stage-and-layer")


def fixed_filtering(output, losses):
    sweep = read_csv("optimization-experiments/muontikh_lambda_sweep/lambda-transfer-curves.csv")
    endpoints = read_csv("optimization-experiments/dynamic_tikh_weak_pilot/analysis/same-input-endpoints.csv")
    depth_losses = read_csv("optimization-experiments/depth_filter_pilot/seed-replication-summary.csv")
    figure, (response, norm, performance) = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    settings = [("0", "0", GREY, "-"), ("6.89e-06", "W", TEAL, "--"),
                ("6.14e-05", "H", ORANGE, "-."), ("0.000683", "Highest", PURPLE, ":")]
    for level, label, color, style in settings:
        rows = select(sweep, setting=f"Tikh λ={level}", curve="empirical")
        response.semilogx(values(rows, "normalized_input_singular_value"), values(rows, "median_output_response"),
                          color=color, ls=style, label=label, lw=2)
    response.set(xlabel="Normalized input singular value", ylabel="Median directional response")
    response.legend(fontsize=8, ncol=2, loc="upper left")
    response.set_title("A   Empirical filter response")

    matched = select(endpoints, run="zero_to_weak_after100_seed17", step=101)
    assert len(matched) == 24
    for j, (key, color) in enumerate([("weak_to_zero_output_norm_ratio", TEAL),
                                      ("high_to_zero_output_norm_ratio", ORANGE)]):
        ratios = values(matched, key)
        norm.scatter(j + np.linspace(-.11, .11, len(ratios)), np.sort(ratios), s=12, color=color, alpha=.58)
        norm.hlines(ratios.mean(), j - .25, j + .25, color="#132E41", lw=2.8)
        norm.text(j + .27, ratios.mean(), f"{ratios.mean():.2f}", va="center", fontsize=9)
    norm.set(xticks=[0, 1], xticklabels=["W / zero", "H / zero"], ylabel="Filtered / zero Frobenius norm",
             xlabel="Same 24 matrices  ·  seed 17, step 101", xlim=(-.5, 1.7), ylim=(0, 1.12))
    norm.axhline(1, color=GREY, lw=.9, ls="--")
    norm.set_title("B   Update magnitude")

    fixed_table = []
    for seed, color, marker in zip(SEEDS, COLORS, MARKERS):
        depth, = select(depth_losses, seed=seed)
        weak_record, = select(losses, seed=seed, setting="fixed_weak")
        zero = float(depth["depth6_lambda0_loss"])
        weak = float(weak_record["original"])
        high = float(depth["depth6_regularized_loss"])
        fixed_table.append([seed, zero, weak, high, weak-zero, high-zero, float(depth["depth8_delta"])])
        performance.plot([0, 1], [weak-zero, high-zero], color=color, marker=marker, label=str(seed))
    write_csv(output, "fixed-regularization",
              ["seed", "d6_zero", "d6_weak", "d6_high", "d6_weak_minus_zero", "d6_high_minus_zero",
               "d8_high_minus_zero"], fixed_table)
    performance.axhline(0, color=GREY, lw=.9)
    performance.set(xticks=[0, 1], xticklabels=["Weak (W)", "Strong (H)"], xlim=(-.3, 1.3),
                    ylabel="Validation loss − fixed zero", xlabel="Depth 6  ·  2,560 tokens", ylim=(-.03, .058))
    performance.set_title("C   Fixed-λ performance")
    performance.legend(fontsize=8, title="Seed", title_fontsize=8, ncols=3, loc="upper center",
                       columnspacing=.7, handlelength=1)
    for axis in figure.axes:
        axis.grid(axis="y", alpha=.16)
    save_figure(figure, output, "02-fixed-filtering")


def scheduled_validation(output, losses):
    pairs = read_csv(f"{HELD_OUT}/paired-differences.csv")
    sequences = read_csv(f"{HELD_OUT}/sequence-losses.csv")
    settings = ["fixed_zero", "fixed_weak", "weak_to_zero"]
    summary = []
    for window, tokens in [("original", 2560), ("expanded", 131072)]:
        rows = []
        for seed in SEEDS:
            zero, weak, scheduled = [float(select(losses, seed=seed, setting=k)[0][window]) for k in settings]
            rows.append([zero, weak, scheduled, scheduled-zero, scheduled-weak])
            if window == "expanded":
                paired = select(sequences, seed=seed)
                assert len(paired) == 4096
                np.testing.assert_allclose([values(paired, k).mean() for k in settings], [zero, weak, scheduled],
                                           rtol=0, atol=1e-10)
        summary.extend([tokens, seed, *row] for seed, row in zip(SEEDS, rows))
        summary.append([tokens, "Mean", *np.mean(rows, axis=0)])
    write_csv(output, "scheduled-validation",
              ["validation_tokens", "seed", "fixed_zero", "fixed_W", "W_to_zero", "schedule_minus_zero",
               "schedule_minus_W"], summary)

    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.4), layout="constrained")
    for axis, reference, title in zip(axes, settings[:2], ["A   Compared with fixed zero", "B   Compared with fixed W"]):
        rows = [select(pairs, seed=seed, configuration="weak_to_zero", reference=reference)[0] for seed in SEEDS]
        original = values(rows, "original_delta")
        expanded = values(rows, "expanded_delta")
        for j, (before, after) in enumerate(zip(np.r_[original, original.mean()], np.r_[expanded, expanded.mean()])):
            axis.plot([before, after], [j, j], color="#A4AFB5", lw=1.5)
            axis.scatter(before, j, s=49, edgecolor=GREY, facecolor="white",
                         label="2,560 tokens" if j == 0 else None, zorder=3)
            axis.scatter(after, j, s=48, color=TEAL, marker="D",
                         label="131,072 tokens" if j == 0 else None, zorder=4)
        axis.axvline(0, color=GREY, lw=1, ls="--")
        axis.axhline(2.5, color="#C7CFD2", lw=.8)
        axis.set(xlim=(-.041, .055), yticks=range(4), yticklabels=["Seed 17", "Seed 18", "Seed 19", "Mean"],
                 ylim=(3.5, -.5), xlabel="Schedule − baseline loss (lower favors W→0)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=.17)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -.28), ncols=2, fontsize=9)
    save_figure(figure, output, "03-scheduled-validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "final-analysis",
                        help="Destination for regenerated figures and tables (default: results/final-analysis)")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans", "axes.spines.right": False,
        "axes.spines.top": False, "axes.edgecolor": "#75828A", "axes.labelcolor": "#263643",
        "text.color": "#172F41", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "pdf.fonttype": 42, "axes.axisbelow": True,
        "legend.frameon": False, "xtick.color": GREY, "ytick.color": GREY,
    })
    losses = read_csv(f"{HELD_OUT}/losses.csv")
    stage_and_layer(output)
    fixed_filtering(output, losses)
    scheduled_validation(output, losses)


if __name__ == "__main__":
    main()
