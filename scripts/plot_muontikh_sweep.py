"""Plot losses and empirical spectral attenuation for a MuonTikh lambda sweep."""

import csv
import argparse
import hashlib
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


def tikh_lambda_at_step(metadata, step):
    switch = metadata.get("tikh_switch_step")
    return (metadata["tikh_lambda_after"] if switch is not None and step > switch
            else metadata["tikh_lambda"])


def group_name(metadata):
    if metadata["method"] == "ns":
        return "Standard NS5"
    lam = metadata["tikh_lambda"]
    if metadata.get("tikh_switch_step") is not None:
        return f"λ={lam:.3g} → {metadata['tikh_lambda_after']:.3g} after {metadata['tikh_switch_step']}"
    return "Tikh λ=0" if lam == 0 else f"Tikh λ={lam:.3g}"


def empirical_response(snapshots, metadata, summary_rows=None):
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
            lam = tikh_lambda_at_step(metadata, snap["step"])
            if "tikh_lambda" in snap and snap["tikh_lambda"] != lam:
                raise ValueError(f"Capture lambda disagrees with schedule: {snap['step']}")
            output = tikhonov_via_gram_newtonschulz(
                matrix, metadata["filter_steps"], lam)
        if transposed:
            output = output.mT
        projections = (u.T @ output @ vh.T).diagonal().abs()
        if summary_rows is not None:
            output_sv = torch.linalg.svdvals(output.double())
            block = int(snap["layer"].split(".")[2])
            quantiles = torch.quantile(singular_values.double(),
                                      torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64))
            summary_rows.append(dict(
                depth=metadata["config"]["n_layer"], seed=metadata["seed"],
                tikh_lambda=tikh_lambda_at_step(metadata, snap["step"]), step=snap["step"],
                layer=snap["layer"], block=block,
                relative_depth=(block + 0.5) / metadata["config"]["n_layer"],
                operator=".".join(snap["layer"].split(".")[3:-1]),
                rows=matrix.shape[0], columns=matrix.shape[1], directions=len(output_sv),
                input_frobenius_norm=matrix.norm().item(),
                output_frobenius_norm=output.norm().item(),
                scaled_update_frobenius_norm=output.norm().item()
                * max(1, matrix.shape[0] / matrix.shape[1])**0.5,
                output_sv_median=output_sv.quantile(0.5).item(),
                fraction_sv_le_0_8=(output_sv <= 0.8).double().mean().item(),
                fraction_sv_le_0_2=(output_sv <= 0.2).double().mean().item(),
                median_directional_output_response=projections.double().quantile(0.5).item(),
                **dict(zip(("input_sv_q10", "input_sv_q25", "input_sv_q50",
                            "input_sv_q75", "input_sv_q90"), quantiles.tolist()))))
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


def depth_pilot(seed=17):
    root = ROOT / "results"
    pilot = root / "depth_filter_pilot"
    paths = [root / "muontikh_lambda_sweep" / name for name in
             ("tikh_lam0_seed_17", "tikh_lam6p14e-5_seed_17")]
    paths += [pilot / name for name in
              ("w88_d8_tikh_lam0_seed17", "w88_d8_tikh_lam6p14e-5_seed17")]
    if seed != 17:
        paths = [pilot / f"w88_d{depth}_tikh_lam{label}_seed{seed}"
                 for depth in (6, 8) for label in ("0", "6p14e-5")]
    runs = [(path, json.loads((path / "results.json").read_text())) for path in paths]
    reference = runs[0][1]
    controls = ("data", "seed", "steps", "batch", "capture", "device", "method",
                "filter_steps", "shuffle_buffer", "eval_every", "eval_batches",
                "muon_lr", "momentum", "adam_lr", "torch", "packages")
    for index, (path, metadata) in enumerate(runs):
        expected_config = dict(reference["config"], n_layer=6 if index < 2 else 8)
        expected_lambda = 0 if index % 2 == 0 else 6.1352038e-5
        if (metadata["config"] != expected_config
                or metadata["tikh_lambda"] != expected_lambda
                or any(metadata[key] != reference[key] for key in controls)):
            raise ValueError(f"Unmatched pilot configuration: {path}")
        for source in ("models/model.py", "utils/muon.py", "utils/data.py", "utils/capture.py"):
            if metadata["source_sha256"][source] != reference["source_sha256"][source]:
                raise ValueError(f"Unmatched {source}: {path}")
        if len(metadata["losses"]) != metadata["steps"]:
            raise ValueError(f"Incomplete loss history: {path}")
    out = pilot / ("analysis" if seed == 17 else f"analysis_seed{seed}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    summary_rows, loss_rows, final_rows, transfer_rows = [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    deltas = {}
    for path, metadata in runs:
        depth, lam = metadata["config"]["n_layer"], metadata["tikh_lambda"]
        label = f"depth {depth}, λ={lam:.3g}"
        snapshots = torch.load(path / "momenta.pt", map_location="cpu", weights_only=True)
        expected = {(step, f"transformer.h.{block}.{operator}.weight")
                    for step in metadata["capture"] for block in range(depth)
                    for operator in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
        if len(snapshots) != len(expected) or {(s["step"], s["layer"]) for s in snapshots} != expected:
            raise ValueError(f"Incomplete momentum captures: {path}")
        response, xs, ys = empirical_response(snapshots, metadata, summary_rows)
        bins = np.geomspace(1e-5, 1, 46)
        centers, medians, lows, highs = binned_quantiles(xs, ys, bins)
        axes[2].plot(centers, medians, label=label)
        axes[2].fill_between(centers, lows, highs, alpha=0.1)
        transfer_rows.extend(dict(depth=depth, tikh_lambda=lam, input_sv=float(center),
                                  median=float(median), p10=float(low), p90=float(high))
                             for center, median, low, high in zip(centers, medians, lows, highs))
        axes[0].plot(range(1, len(metadata["losses"]) + 1), metadata["losses"], label=label)
        validation = metadata["validation_losses"]
        axes[1].plot([item["step"] for item in validation],
                     [item["loss"] for item in validation], marker="o", label=label)
        for metric, series in (
                ("training_loss", enumerate(metadata["losses"], 1)),
                ("validation_loss", ((item["step"], item["loss"]) for item in validation)),
                ("median_directional_output_response", sorted(response.items()))):
            loss_rows.extend(dict(depth=depth, tikh_lambda=lam, seed=metadata["seed"],
                                  metric=metric, step=step, value=value) for step, value in series)
        width = metadata["config"]["n_embd"]
        muon_parameters = 12 * depth * width**2
        final_rows.append(dict(depth=depth, tikh_lambda=lam, seed=metadata["seed"],
                               validation_loss=validation[-1]["loss"],
                               parameters=metadata["parameters"], muon_parameters=muon_parameters,
                               adam_parameters=metadata["parameters"] - muon_parameters,
                               training_tokens=metadata["training_tokens"],
                               training_and_eval_seconds=metadata["training_and_eval_seconds"]))
        deltas.setdefault(depth, {})[lam] = {item["step"]: item["loss"] for item in validation}
    for axis, title in zip(axes, ("Training loss", "Validation loss", "Native-run transfer response")):
        axis.set_title(title)
        axis.set_xlabel("Normalized input SV" if axis == axes[2] else "Training step")
        axis.set_ylabel("Directional output response" if axis == axes[2] else "Cross-entropy")
        axis.grid(alpha=0.25)
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    interaction_rows = []
    for step in sorted(deltas[6][0]):
        delta6 = deltas[6][6.1352038e-5][step] - deltas[6][0][step]
        delta8 = deltas[8][6.1352038e-5][step] - deltas[8][0][step]
        interaction_rows.append(dict(step=step, depth6_regularization_delta=delta6,
                                     depth8_regularization_delta=delta8, interaction=delta8 - delta6))
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out / "losses-and-transfer.png", dpi=180)
    plt.close(fig)
    for name, rows in (("spectral-summary", summary_rows), ("metrics", loss_rows),
                       ("final-results", final_rows), ("transfer", transfer_rows),
                       ("interaction", interaction_rows)):
        with (out / f"{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    operators = sorted({row["operator"] for row in summary_rows})
    for dimension in ("step", "relative_depth"):
        metrics = ("output_sv_median", "fraction_sv_le_0_8", "fraction_sv_le_0_2", "input_sv_q50")
        fig, axes = plt.subplots(len(metrics), len(operators), figsize=(17, 13), squeeze=False)
        for metric_index, metric in enumerate(metrics):
            for operator_index, operator in enumerate(operators):
                axis = axes[metric_index, operator_index]
                for _, metadata in runs:
                    depth, lam = metadata["config"]["n_layer"], metadata["tikh_lambda"]
                    selected = [row for row in summary_rows if row["depth"] == depth
                                and row["tikh_lambda"] == lam and row["operator"] == operator
                                and (dimension == "step" or row["step"] == metadata["steps"])]
                    coordinates = sorted({row[dimension] for row in selected})
                    means = [np.mean([row[metric] for row in selected if row[dimension] == coordinate])
                             for coordinate in coordinates]
                    axis.plot(coordinates, means, marker=".", label=f"d{depth}, λ={lam:.3g}")
                axis.set_title(f"{operator}: {metric}", fontsize=8)
                axis.set_xlabel(dimension)
                axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Per-matrix summaries averaged across blocks" if dimension == "step"
                     else "Final capture by relative layer depth")
        fig.tight_layout()
        fig.savefig(out / f"spectra-by-{dimension}.png", dpi=180)
        plt.close(fig)
    (out / "sources.json").write_text(json.dumps(dict(
        runs=[str(path) for path, _ in runs],
        reference="SVs before aspect-ratio scaling; input SVs Frobenius-normalized; native Nesterov inputs",
        interaction="(lambda-positive minus lambda-zero loss at depth 8) minus the same at depth 6",
        metadata=[metadata for _, metadata in runs]), indent=2))
    print(f"Wrote pilot report to {out}")


@torch.no_grad()
def architecture_spectra(snapshot, metadata, direction_writer):
    matrix = snapshot["input"].float()
    transposed = matrix.shape[0] > matrix.shape[1]
    normalized = matrix / (matrix.norm() + 1e-7)
    normalized = normalized.mT if transposed else normalized
    left, singular_values, right = torch.linalg.svd(normalized.double(), full_matrices=False)
    rank = len(singular_values)
    block = int(snapshot["layer"].split(".")[2])
    identity = dict(width=metadata["config"]["n_embd"], depth=metadata["config"]["n_layer"],
                    seed=metadata["seed"], step=snapshot["step"], layer=snapshot["layer"],
                    block=block, relative_depth=(block + 0.5) / metadata["config"]["n_layer"],
                    operator=".".join(snapshot["layer"].split(".")[3:-1]),
                    rows=matrix.shape[0], columns=matrix.shape[1], directions=rank)
    regularization = 6.1352038e-5
    outputs = {
        "gram0": tikhonov_via_gram_newtonschulz(matrix, 5, 0).double(),
        "gram_regularized": tikhonov_via_gram_newtonschulz(matrix, 5, regularization).double(),
        "ns5_bf16": zeropower_via_newtonschulz5(matrix, 5).double(),
    }
    outputs = {name: output.mT if transposed else output for name, output in outputs.items()}
    projections = {name: (left.T @ output @ right.T).diagonal()
                   for name, output in outputs.items()}
    quantiles = torch.quantile(singular_values, torch.tensor([.1, .25, .5, .75, .9], dtype=torch.float64))
    baseline = projections["gram0"].abs()
    summaries, paired = [], []
    for name, output in outputs.items():
        if not torch.isfinite(output).all():
            raise ValueError(f"Nonfinite output: {identity}, {name}")
        output_sv = torch.linalg.svdvals(output)
        projection = projections[name]
        summaries.append(dict(identity, filter=name, filter_lambda=regularization if name == "gram_regularized" else 0,
                              matrices=1, input_frobenius_norm=matrix.norm().item(),
                              output_frobenius_norm=output.norm().item(),
                              output_norm_ratio_to_gram0=(output.norm() / outputs["gram0"].norm()).item(),
                              scaled_update_frobenius_norm=output.norm().item()
                              * max(1, matrix.shape[0] / matrix.shape[1])**.5,
                              output_sv_median=output_sv.quantile(.5).item(),
                              fraction_sv_le_0_8=(output_sv <= .8).double().mean().item(),
                              fraction_sv_le_0_2=(output_sv <= .2).double().mean().item(),
                              median_directional_output_response=projection.abs().quantile(.5).item(),
                              **{f"input_sv_q{percent}": value for percent, value in
                                 zip((10, 25, 50, 75, 90), quantiles.tolist())},
                              **{f"rank_adjusted_input_sv_q{percent}": value * rank**.5 for percent, value in
                                 zip((10, 25, 50, 75, 90), quantiles.tolist())}))
        for direction, (input_sv, signed_response) in enumerate(zip(singular_values.tolist(), projection.tolist()), 1):
            direction_writer.writerow(dict(identity, filter=name, direction=direction,
                                           normalized_input_sv=input_sv, rank_adjusted_input_sv=input_sv * rank**.5,
                                           signed_directional_projection=signed_response,
                                           absolute_directional_projection=abs(signed_response)))
        for band, lower, upper in (("tiny", 0, .001), ("small", .001, regularization**.5),
                                   ("middle", regularization**.5, .03), ("large", .03, float("inf"))):
            selected = (singular_values >= lower) & (singular_values < upper)
            valid = selected & (baseline > 1e-8)
            ratio = projection.abs()[valid] / baseline[valid]
            paired.append(dict(identity, filter=name, band=band, matrices=1,
                               band_lower=lower, band_upper=upper,
                               band_directions=int(selected.sum()), ratio_directions=int(valid.sum()),
                               median_response_ratio_to_gram0=ratio.quantile(.5).item() if len(ratio) else None,
                               median_response_change=(projection.abs()[selected] - baseline[selected]).quantile(.5).item()
                               if selected.any() else None))
            summaries[-1][f"{band}_response_ratio_to_gram0"] = paired[-1]["median_response_ratio_to_gram0"]
    return summaries, paired


def architecture_report():
    root = ROOT / "results"
    grid = root / "architecture_spectral_grid"
    paths = {(88, 6): root / "muontikh_lambda_sweep/tikh_lam0_seed_17",
             (88, 8): root / "depth_filter_pilot/w88_d8_tikh_lam0_seed17"}
    paths.update({(width, depth): grid / f"w{width}_d{depth}_tikh_lam0_seed17"
                  for width, depth in ((64, 6), (128, 6), (88, 4))})
    runs = [(dimensions, path, json.loads((path / "results.json").read_text()))
            for dimensions, path in paths.items()]
    reference = runs[0][2]
    controls = ("data", "seed", "steps", "batch", "capture", "device", "method", "filter_steps",
                "tikh_lambda", "shuffle_buffer", "eval_every", "eval_batches", "muon_lr", "momentum",
                "adam_lr", "torch", "packages")
    for (width, depth), path, metadata in runs:
        if (metadata["config"] != dict(reference["config"], n_embd=width, n_layer=depth)
                or any(metadata[key] != reference[key] for key in controls)):
            raise ValueError(f"Unmatched architecture configuration: {path}")
        if len(metadata["losses"]) != 500 or [item["step"] for item in metadata["validation_losses"]] != [100, 200, 300, 400, 500]:
            raise ValueError(f"Incomplete loss history: {path}")
        for source in ("models/model.py", "utils/muon.py", "utils/data.py", "utils/capture.py"):
            if metadata["source_sha256"][source] != hashlib.sha256((ROOT / source).read_bytes()).hexdigest():
                raise ValueError(f"Saved/current source mismatch for {source}: {path}")
    out = grid / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    summaries, paired, model_rows, losses = [], [], [], []
    direction_fields = ["width", "depth", "seed", "step", "layer", "block", "relative_depth", "operator",
                        "rows", "columns", "directions", "filter", "direction", "normalized_input_sv",
                        "rank_adjusted_input_sv", "signed_directional_projection", "absolute_directional_projection"]
    with (out / "directions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=direction_fields)
        writer.writeheader()
        for (width, depth), path, metadata in runs:
            snapshots = torch.load(path / "momenta.pt", map_location="cpu", weights_only=True)
            shapes = {"attn.c_attn": (3 * width, width), "attn.c_proj": (width, width),
                      "mlp.c_fc": (4 * width, width), "mlp.c_proj": (width, 4 * width)}
            expected = {(step, f"transformer.h.{block}.{operator}.weight")
                        for step in metadata["capture"] for block in range(depth) for operator in shapes}
            if len(snapshots) != len(expected) or {(snap["step"], snap["layer"]) for snap in snapshots} != expected:
                raise ValueError(f"Incomplete captures: {path}")
            for snapshot in snapshots:
                operator = ".".join(snapshot["layer"].split(".")[3:-1])
                if tuple(snapshot["input"].shape) != shapes[operator] or not torch.isfinite(snapshot["input"]).all():
                    raise ValueError(f"Invalid captured input: {path}, {snapshot['layer']}")
                native, comparison = architecture_spectra(snapshot, metadata, writer)
                summaries.extend(native)
                paired.extend(comparison)
            muon_parameters = 12 * depth * width**2
            model_rows.append(dict(width=width, depth=depth, parameters=metadata["parameters"],
                                   muon_parameters=muon_parameters, adam_parameters=metadata["parameters"] - muon_parameters,
                                   head_dimension=width // metadata["config"]["n_head"],
                                   projection_init_std=.02 / (2 * depth)**.5, captured_matrices=len(snapshots),
                                   captured_directions=len(snapshots) * width,
                                   final_validation_loss=metadata["validation_losses"][-1]["loss"],
                                   training_tokens=metadata["training_tokens"],
                                   training_and_eval_seconds=metadata["training_and_eval_seconds"]))
            losses.extend(dict(width=width, depth=depth, metric="training_loss", step=step, value=loss)
                          for step, loss in enumerate(metadata["losses"], 1))
            for item in metadata["validation_losses"]:
                losses.extend(dict(width=width, depth=depth, metric=f"validation_{metric}", step=item["step"], value=item[metric])
                              for metric in ("loss", "perplexity"))
    for name, rows in (("spectral-summary", summaries), ("paired-responses", paired),
                       ("architectures", model_rows), ("losses", losses)):
        with (out / f"{name}.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    architecture_figures(summaries, losses, out)
    (out / "sources.json").write_text(json.dumps(dict(
        runs=[dict(path=str(path), metadata=metadata) for _, path, metadata in runs],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="Same Nesterov inputs for all filters; SVs before aspect-ratio scaling; rank adjustment sqrt(min(rows,columns))",
        caveats=["One seed; directions are not independent replicates", "Width changes head dimension and matrix size",
                 "Depth changes projection initialization", "Gram0 is the trained filter; other filters are offline counterfactuals",
                 "Fractions use <= thresholds; response ratios exclude baseline magnitudes <=1e-8"]), indent=2))
    print(f"Wrote five-model architecture report to {out}")


def architecture_figures(rows, losses, out):
    operators = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    metrics = ("input_sv_q50", "rank_adjusted_input_sv_q50", "output_sv_median",
               "fraction_sv_le_0_8", "fraction_sv_le_0_2", "output_norm_ratio_to_gram0",
               "tiny_response_ratio_to_gram0", "small_response_ratio_to_gram0")
    for comparison in ("width", "depth"):
        selected = [row for row in rows if (row["depth"] == 6 if comparison == "width" else row["width"] == 88)]
        for step in sorted({row["step"] for row in selected}):
            fig, axes = plt.subplots(len(metrics), len(operators), figsize=(18, 24), squeeze=False)
            for metric_index, metric in enumerate(metrics):
                for operator_index, operator in enumerate(operators):
                    axis = axes[metric_index, operator_index]
                    for value in sorted({row[comparison] for row in selected}):
                        for name, style in (("gram0", "-"), ("gram_regularized", "--"), ("ns5_bf16", ":")):
                            if "input_sv" in metric and name != "gram0":
                                continue
                            points = sorted((row for row in selected if row["step"] == step and row["operator"] == operator
                                             and row[comparison] == value and row["filter"] == name), key=lambda row: row["relative_depth"])
                            line, = axis.plot([row["relative_depth"] for row in points], [row[metric] for row in points],
                                              linestyle=style, marker=".", label=f"{comparison}={value}, {name}")
                            if "input_sv" in metric:
                                prefix = metric.removesuffix("50")
                                axis.fill_between([row["relative_depth"] for row in points],
                                                  [row[f"{prefix}10"] for row in points],
                                                  [row[f"{prefix}90"] for row in points],
                                                  color=line.get_color(), alpha=.1)
                    axis.set_title(f"{operator}: {metric}", fontsize=8)
                    axis.set_xlabel("Relative layer depth")
                    axis.grid(alpha=.25)
            axes[2, 0].legend(fontsize=6)
            fig.suptitle(f"{comparison} comparison, step {step}; one matrix per point; input bands: SV quantiles 10–90%")
            fig.tight_layout()
            fig.savefig(out / f"{comparison}-step{step}.png", dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, comparison in enumerate(("width", "depth")):
        for index, metric in enumerate(("training_loss", "validation_loss")):
            selected = [row for row in losses if row["metric"] == metric
                        and (row["depth"] == 6 if comparison == "width" else row["width"] == 88)]
            for value in sorted({row[comparison] for row in selected}):
                points = [row for row in selected if row[comparison] == value]
                axes[index, column].plot([row["step"] for row in points], [row["value"] for row in points], label=f"{comparison}={value}")
            axes[index, column].set_title(f"{comparison}: {metric}")
            axes[index, column].set_xlabel("Training step")
            axes[index, column].legend()
    fig.tight_layout()
    fig.savefig(out / "losses.png", dpi=150)
    plt.close(fig)


def dynamic_pilot_report(weak=False):
    root = ROOT / "results"
    pilot = root / ("dynamic_tikh_weak_pilot" if weak else "dynamic_tikh_pilot")
    out = pilot / "analysis"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    directions = ("weak_to_zero", "zero_to_weak") if weak else ("high_to_zero", "zero_to_high")
    paths = {f"{direction}_after{switch}_seed17": pilot / f"{direction}_after{switch}_seed17"
             for switch in ((100,) if weak else (100, 250)) for direction in directions}
    if weak:
        paths.update({f"{direction}_after100_seed17": root / "dynamic_tikh_pilot" / f"{direction}_after100_seed17"
                      for direction in ("high_to_zero", "zero_to_high")})
    paths.update({f"fixed_{label}": root / "muontikh_lambda_sweep" / name
                  for label, name in (("zero", "tikh_lam0_seed_17"),
                                      ("high", "tikh_lam6p14e-5_seed_17"),
                                      ("weak", "tikh_lam6p89e-6_seed_17"))})
    metadata = {name: json.loads((path / "results.json").read_text()) for name, path in paths.items()}
    reference = metadata["fixed_zero"]
    controls = ("config", "data", "seed", "steps", "batch", "device", "method", "filter_steps",
                "shuffle_buffer", "eval_every", "eval_batches", "muon_lr", "momentum", "adam_lr",
                "torch", "packages")
    high = metadata["fixed_high"]["tikh_lambda"]
    weak_lambda = metadata["fixed_weak"]["tikh_lambda"]
    spectral, counterfactual, losses, final = [], [], [], []
    for name, path in paths.items():
        meta = metadata[name]
        if any(meta[key] != reference[key] for key in controls):
            raise ValueError(f"Unmatched training configuration: {name}")
        if len(meta["losses"]) != 500 or not np.isfinite(meta["losses"]).all():
            raise ValueError(f"Incomplete or nonfinite loss history: {name}")
        snapshots = torch.load(path / "momenta.pt", map_location="cpu", weights_only=True)
        expected = {(step, f"transformer.h.{block}.{operator}.weight")
                    for step in meta["capture"] for block in range(6)
                    for operator in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
        if len(snapshots) != len(expected) or {(s["step"], s["layer"]) for s in snapshots} != expected:
            raise ValueError(f"Incomplete captures: {name}")
        rows = []
        empirical_response(snapshots, meta, rows)
        switch = meta.get("tikh_switch_step")
        for row in rows:
            row.update(run=name, switch_step=switch,
                       switch_boundary=("last_initial_update" if row["step"] == switch else
                                        "first_changed_update" if switch is not None and row["step"] == switch + 1
                                        else ""),
                       parameter_step_frobenius_norm=row["scaled_update_frobenius_norm"] * meta["muon_lr"])
        spectral.extend(rows)
        for snap in snapshots:
            matrix = snap["input"].float()
            zero = tikhonov_via_gram_newtonschulz(matrix, 5, 0)
            regularized = tikhonov_via_gram_newtonschulz(matrix, 5, high)
            weak_output = tikhonov_via_gram_newtonschulz(matrix, 5, weak_lambda)
            input_sv = torch.linalg.svdvals((matrix / (matrix.norm() + 1e-7)).double())
            counterfactual.append(dict(run=name, step=snap["step"], layer=snap["layer"],
                                      active_lambda=tikh_lambda_at_step(meta, snap["step"]),
                                      fraction_input_sv_le_sqrt_high=(input_sv <= high**.5).double().mean().item(),
                                      fraction_input_sv_le_sqrt_weak=(input_sv <= weak_lambda**.5).double().mean().item(),
                                      weak_to_zero_output_norm_ratio=(weak_output.norm() / zero.norm()).item(),
                                      high_to_zero_output_norm_ratio=(regularized.norm() / zero.norm()).item(),
                                      high_zero_update_cosine=torch.nn.functional.cosine_similarity(
                                          regularized.flatten(), zero.flatten(), dim=0).item()))
        validation = {v["step"]: v["loss"] for v in meta["validation_losses"]}
        if sorted(validation) != [100, 200, 300, 400, 500] or not np.isfinite(list(validation.values())).all():
            raise ValueError(f"Incomplete or nonfinite validation: {name}")
        for step, loss in validation.items():
            losses.append(dict(run=name, seed=17, step=step, validation_loss=loss,
                               active_lambda=tikh_lambda_at_step(meta, step),
                               **{f"delta_vs_fixed_{label}": loss - next(v["loss"] for v in
                                  metadata[f"fixed_{label}"]["validation_losses"] if v["step"] == step)
                                  for label in ("zero", "high", "weak")}))
        prefix_exact = None
        if switch is not None:
            baseline = next(key for key in ("fixed_zero", "fixed_high", "fixed_weak")
                            if metadata[key]["tikh_lambda"] == meta["tikh_lambda"])
            prefix_exact = meta["losses"][:switch] == metadata[baseline]["losses"][:switch]
            if not prefix_exact:
                raise ValueError(f"Pre-switch training prefix differs from {baseline}: {name}")
        final.append(dict(run=name, seed=17, switch_step=switch, initial_lambda=meta["tikh_lambda"],
                          final_lambda=tikh_lambda_at_step(meta, 500),
                          final_validation_loss=validation[500], prefix_training_losses_exact=prefix_exact,
                          training_and_eval_seconds=meta["training_and_eval_seconds"],
                          **{f"delta_vs_fixed_{label}": validation[500] -
                             metadata[f"fixed_{label}"]["validation_losses"][-1]["loss"]
                             for label in ("zero", "high", "weak")}))
    switch_metrics = ("fraction_sv_le_0_8", "fraction_sv_le_0_2", "output_sv_median",
                      "input_sv_q50", "parameter_step_frobenius_norm")
    boundaries = []
    operators = sorted({r["operator"] for r in spectral})
    for name, meta in metadata.items():
        switch = meta.get("tikh_switch_step")
        if switch is None:
            continue
        for operator in operators:
            before = [r for r in spectral if r["run"] == name and r["step"] == switch and r["operator"] == operator]
            after = [r for r in spectral if r["run"] == name and r["step"] == switch + 1 and r["operator"] == operator]
            for metric in switch_metrics:
                pre, post = (float(np.mean([r[metric] for r in rr])) for rr in (before, after))
                boundaries.append(dict(run=name, operator=operator, metric=metric,
                                       last_initial_update=switch, first_changed_update=switch + 1,
                                       lambda_before=meta["tikh_lambda"], lambda_after=meta["tikh_lambda_after"],
                                       before=pre, after=post, change=post - pre))
    out.mkdir()
    for filename, rows in (("final-results", final), ("validation-losses", losses),
                           ("spectral-summary", spectral), ("same-input-endpoints", counterfactual),
                           ("switch-effects", boundaries)):
        with (out / f"{filename}.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dynamic_pilot_figures(metadata, spectral, losses, operators, out)
    (out / "sources.json").write_text(json.dumps(dict(
        runs={name: dict(path=str(path), metadata=metadata[name],
                        momenta_sha256=hashlib.sha256((path / "momenta.pt").read_bytes()).hexdigest())
              for name, path in paths.items()},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        switch_convention="Initial lambda through update T; subsequent lambda from update T+1",
        caveats=["Seed-17 exploratory pilot; no replication seeds launched",
                 "Boundary captures use consecutive, different batches; same-input endpoints isolate immediate filter effects",
                 "Spectra precede aspect scaling; parameter-step norms include aspect scaling and learning rate",
                 "Same-input endpoint comparisons are offline diagnostics, not counterfactual loss trajectories"]), indent=2))
    for row in final:
        print(row)
    print(f"Wrote {out}")


def dynamic_pilot_figures(metadata, spectral, losses, operators, out):
    schedules = [name for name, meta in metadata.items() if meta.get("tikh_switch_step") is not None]
    fig, axes = plt.subplots(3, 4, figsize=(21, 13), squeeze=False)
    for column, name in enumerate(schedules):
        meta = metadata[name]
        for run in ("fixed_zero", "fixed_high", "fixed_weak", name):
            label = group_name(metadata[run])
            values = np.asarray(metadata[run]["losses"])
            axes[0, column].plot(np.arange(25, len(values) + 1),
                                 np.convolve(values, np.ones(25) / 25, mode="valid"), label=label)
            rr = [r for r in losses if r["run"] == run]
            axes[1, column].plot([r["step"] for r in rr], [r["validation_loss"] for r in rr], marker=".", label=label)
        rr = [r for r in losses if r["run"] == name]
        for label in ("zero", "high", "weak"):
            axes[2, column].plot([r["step"] for r in rr], [r[f"delta_vs_fixed_{label}"] for r in rr],
                                 marker=".", label=f"vs fixed {label}")
        axes[2, column].axhline(0, color="black", linewidth=.8)
        axes[0, column].set_title(group_name(meta))
        for row, ylabel in enumerate(("Training loss (25-step mean)", "Validation loss", "Validation loss difference")):
            axis = axes[row, column]
            axis.axvline(meta["tikh_switch_step"] + .5, color="gray", linestyle=":")
            axis.set(xlabel="Training step", ylabel=ylabel)
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "loss-trajectories.png", dpi=150)
    plt.close(fig)
    metrics = ("fraction_sv_le_0_8", "fraction_sv_le_0_2", "input_sv_q50", "parameter_step_frobenius_norm")
    fig, axes = plt.subplots(4, 4, figsize=(21, 17), squeeze=False, sharey="row")
    for column, name in enumerate(schedules):
        switch = metadata[name]["tikh_switch_step"]
        axes[0, column].set_title(group_name(metadata[name]))
        for index, metric in enumerate(metrics):
            axis = axes[index, column]
            for color, operator in enumerate(operators):
                for step, style, phase in ((switch, "--o", "last initial"), (switch + 1, "-x", "first changed")):
                    rr = sorted((r for r in spectral if r["run"] == name and r["operator"] == operator
                                 and r["step"] == step), key=lambda r: r["block"])
                    axis.plot([r["block"] for r in rr], [r[metric] for r in rr], style,
                              color=f"C{color}", label=f"{operator}: {phase}, step {step}")
            axis.set(xlabel="Block (zero indexed)", ylabel=metric)
            if index >= 2:
                axis.set_yscale("log")
            axis.grid(alpha=.2)
            if index == 0:
                axis.legend(fontsize=6)
    fig.suptitle("Switch boundaries: dashed = last initial-lambda update; solid = first changed-lambda update")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out / "switch-boundaries.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
    norm_max = max(r["parameter_step_frobenius_norm"] for r in spectral)
    for column, name in enumerate(schedules):
        steps = metadata[name]["capture"]
        for index, metric in enumerate(("fraction_sv_le_0_8", "parameter_step_frobenius_norm")):
            values = [[np.mean([r[metric] for r in spectral if r["run"] == name
                                and r["step"] == step and r["block"] == block]) for step in steps]
                      for block in range(6)]
            axis = axes[index, column]
            mesh = axis.imshow(values, aspect="auto", origin="lower", vmin=0,
                               vmax=1 if index == 0 else norm_max)
            axis.axvline(steps.index(metadata[name]["tikh_switch_step"]) + .5,
                         color="white", linestyle=":")
            axis.set(xticks=range(len(steps)), xticklabels=steps, xlabel="Capture step (uneven spacing)",
                     ylabel="Block; mean across four matrix types", title=f"{group_name(metadata[name])}\n{metric}")
            fig.colorbar(mesh, ax=axis)
    fig.tight_layout()
    fig.savefig(out / "stage-by-layer.png", dpi=150)
    plt.close(fig)


def switch_loss_report(weak=False):
    root = ROOT / "results"
    pilot = root / ("dynamic_tikh_weak_pilot" if weak else "dynamic_tikh_pilot")
    out = pilot / "analysis"
    outputs = [out / f"switch-loss{suffix}" for suffix in (".png", ".csv", "-summary.csv")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite switch-loss analysis")
    points, summaries = [], []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    directions = ("weak_to_zero", "zero_to_weak") if weak else ("high_to_zero", "zero_to_high")
    paths = {f"{direction}_after{switch}_seed17": pilot / f"{direction}_after{switch}_seed17"
             for switch in ((100,) if weak else (100, 250)) for direction in directions}
    if weak:
        paths.update({f"{direction}_after100_seed17": root / "dynamic_tikh_pilot" / f"{direction}_after100_seed17"
                      for direction in ("high_to_zero", "zero_to_high")})
    for index, (name, path) in enumerate(paths.items()):
        meta = json.loads((path / "results.json").read_text())
        switch = meta["tikh_switch_step"]
        fixed_name = ("tikh_lam0_seed_17" if meta["tikh_lambda"] == 0 else
                      "tikh_lam6p89e-6_seed_17" if meta["tikh_lambda"] == 6.8851809e-6
                      else "tikh_lam6p14e-5_seed_17")
        baseline = ROOT / "results/muontikh_lambda_sweep" / fixed_name
        fixed = json.loads((baseline / "results.json").read_text())
        controls = ("config", "data", "seed", "steps", "batch", "shuffle_buffer",
                    "method", "filter_steps", "tikh_lambda", "muon_lr", "momentum", "adam_lr")
        if any(meta[key] != fixed[key] for key in controls):
            raise ValueError(f"Unmatched fixed baseline: {name}")
        values, reference = np.asarray(meta["losses"]), np.asarray(fixed["losses"])
        if values.shape != (500,) or reference.shape != (500,) or not np.isfinite([values, reference]).all():
            raise ValueError(f"Incomplete or nonfinite loss history: {name}")
        delta = values - reference
        # Forward loss precedes the update: the first changed weights are evaluated at T+2.
        if np.any(delta[:switch + 1] != 0):
            raise ValueError(f"Loss differs before switched weights can be evaluated: {name}")
        first = switch + 2
        summary = dict(run=name, baseline=fixed_name, last_initial_update=switch,
                       first_changed_update=switch + 1, first_affected_loss_step=first,
                       raw_loss_change_at_first_changed_update=float(values[switch] - values[switch - 1]),
                       baseline_raw_loss_change_same_step=float(reference[switch] - reference[switch - 1]),
                       first_affected_loss_delta=float(delta[first - 1]))
        for count in (5, 10, 25, 50):
            window = delta[first - 1:first - 1 + count]
            summary[f"mean_delta_first_{count}_affected_batches"] = float(window.mean())
            summary[f"positive_deltas_first_{count}_affected_batches"] = int((window > 0).sum())
        summaries.append(summary)
        for step in range(switch - 9, switch + 52):
            points.append(dict(run=name, baseline=fixed_name, step=step,
                               steps_since_first_changed_update=step - switch - 1,
                               weights_affected_by_switch=step >= first,
                               training_loss=float(values[step - 1]), fixed_training_loss=float(reference[step - 1]),
                               loss_delta=float(delta[step - 1])))
        axis = axes.flat[index]
        rr = [r for r in points if r["run"] == name]
        axis.plot([r["steps_since_first_changed_update"] for r in rr],
                  [r["loss_delta"] for r in rr], color="C0", alpha=.6, linewidth=1, label="Per-batch difference")
        affected = delta[first - 1:switch + 51]
        axis.plot(np.arange(5, 51), np.convolve(affected, np.ones(5) / 5, mode="valid"),
                  color="C1", linewidth=2, label="Trailing 5 affected batches")
        axis.axhline(0, color="black", linewidth=.8)
        axis.axvline(0, color="gray", linestyle=":", label="First changed update")
        axis.axvspan(1, 5, color="gray", alpha=.1)
        axis.set(title=f"{group_name(meta)}; baseline: fixed initial λ",
                 xlabel="Steps since first changed update (first affected loss at +1)",
                 ylabel="Training loss − matched fixed loss (nats/token)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Local switch response on matched training batches; shaded = first five affected losses")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(outputs[0], dpi=160)
    plt.close(fig)
    for path, rows in ((outputs[1], points), (outputs[2], summaries)):
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in summaries:
        print(row)
    print(f"Wrote {outputs[0]}")


def weak_replication_report():
    root = ROOT / "results"
    pilot = root / "dynamic_tikh_weak_pilot"
    out = pilot / "replication_analysis"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    paths = {}
    for seed in (17, 18, 19):
        paths[seed, "fixed_zero"] = (root / "muontikh_lambda_sweep/tikh_lam0_seed_17" if seed == 17
                                      else root / f"depth_filter_pilot/w88_d6_tikh_lam0_seed{seed}")
        paths[seed, "fixed_weak"] = (root / "muontikh_lambda_sweep/tikh_lam6p89e-6_seed_17" if seed == 17
                                      else pilot / f"fixed_weak_seed{seed}")
        paths[seed, "weak_to_zero"] = pilot / f"weak_to_zero_after100_seed{seed}"
    metadata = {key: json.loads((path / "results.json").read_text()) for key, path in paths.items()}
    reference = metadata[17, "fixed_zero"]
    controls = ("config", "data", "steps", "batch", "device", "method", "filter_steps", "shuffle_buffer",
                "eval_every", "eval_batches", "muon_lr", "momentum", "adam_lr", "torch", "packages")
    weak = 6.8851809e-6
    spectral, endpoints = [], []
    for (seed, setting), path in paths.items():
        meta = metadata[seed, setting]
        expected_lambda = 0 if setting == "fixed_zero" else weak
        expected_switch = 100 if setting == "weak_to_zero" else None
        if (meta["seed"] != seed or meta["tikh_lambda"] != expected_lambda
                or meta.get("tikh_switch_step") != expected_switch
                or meta.get("tikh_lambda_after") != (0 if expected_switch else None)
                or any(meta[k] != reference[k] for k in controls)):
            raise ValueError(f"Unmatched replication configuration: {path}")
        for source in ("models/model.py", "utils/muon.py", "utils/data.py", "utils/capture.py"):
            if meta["source_sha256"][source] != reference["source_sha256"][source]:
                raise ValueError(f"Unmatched core source: {path}, {source}")
        if (len(meta["losses"]) != 500 or not np.isfinite(meta["losses"]).all()
                or [v["step"] for v in meta["validation_losses"]] != [100, 200, 300, 400, 500]
                or not np.isfinite([v["loss"] for v in meta["validation_losses"]]).all()):
            raise ValueError(f"Incomplete/nonfinite training history: {path}")
        snapshots = torch.load(path / "momenta.pt", map_location="cpu", weights_only=True)
        expected = {(step, f"transformer.h.{block}.{operator}.weight")
                    for step in meta["capture"] for block in range(6)
                    for operator in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
        if len(snapshots) != len(expected) or {(s["step"], s["layer"]) for s in snapshots} != expected:
            raise ValueError(f"Incomplete captures: {path}")
        rows = []
        empirical_response(snapshots, meta, rows)
        for row in rows:
            row.update(setting=setting,
                       parameter_step_frobenius_norm=row["scaled_update_frobenius_norm"] * meta["muon_lr"])
        spectral.extend(rows)
        for snap in snapshots:
            zero = tikhonov_via_gram_newtonschulz(snap["input"], 5, 0)
            regularized = tikhonov_via_gram_newtonschulz(snap["input"], 5, weak)
            endpoints.append(dict(seed=seed, setting=setting, step=snap["step"], layer=snap["layer"],
                                  weak_zero_norm_ratio=(regularized.norm() / zero.norm()).item(),
                                  norm_matched_shape_difference=(regularized / regularized.norm()
                                                                - zero / zero.norm()).norm().item()))
    validation, final = [], []
    for seed in (17, 18, 19):
        scheduled, fixed = metadata[seed, "weak_to_zero"], metadata[seed, "fixed_weak"]
        if scheduled["losses"][:101] != fixed["losses"][:101]:
            raise ValueError(f"Pre-switch training prefix mismatch for seed {seed}")
        if scheduled["validation_losses"][0] != fixed["validation_losses"][0]:
            raise ValueError(f"Pre-switch validation mismatch for seed {seed}")
        for index, step in enumerate((100, 200, 300, 400, 500)):
            row = dict(seed=seed, step=step,
                       **{setting: metadata[seed, setting]["validation_losses"][index]["loss"]
                          for setting in ("fixed_zero", "fixed_weak", "weak_to_zero")})
            row.update(improvement_vs_fixed_zero=row["fixed_zero"] - row["weak_to_zero"],
                       improvement_vs_fixed_weak=row["fixed_weak"] - row["weak_to_zero"])
            validation.append(row)
        final.append(validation[-1].copy())
    aggregate = []
    for label, seeds in (("all_three_including_discovery", (17, 18, 19)), ("replication_only", (18, 19))):
        selected = [r for r in final if r["seed"] in seeds]
        row = dict(group=label, seeds=",".join(map(str, seeds)),
                   wins_against_both=sum(r["improvement_vs_fixed_zero"] > 0
                                         and r["improvement_vs_fixed_weak"] > 0 for r in selected))
        for baseline in ("fixed_zero", "fixed_weak"):
            values = [r[f"improvement_vs_{baseline}"] for r in selected]
            row.update({f"mean_improvement_vs_{baseline}": float(np.mean(values)),
                        f"sample_sd_improvement_vs_{baseline}": float(np.std(values, ddof=1)),
                        f"wins_vs_{baseline}": sum(v > 0 for v in values)})
        aggregate.append(row)
    out.mkdir()
    for filename, rows in (("final-results", final), ("validation-trajectories", validation),
                           ("aggregate-results", aggregate), ("spectral-summary", spectral),
                           ("same-input-endpoints", endpoints)):
        with (out / f"{filename}.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    settings = ("fixed_zero", "fixed_weak", "weak_to_zero")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False, sharey="row")
    for column, seed in enumerate((17, 18, 19)):
        rr = [r for r in validation if r["seed"] == seed]
        for setting in settings:
            axes[0, column].plot([r["step"] for r in rr], [r[setting] for r in rr], marker=".", label=setting)
        for baseline in settings[:2]:
            axes[1, column].plot([r["step"] for r in rr], [r[f"improvement_vs_{baseline}"] for r in rr],
                                  marker=".", label=f"{baseline} − schedule")
        axes[0, column].set_title(f"Seed {seed}" + (" (discovery)" if seed == 17 else " (replication)"))
        axes[1, column].axhline(0, color="black", linewidth=.8)
        for index, ylabel in enumerate(("Validation loss", "Improvement; positive favors schedule")):
            axes[index, column].axvline(100.5, color="gray", linestyle=":")
            axes[index, column].set(xlabel="Training step", ylabel=ylabel)
            axes[index, column].grid(alpha=.2)
            axes[index, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "validation-trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False, sharey="row")
    for column, seed in enumerate((17, 18, 19)):
        axes[0, column].set_title(f"Seed {seed}")
        for setting in settings:
            for index, (rows, metric) in enumerate(((spectral, "parameter_step_frobenius_norm"),
                                                     (endpoints, "weak_zero_norm_ratio"),
                                                     (endpoints, "norm_matched_shape_difference"))):
                rr = [r for r in rows if r["seed"] == seed and r["setting"] == setting]
                steps = sorted({r["step"] for r in rr})
                axes[index, column].plot(steps, [np.mean([r[metric] for r in rr if r["step"] == step])
                                                for step in steps], marker=".", label=setting)
        for index, ylabel in enumerate(("Actual parameter-step norm (mean per matrix)",
                                        "Same-input weak/zero norm ratio",
                                        "Shape difference after matching Frobenius norms")):
            axes[index, column].axvline(100.5, color="gray", linestyle=":")
            axes[index, column].set(xlabel="Capture step", ylabel=ylabel)
            axes[index, column].grid(alpha=.2)
            axes[index, column].legend(fontsize=8)
    fig.suptitle("Norm and shape diagnostics: offline endpoint comparisons use identical inputs; curves identify input source")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "update-norms.png", dpi=160)
    plt.close(fig)
    (out / "sources.json").write_text(json.dumps(dict(
        runs={f"{seed}/{setting}": dict(path=str(path), metadata=metadata[seed, setting],
                                       momenta_sha256=hashlib.sha256((path / "momenta.pt").read_bytes()).hexdigest())
              for (seed, setting), path in paths.items()},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=["Seed 17 selected the schedule; seeds 18 and 19 are replication",
                 "Positive improvement is baseline loss minus schedule loss",
                 "Fixed-zero baselines and seed-17 fixed weak lack captures at 101/251; no values are imputed",
                 "Norm and shape statistics are diagnostics; no norm-matched training control was run",
                 "Switch uses W through update 100 and zero from 101; first affected training loss is step 102"]), indent=2))
    for row in final + aggregate:
        print(row)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int,
                        help="Plot only this seed, for a matched single-seed comparison")
    parser.add_argument("--depth-pilot", action="store_true",
                        help="Compare the approved depth 6/8, lambda 0/6.1352038e-5 pilot")
    parser.add_argument("--architecture", action="store_true",
                        help="Analyze the five-model width/depth grid on identical saved inputs")
    parser.add_argument("--dynamic-pilot", action="store_true",
                        help="Compare the four seed-17 switching runs and three fixed references")
    parser.add_argument("--weak-dynamic-pilot", action="store_true",
                        help="Analyze the two weak-lambda switches, reusing the high-lambda step-100 pilots")
    parser.add_argument("--weak-replication", action="store_true",
                        help="Compare fixed zero, fixed weak, and weak-to-zero across seeds 17/18/19")
    parser.add_argument("--switch-loss", action="store_true",
                        help="Analyze saved training losses around each switch against its fixed initial lambda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.weak_replication:
        if args.switch_loss or args.weak_dynamic_pilot or args.dynamic_pilot or args.architecture or args.depth_pilot or args.seed is not None:
            parser.error("Weak replication uses all three seeds and cannot be combined with other modes")
        weak_replication_report()
        return
    if args.weak_dynamic_pilot:
        if args.switch_loss or args.dynamic_pilot or args.architecture or args.depth_pilot or args.seed not in (None, 17):
            parser.error("Weak dynamic pilot uses seed 17 and cannot be combined with other modes")
        dynamic_pilot_report(weak=True)
        switch_loss_report(weak=True)
        return
    if args.switch_loss:
        if args.dynamic_pilot or args.architecture or args.depth_pilot or args.seed not in (None, 17):
            parser.error("Switch-loss analysis uses seed 17 and cannot be combined with other modes")
        switch_loss_report()
        return
    if args.dynamic_pilot:
        if args.architecture or args.depth_pilot or args.seed not in (None, 17):
            parser.error("Dynamic pilot uses seed 17 and cannot be combined with other modes")
        dynamic_pilot_report()
        return
    if args.architecture:
        if args.depth_pilot or args.seed not in (None, 17):
            parser.error("Architecture mode uses seed 17 and cannot be combined with --depth-pilot")
        architecture_report()
        return
    if args.depth_pilot:
        depth_pilot(17 if args.seed is None else args.seed)
        return
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
