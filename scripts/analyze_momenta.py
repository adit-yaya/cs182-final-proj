"""Measure spectral attenuation in saved momentum inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.muon import zeropower_via_newtonschulz5
from utils.polar_express import PolarExpress, coeffs_list

COEFFICIENTS = {"NS5": [(3.4445, -4.775, 2.0315)] * 5,
                "PE3": coeffs_list[:3], "PE5": coeffs_list[:5]}


def normalize(matrix, method):
    x = matrix.bfloat16()
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    factor = 1 if method == "NS5" else 1.01
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * factor + 1e-7)
    return x.mT if transpose else x


def orthogonalize(matrix, method):
    if method == "NS5":
        return zeropower_via_newtonschulz5(matrix, 5)
    return PolarExpress(matrix, len(COEFFICIENTS[method]))


@torch.no_grad()
def analyze(snapshots, out):
    # Analyze on CPU in FP64; matrix transforms retain the pilot's BF16 arithmetic.
    with (out / "spectra.csv").open("w", newline="") as f, \
            (out / "attenuation.csv").open("w", newline="") as g:
        points, summary = csv.writer(f), csv.writer(g)
        points.writerow(["step", "layer", "method", "rank", "normalized_input_sv",
                         "polynomial_signed_output", "bf16_directional_projection"])
        summary.writerow(["step", "layer", "method", "directions",
                          "poly_fraction_le_0.8", "bf16_sv_fraction_le_0.8"])
        for snap in snapshots:
            matrix = snap["input"].cpu()
            for method, coefficients in COEFFICIENTS.items():
                u, s, vh = torch.linalg.svd(normalize(matrix, method).double(),
                                           full_matrices=False)
                response = s.clone()
                for a, b, c in coefficients:
                    response = a * response + b * response**3 + c * response**5
                output = orthogonalize(matrix, method).double()
                projection = (u.T @ output @ vh.T).diagonal()
                actual_sv = torch.linalg.svdvals(output)
                prefix = [snap["step"], snap["layer"], method]
                points.writerows(prefix + [i, x, y, z] for i, (x, y, z) in enumerate(
                    zip(s.tolist(), response.tolist(), projection.tolist()), 1))
                summary.writerow(prefix + [len(s), (response.abs() <= 0.8).double().mean().item(),
                                           (actual_sv <= 0.8).double().mean().item()])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--momenta", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/spectral-analysis/analysis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    snapshots = torch.load(args.momenta, map_location="cpu", weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = dict(momenta_source=str(args.momenta.resolve()),
                    momenta_sha256=hashlib.sha256(args.momenta.read_bytes()).hexdigest(),
                    torch=str(torch.__version__), coefficients=COEFFICIENTS,
                    reference="Unit output for positive input singular values; before aspect-ratio scaling",
                    metric="Equal direction counts, not energy weighted; BF16 projections are signed, not output SVs",
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [ROOT / "utils/muon.py", ROOT / "utils/polar_express.py", Path(__file__).resolve()]})
    (args.out / "results.json").write_text(json.dumps(metadata, indent=2))
    analyze(snapshots, args.out)


if __name__ == "__main__":
    main()
