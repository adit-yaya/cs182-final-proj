"""Reviewer entry point for the width-256 experiments."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from run_w256 import BPE, NS_PE, TIKH, run
from train import ROOT, main as train

STEPS = [1, 10, 50, 100, 250, 300, 350, 400, 450, 500]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("demo", "figures", "train"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    if args.mode == "demo":
        with tempfile.TemporaryDirectory() as dest:
            directory = Path(dest) / "demo"
            train(["--tiny", "--bpe", str(BPE), "--steps", "3", "--width", "32",
                   "--layers", "1", "--heads", "2", "--context", "8", "--batch", "1",
                   "--capture", "1", "3", "--spectra", "--eval-every", "1",
                   "--eval-batches", "1", "--out", str(directory)])
            report = directory / "test.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/test.py"), "--checkpoint",
                            str(directory / "checkpoint.pt"), "--batch", "1", "--batches", "1",
                            "--out", str(report)], check=True, stdout=subprocess.DEVNULL)
            import json
            print("Held-out test loss:", json.loads(report.read_text())["loss"])
        return
    from reviewer_analysis import analyze
    if args.mode == "train":
        run({**NS_PE, **TIKH}, create_tokenizer=False, out=args.out, capture=STEPS)
    analyze(args.out, use_runs=args.mode == "train")


if __name__ == "__main__":
    main()
