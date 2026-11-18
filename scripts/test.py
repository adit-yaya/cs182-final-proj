"""Report checkpoint loss and perplexity on held-out test data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.data import batches, evaluate
from models.model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, help="Optional JSON report path; refuses overwrite")
    args = parser.parse_args()
    if min(args.batch, args.batches) < 1:
        parser.error("batch and batches must be positive")
    if args.out and args.out.exists():
        parser.error("output report already exists")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    source = batches(checkpoint["data"], "test", args.batch, model.config.block_size)
    metrics = evaluate(model, source, args.batches, args.device)
    metrics.update(split="test", data=checkpoint["data"], step=checkpoint["step"],
                   checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   torch=str(torch.__version__), device=args.device)
    report = json.dumps(metrics, indent=2)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
