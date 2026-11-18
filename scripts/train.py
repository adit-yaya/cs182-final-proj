"""Train on FineWeb and save the model and momentum inputs."""

import argparse
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import HfApi
import tiktoken
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.data import DATASET, SPLIT_POLICY, batches, evaluate
from utils.muon import SingleDeviceMuon
from utils.capture import capture_inputs
from models.model import GPT, GPTConfig


def train(args, data):
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    config = GPTConfig(tiktoken.get_encoding(data["tokenizer"]).n_vocab,
                       args.context, args.layers, args.heads, args.width)
    model = GPT(config).to(args.device)
    hidden = [(n, p) for n, p in model.named_parameters()
              if n.startswith("transformer.h.") and p.ndim == 2]
    hidden_ids = {id(p) for _, p in hidden}
    muon = SingleDeviceMuon([p for _, p in hidden], lr=0.02, momentum=0.95)
    adam = torch.optim.AdamW([p for p in model.parameters() if id(p) not in hidden_ids],
                             lr=0.003, weight_decay=0)
    training = iter(batches(data, "train", args.batch, args.context,
                            args.seed, args.shuffle_buffer))
    # Cache only the evaluation budget on CPU; reuse identical validation tokens.
    print("Loading fixed validation batches...", flush=True)
    validation = list(islice(batches(data, "validation", args.batch, args.context), args.eval_batches))
    if len(validation) != args.eval_batches:
        raise ValueError("Validation stream is shorter than the requested evaluation budget.")
    snapshots, losses, validation_losses = [], [], []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(training)
        except StopIteration as error:
            raise ValueError("Training stream exhausted; reduce steps or use a larger subset.") from error
        muon.zero_grad()
        adam.zero_grad()
        _, loss = model(x.to(args.device), y.to(args.device))
        loss.backward()
        if step in args.capture:
            with capture_inputs(hidden, step, snapshots):
                muon.step()
        else:
            muon.step()
        adam.step()
        losses.append(loss.item())
        if step % 25 == 0:
            print(f"step={step} train_loss={loss.item():.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation, args.eval_batches, args.device)
            validation_losses.append(dict(step=step, **metrics))
            print(f"step={step} validation_loss={metrics['loss']:.4f}", flush=True)
    torch.save(snapshots, args.out / "momenta.pt")
    torch.save(dict(model=model.state_dict(), config=vars(config), data=data,
                    step=args.steps, seed=args.seed), args.out / "checkpoint.pt")
    return dict(config=vars(config), data=data, seed=args.seed, steps=args.steps,
                batch=args.batch, capture=args.capture, device=args.device,
                shuffle_buffer=args.shuffle_buffer, eval_every=args.eval_every,
                eval_batches=args.eval_batches, muon_lr=0.02, momentum=0.95, adam_lr=0.003,
                parameters=sum(p.numel() for p in model.parameters()),
                training_and_eval_seconds=time.perf_counter() - start,
                training_tokens=args.steps * args.batch * args.context,
                losses=losses, validation_losses=validation_losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="sample-10BT", help="FineWeb config; default selects its 10B-token sample")
    parser.add_argument("--revision", default="main", help="Resolved to a commit before streaming")
    parser.add_argument("--out", type=Path, default=ROOT / "results/train")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=88)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--capture", type=int, nargs="+", default=[1, 10, 50, 100, 250, 500])
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.context, args.layers, args.width, args.heads,
           args.shuffle_buffer, args.eval_every, args.eval_batches) < 1:
        parser.error("dimensions, batch counts, intervals, and buffer size must be positive")
    if args.width % args.heads:
        parser.error("width must be divisible by heads")
    if not any(1 <= step <= args.steps for step in args.capture):
        parser.error("at least one capture step must be within training")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    data = dict(dataset=DATASET, subset=args.subset, revision=revision,
                tokenizer="gpt2", split_policy=SPLIT_POLICY)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = train(args, data)
    metadata.update(torch=str(torch.__version__),
                    packages={name: version(name) for name in ("datasets", "tiktoken", "huggingface_hub")},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [ROOT / "models/model.py", ROOT / "utils/muon.py",
                                             ROOT / "utils/data.py", ROOT / "utils/capture.py", Path(__file__).resolve()]})
    (args.out / "results.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
