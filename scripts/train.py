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
from utils.muon import (SingleDeviceMuon, exact_tikhonov_polar, tikhonov_via_gram_newtonschulz,
                        zeropower_via_newtonschulz5)
from utils.capture import capture_inputs
from utils.polar_express import PolarExpress
from models.model import GPT, GPTConfig


def train(args, data):
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    vocab_size = data.get("vocab_size") or tiktoken.get_encoding(data["tokenizer"]).n_vocab
    config = GPTConfig(vocab_size,
                       args.context, args.layers, args.heads, args.width)
    model = GPT(config).to(args.device)
    hidden = [(n, p) for n, p in model.named_parameters()
              if n.startswith("transformer.h.") and p.ndim == 2]
    hidden_ids = {id(p) for _, p in hidden}
    active_lambda = args.tikh_lambda
    if args.method == "ns":
        orthogonalizer = zeropower_via_newtonschulz5
    elif args.method == "pe":
        orthogonalizer = PolarExpress
    elif args.method == "tikh-exact":
        orthogonalizer = lambda matrix, steps: exact_tikhonov_polar(matrix, active_lambda)
    else:
        orthogonalizer = lambda matrix, steps: tikhonov_via_gram_newtonschulz(
            matrix, steps, active_lambda)
    muon = SingleDeviceMuon([p for _, p in hidden], lr=0.02, momentum=0.95,
                            ns_steps=args.filter_steps, orthogonalizer=orthogonalizer)
    adam = torch.optim.AdamW([p for p in model.parameters() if id(p) not in hidden_ids],
                             lr=0.003, weight_decay=0)
    training = iter(batches(data, "train", args.batch, args.context,
                            args.seed, args.shuffle_buffer))
    # Cache only the evaluation budget on CPU; reuse identical validation tokens.
    print("Loading fixed validation batches...", flush=True)
    validation_start = time.perf_counter()
    validation = list(islice(batches(data, "validation", args.batch, args.context), args.eval_batches))
    if len(validation) != args.eval_batches:
        raise ValueError("Validation stream is shorter than the requested evaluation budget.")
    print(f"Validation batches loaded in {time.perf_counter() - validation_start:.1f}s", flush=True)
    snapshots, losses, validation_losses = [], [], []
    train_digest = hashlib.sha256()
    val_digest = hashlib.sha256()
    for x, y in validation:
        for tensor in (x, y):
            val_digest.update(tensor.numpy().astype("<i8", copy=False).tobytes())
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        active_lambda = (args.tikh_lambda_after
                         if args.tikh_switch_step is not None and step > args.tikh_switch_step
                         else args.tikh_lambda)
        phase_start = time.perf_counter()
        if step == 1:
            print("step=1 fetching training batch...", flush=True)
        try:
            x, y = next(training)
        except StopIteration as error:
            raise ValueError("Training stream exhausted; reduce steps or use a larger subset.") from error
        if step == 1:
            print(f"step=1 training batch loaded in {time.perf_counter() - phase_start:.1f}s", flush=True)
        for tensor in (x, y):
            train_digest.update(tensor.numpy().astype("<i8", copy=False).tobytes())
        muon.zero_grad()
        adam.zero_grad()
        phase_start = time.perf_counter()
        if step == 1:
            print("step=1 forward/backward...", flush=True)
        _, loss = model(x.to(args.device), y.to(args.device))
        loss.backward()
        if step == 1:
            print(f"step=1 forward/backward finished in {time.perf_counter() - phase_start:.1f}s", flush=True)
            print("step=1 Muon optimizer step (includes PE compilation if enabled)...", flush=True)
        phase_start = time.perf_counter()
        if step in args.capture:
            with capture_inputs(hidden, step, snapshots, muon, spectra=args.spectra):
                muon.step()
            if args.method in ("tikh", "tikh-exact"):
                for snapshot in snapshots[-len(hidden):]:
                    snapshot["tikh_lambda"] = active_lambda
        else:
            muon.step()
        if step == 1:
            print(f"step=1 Muon optimizer step finished in {time.perf_counter() - phase_start:.1f}s", flush=True)
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
                method=args.method, filter_steps=None if args.method == "tikh-exact" else args.filter_steps,
                tikh_lambda=args.tikh_lambda if args.method in ("tikh", "tikh-exact") else None,
                tikh_lambda_after=args.tikh_lambda_after,
                tikh_switch_step=args.tikh_switch_step,
                pe_compile=args.pe_compile if args.method == "pe" else None,
                shuffle_buffer=args.shuffle_buffer, eval_every=args.eval_every,
                eval_batches=args.eval_batches, muon_lr=0.02, momentum=0.95, adam_lr=0.003,
                parameters=sum(p.numel() for p in model.parameters()),
                muon_parameters=sum(p.numel() for _, p in hidden),
                adam_parameters=sum(p.numel() for p in model.parameters()
                                    if id(p) not in hidden_ids),
                training_and_eval_seconds=time.perf_counter() - start,
                train_sha256=train_digest.hexdigest(), validation_sha256=val_digest.hexdigest(),
                training_tokens=args.steps * args.batch * args.context,
                losses=losses, validation_losses=validation_losses)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="sample-10BT", help="FineWeb config; default selects its 10B-token sample")
    parser.add_argument("--revision", default="main", help="Resolved to a commit before streaming")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to results/train_<method><steps>")
    parser.add_argument("--method", choices=["ns", "pe", "tikh", "tikh-exact"], default="ns")
    parser.add_argument("--filter-steps", type=int, default=5)
    parser.add_argument("--tikh-lambda", type=float, default=0.0)
    parser.add_argument("--tikh-lambda-after", type=float,
                        help="Lambda for updates strictly after --tikh-switch-step")
    parser.add_argument("--tikh-switch-step", type=int,
                        help="Last update using the initial lambda; keeps optimizer state")
    parser.add_argument("--pe-compile", choices=["inductor", "eager"], default="inductor",
                        help="PE execution mode; eager bypasses torch.compile/Inductor")
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
    parser.add_argument("--bpe", type=Path, help="Saved byte-level BPE tokenizer JSON")
    parser.add_argument("--spectra", action="store_true", help="Log singular spectra at captures")
    parser.add_argument("--tiny", action="store_true", help="Use the bundled train, validation and test documents")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.context, args.layers, args.width, args.heads,
           args.shuffle_buffer, args.eval_every, args.eval_batches) < 1:
        parser.error("dimensions, batch counts, intervals, and buffer size must be positive")
    if args.filter_steps < 1:
        parser.error("filter steps must be positive")
    if args.tikh_lambda < 0:
        parser.error("Tikhonov lambda must be nonnegative")
    if (args.tikh_switch_step is None) != (args.tikh_lambda_after is None):
        parser.error("--tikh-switch-step and --tikh-lambda-after must be supplied together")
    if args.tikh_switch_step is not None:
        if args.method != "tikh" or not 1 <= args.tikh_switch_step < args.steps:
            parser.error("Tikh switching requires --method tikh and 1 <= switch step < steps")
        if args.tikh_lambda_after < 0:
            parser.error("Tikhonov lambda after switching must be nonnegative")
    if args.width % args.heads:
        parser.error("width must be divisible by heads")
    if not any(1 <= step <= args.steps for step in args.capture):
        parser.error("at least one capture step must be within training")
    if args.out is None:
        suffix = f"_lam{args.tikh_lambda}" if args.method == "tikh-exact" else args.filter_steps
        args.out = ROOT / "results" / f"train_{args.method}{suffix}"
    if args.method == "pe" and args.pe_compile == "eager":
        torch.compiler.set_stance("force_eager")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    print("Resolving FineWeb revision...", flush=True)
    revision_start = time.perf_counter()
    revision = "local" if args.tiny else HfApi().dataset_info(DATASET, revision=args.revision).sha
    print(f"FineWeb revision resolved in {time.perf_counter() - revision_start:.1f}s", flush=True)
    data = dict(dataset=DATASET, subset=args.subset, revision=revision,
                tokenizer="gpt2", split_policy=SPLIT_POLICY)
    if args.tiny:
        data.update(dataset="bundled-tiny", split_policy="fixed-files")
    if args.bpe:
        from tokenizers import Tokenizer
        raw = args.bpe.read_text()
        data.update(tokenizer="fineweb-bpe", bpe_json=raw,
                    vocab_size=Tokenizer.from_str(raw).get_vocab_size())
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = train(args, data)
    metadata.update(torch=str(torch.__version__),
                    packages={name: version(name) for name in ("datasets", "tiktoken", "huggingface_hub")},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [ROOT / "models/model.py", ROOT / "utils/muon.py",
                                             ROOT / "utils/polar_express.py",
                                             ROOT / "utils/data.py", ROOT / "utils/capture.py",
                                             Path(__file__).resolve()]})
    (args.out / "results.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
