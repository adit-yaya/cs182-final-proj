"""Run the paired FineWeb BPE experiments from the repository root."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from train import ROOT, main as train
from utils.data import DATASET, documents

OUT = ROOT / "results/w256_bpe"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
BPE = OUT / "tokenizer.json"


def training_text():
    remaining = 10_000_000
    data = dict(dataset=DATASET, subset="sample-10BT", revision=REVISION)
    for row in documents(data, "train"):
        text = row["text"][:remaining]
        yield text
        remaining -= len(text)
        if remaining == 0:
            return
    raise ValueError("Tokenizer training stream ended before the character budget")


def prepare_bpe():
    encoder = Tokenizer(models.BPE())
    encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=4096, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<|endoftext|>"])
    encoder.train_from_iterator(training_text(), trainer)
    if encoder.get_vocab_size() != 4096:
        raise ValueError("Expected 4096 vocabulary entries")
    temp = BPE.with_suffix(".tmp")
    encoder.save(str(temp))
    temp.replace(BPE)


def plot_transfer(directory):
    snapshots = torch.load(directory / "momenta.pt", map_location="cpu", weights_only=True)
    x = torch.cat([s["normalized_sv"] for s in snapshots])
    y = torch.cat([s["projection"].abs() for s in snapshots])
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, alpha=0.12, rasterized=True)
    ax.set(xscale="log", xlabel="Frobenius-normalized input singular value",
           ylabel="Absolute directional response", title=directory.name)
    fig.tight_layout()
    fig.savefig(directory / "transfer.png")
    plt.close(fig)


NS_PE = {"ns5": ["--method", "ns", "--filter-steps", "5"],
         "pe3": ["--method", "pe", "--filter-steps", "3"],
         "pe5": ["--method", "pe", "--filter-steps", "5"]}
TIKH = {f"tikh_{lam}": ["--method", "tikh-exact", "--tikh-lambda", lam]
        for lam in ("1e-6", "1e-4", "1e-2", "0.1", "0")}


def run(conditions, *, create_tokenizer, device="cpu", out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    if not BPE.exists():
        if not create_tokenizer:
            raise SystemExit(f"Copy the NS/PE teammate's tokenizer.json to {BPE} first.")
        prepare_bpe()
    validation_hash = None
    tokenizer_hash = hashlib.sha256(BPE.read_bytes()).hexdigest()
    for seed in (24, 25, 26):
        train_hash = None
        for name, options in conditions.items():
            directory = out / f"{name}_seed{seed}"
            result = directory / "results.json"
            if not result.exists():
                train([*options, "--seed", str(seed), "--width", "256", "--heads", "8",
                       "--layers", "6", "--context", "32", "--batch", "8", "--steps", "500",
                       "--device", device, "--bpe", str(BPE), "--spectra", "--revision", REVISION,
                       "--pe-compile", "eager", "--out", str(directory)])
            run = json.loads(result.read_text())
            if hashlib.sha256(run["data"]["bpe_json"].encode()).hexdigest() != tokenizer_hash:
                raise ValueError(f"Tokenizer mismatch in {directory}")
            train_hash = train_hash or run["train_sha256"]
            validation_hash = validation_hash or run["validation_sha256"]
            if (train_hash, validation_hash) != (run["train_sha256"], run["validation_sha256"]):
                raise ValueError(f"Batch mismatch in {directory}")
            if not (directory / "transfer.png").exists():
                plot_transfer(directory)


if __name__ == "__main__":
    run({**NS_PE, **TIKH}, create_tokenizer=True)
