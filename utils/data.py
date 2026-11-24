"""Stream FineWeb, tokenize text, and evaluate loss."""

import hashlib
import math

from datasets import load_dataset
import tiktoken
import torch


DATASET = "HuggingFaceFW/fineweb"
SPLIT_POLICY = "sha256-text-mod100:0=test,1=validation,2..99=train:v1"


def document_split(text):
    # Hash text rather than row ID so identical documents cannot cross splits.
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big") % 100
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def documents(data, split):
    stream = load_dataset(data["dataset"], name=data["subset"], revision=data["revision"],
                          split="train", streaming=True).select_columns(["text"])
    return stream.filter(lambda row: document_split(row["text"]) == split)


def batches(data, split, batch_size, context, seed=17, shuffle_buffer=1000):
    """Pack documents with EOS separators into fixed-length next-token batches."""
    if data["split_policy"] != SPLIT_POLICY:
        raise ValueError("Checkpoint uses a different split policy.")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    if "bpe_json" in data:
        from tokenizers import Tokenizer
        encoder = Tokenizer.from_str(data["bpe_json"])
        encode = lambda text: encoder.encode(text).ids
        eos = encoder.token_to_id("<|endoftext|>")
    else:
        encoder = tiktoken.get_encoding(data["tokenizer"])
        encode, eos = encoder.encode_ordinary, encoder.eot_token
    stream = documents(data, split)
    if split == "train":
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    buffer = []
    size = batch_size * context
    for row in stream:
        buffer.extend(encode(row["text"]))
        buffer.append(eos)
        offset = 0
        while len(buffer) - offset >= size + 1:
            tokens = torch.tensor(buffer[offset:offset + size + 1], dtype=torch.long)
            yield tokens[:-1].view(batch_size, context), tokens[1:].view(batch_size, context)
            offset += size
        buffer = buffer[offset:]


@torch.no_grad()
def evaluate(model, source, count, device):
    """Token-weighted cross-entropy (nats/token) and perplexity on count batches."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    source = iter(source)
    try:
        for _ in range(count):
            try:
                x, y = next(source)
            except StopIteration as error:
                raise ValueError("Evaluation stream ended before the requested batch count.") from error
            _, loss = model(x.to(device), y.to(device))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Evaluation requires at least one batch.")
    loss = total_loss / total_tokens
    return dict(loss=loss, perplexity=math.exp(loss) if loss < 700 else float("inf"),
                tokens=total_tokens, batches=count)
