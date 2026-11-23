"""Evaluate the nine final weak-regularization checkpoints on one validation stream."""
import csv
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.model import GPT, GPTConfig
from utils.data import batches, SPLIT_POLICY

PILOT = ROOT / 'results/optimization-experiments/dynamic_tikh_weak_pilot'
OUT = PILOT / 'heldout_validation'
SETTINGS = ('fixed_zero', 'fixed_weak', 'weak_to_zero')
CONFIG = dict(vocab_size=50257, block_size=32, n_layer=6, n_head=8, n_embd=88)
PREFIX_BATCHES, NEW_BATCHES = 10, 512


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows):
    with (OUT / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    torch.manual_seed(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if OUT.exists():
        raise FileExistsError(OUT)
    runs = json.loads((PILOT / 'replication_analysis/sources.json').read_text())['runs']
    data = runs['17/fixed_zero']['metadata']['data']
    assert data['split_policy'] == SPLIT_POLICY and data['tokenizer'] == 'gpt2'
    models, original, hashes = {}, {}, {}
    for seed in (17, 18, 19):
        for setting in SETTINGS:
            key = f'{seed}/{setting}'
            folder = Path(runs[key]['path'])
            meta = json.loads((folder / 'results.json').read_text())
            path = folder / 'checkpoint.pt'
            hashes[key] = sha(path)
            ck = torch.load(path, map_location='cpu', weights_only=True)
            assert ck['config'] == meta['config'] == CONFIG
            assert ck['data'] == meta['data'] == data and ck['step'] == meta['steps'] == 500
            assert meta['source_sha256']['utils/data.py'] == sha(ROOT / 'utils/data.py')
            assert meta['source_sha256']['models/model.py'] == sha(ROOT / 'models/model.py')
            assert ck['seed'] == meta['seed'] == seed and meta['batch'] == 8
            assert meta['method'] == 'tikh' and meta['filter_steps'] == 5
            assert meta['tikh_lambda'] == (0 if setting == 'fixed_zero' else 6.8851809e-6)
            assert meta.get('tikh_switch_step') == (100 if setting == 'weak_to_zero' else None)
            assert meta.get('tikh_lambda_after') == (0 if setting == 'weak_to_zero' else None)
            val = meta['validation_losses'][-1]
            assert val['step'] == 500 and val['tokens'] == 2560
            original[key] = val['loss']
            model = GPT(GPTConfig(**CONFIG))
            model.load_state_dict(ck['model'], strict=True)
            models[key] = model.eval()
    print('Verified all nine checkpoints. Reading pinned validation stream.', flush=True)
    total = PREFIX_BATCHES + NEW_BATCHES
    packed = list(islice(batches(data, 'validation', 8, 32), total))
    assert len(packed) == total
    stream = torch.cat([x.flatten() for x,y in packed] + [packed[-1][1].flatten()[-1:]])
    for i, (x,y) in enumerate(packed):
        assert torch.equal(y.flatten(), stream[256*i+1:256*(i+1)+1])
    stream_hash = hashlib.sha256(stream.numpy().astype('<i8').tobytes()).hexdigest()
    print(f'Materialized {total} batches. Token SHA-256 {stream_hash}', flush=True)
    scores, replay = {}, {}
    for key, model in models.items():
        losses = []
        for x,y in packed:
            logits,_ = model(x)
            token_loss = F.cross_entropy(logits.flatten(0,1), y.flatten(), reduction='none')
            losses.extend(token_loss.view(8,32).double().mean(1).tolist())
        arr = np.array(losses)
        assert np.isfinite(arr).all()
        replay[key] = arr[:80].mean()
        if abs(replay[key]-original[key]) > 2e-6:
            raise ValueError(f'Original validation mismatch for {key}: {replay[key]} vs {original[key]}')
        scores[key] = arr[80:]
        print(key, 'replayed', replay[key], 'larger', scores[key].mean(), flush=True)
    rows, comparisons, sequences = [], [], []
    for seed in (17,18,19):
        values = {s:scores[f'{seed}/{s}'] for s in SETTINGS}
        for setting in SETTINGS:
            key = f'{seed}/{setting}'
            rows.append(dict(seed=seed, setting=setting, original=original[key], replay=replay[key],
                             expanded=values[setting].mean()))
        pairs = [(a,b) for a,b in [('fixed_weak','fixed_zero'),('weak_to_zero','fixed_zero'),('weak_to_zero','fixed_weak')]]
        diff = {f'{a}_minus_{b}':values[a]-values[b] for a,b in pairs}
        for a,b in pairs:
            d = diff[f'{a}_minus_{b}']
            blocks = d.reshape(-1,64).mean(1)
            comparisons.append(dict(seed=seed, configuration=a, reference=b,
                original_delta=original[f'{seed}/{a}']-original[f'{seed}/{b}'], expanded_delta=d.mean(),
                sequence_sd=d.std(ddof=1), fraction_sequences_lower=(d<0).mean(),
                block_sd=blocks.std(ddof=1), blocks_lower=int((blocks<0).sum()), block_count=len(blocks)))
        for i in range(len(values['fixed_zero'])):
            sequences.append(dict(seed=seed,sequence=i,first_target_offset=2561+32*i,block=i//64,
                                  **{s:v[i] for s,v in values.items()}, **{s:v[i] for s,v in diff.items()}))
    OUT.mkdir()
    save_csv('losses.csv', rows)
    save_csv('paired-differences.csv', comparisons)
    save_csv('sequence-losses.csv', sequences)
    (OUT/'manifest.json').write_text(json.dumps(dict(data=data, split='validation', config=CONFIG,
        prefix_targets=2560, expanded_targets=131072, block_sequences=64, sequence_tokens=32,
        token_sha256_le_i64_including_prefix=stream_hash, checkpoint_sha256=hashes,
        torch=str(torch.__version__), device='cpu', threads=4, deterministic=True,
        source_sha256=sha(Path(__file__))),indent=2))
    print('Saved',OUT,flush=True)


if __name__ == '__main__':
    main()
