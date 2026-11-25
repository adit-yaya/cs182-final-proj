# Spectral filtering in Muon

We measure how Muon, Polar Express, and regularized polar maps transform momentum singular values during language-model training.

## Environment

Use Python 3.11 on macOS or Linux. The published runs used CPU PyTorch on an Apple M5 MacBook Pro.

```sh
git clone --depth 1 --branch reviewer-reproduction https://github.com/adit-yaya/cs182-final-proj.git
cd cs182-final-proj
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```sh
python scripts/reproduce.py demo       # train, validate, reload checkpoint, evaluate held-out text
python scripts/reproduce.py figures    # rebuild the 20 plots and two CSV tables
python -u scripts/reproduce.py train   # rerun the full dense experiment and rebuild the analysis
```

The first two commands run offline after installation. The demo uses the three bundled text splits. Figures use the per-seed measurements, loss histories, and histogram counts in `data/dense_summary.npz`. Outputs go to `results/reproduced/`. Compare them with the published figures and tables in [results/w256_bpe_dense](results/w256_bpe_dense).

Full training streams [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) at a pinned revision using the committed tokenizer. Budget about four hours on the reference laptop and at least 5 GB for outputs plus dataset cache space. It runs eight methods on seeds 24–26 for 500 updates each. Completed runs are reused on restart. To start fresh, pass a new directory with `--out PATH`. Small numerical differences are possible across platforms.
