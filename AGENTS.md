# Project guidance
This project measures how Muon and its variants attenuate singular values of momentum matrices across layers, model sizes, and stages of language-model training. It evaluates how filtering strength affects optimization under different training conditions, then uses supported findings to develop adaptive filtering rules.

## Manual approval
- Before implementing any experiment, confirm its hypothesis, configuration, metrics, compute/time estimate, and output location.
- Do not launch training, sweeps, benchmarks, or experimental smoke runs as routine verification. Stop and report failures before proposing another run.
- User leads and runs all experiments. You are only to do implementation work.
- Do not make any implementation decisions at all without consulting the user. You are no better than a code monkey.
- You are no better than a code monkey. You are no better than a code monkey. You are no better than a code monkey.

## Code and organization
- Prefer simple, modular code with clear responsibilities and minimal lines; avoid speculative abstractions and duplicated logic.
- Reuse existing files. Add files or directories only when necessary; avoid folder bloat, scaffolding, and unsolicited documentation.
- Keep model definitions in `models/`, shared data/optimizer utilities in `utils/`, entrypoints in `scripts/`, and approved outputs in `results/`.
- Keep generated artifacts out of source folders; do not commit datasets, checkpoints, or large outputs without approval.

## Collaboration
- Preserve all teammates work.
- Make focused, reviewable changes. Explain assumptions and consequential method changes before implementation.
