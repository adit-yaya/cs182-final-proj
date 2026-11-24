"""Capture SingleDeviceMuon's pre-orthogonalization input matrices"""

from contextlib import contextmanager

import torch

@contextmanager
def capture_inputs(named_parameters, step, snapshots, optimizer, spectra=False):
    """Capture pre-orthogonalization inputs for one optimizer step."""
    names = iter(name for name, _ in named_parameters)
    original = optimizer.orthogonalizer

    def record(matrix, steps):
        snapshots.append(dict(step=step, layer=next(names),
                              input=matrix.detach().cpu().clone()))
        output = original(matrix, steps)
        if spectra:
            x = snapshots[-1]["input"].double()
            u, singular, vh = torch.linalg.svd(x, full_matrices=False)
            filtered = output.detach().cpu().double()
            snapshots[-1].update(input_sv=singular, normalized_sv=singular / (x.norm() + 1e-7),
                                 projection=(u.T @ filtered @ vh.T).diagonal().clone(),
                                 output_sv=torch.linalg.svdvals(filtered))
        return output

    optimizer.orthogonalizer = record
    try:
        yield
    finally:
        optimizer.orthogonalizer = original
