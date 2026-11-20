"""Capture SingleDeviceMuon's pre-orthogonalization input matrices"""

from contextlib import contextmanager

@contextmanager
def capture_inputs(named_parameters, step, snapshots, optimizer):
    """Capture pre-orthogonalization inputs for one optimizer step."""
    names = iter(name for name, _ in named_parameters)
    original = optimizer.orthogonalizer

    def record(matrix, steps):
        snapshots.append(dict(step=step, layer=next(names),
                              input=matrix.detach().cpu().clone()))
        return original(matrix, steps)

    optimizer.orthogonalizer = record
    try:
        yield
    finally:
        optimizer.orthogonalizer = original
