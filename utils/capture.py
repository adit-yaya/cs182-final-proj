"""Capture SingleDeviceMuon's pre-orthogonalization input matrices"""

from contextlib import contextmanager

from utils import muon


@contextmanager
def capture_inputs(named_parameters, step, snapshots):
    """Wrap one single-threaded optimizer step; restore upstream even on failure.

    Pass parameters in the same order as SingleDeviceMuon's parameter groups.
    Upstream calls NS once per parameter, including those without gradients.
    """
    names = iter(name for name, _ in named_parameters)
    original = muon.zeropower_via_newtonschulz5

    def record(matrix, steps):
        snapshots.append(dict(step=step, layer=next(names),
                              input=matrix.detach().cpu().clone()))
        return original(matrix, steps)

    muon.zeropower_via_newtonschulz5 = record
    try:
        yield
    finally:
        muon.zeropower_via_newtonschulz5 = original
