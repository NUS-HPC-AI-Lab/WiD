"""Strict precision guards for analytic spectral operators."""

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def spectral_fp32_context(ref=None, enabled=True):
    """Use FP32 values and full FP32 matmul internals, then restore the caller."""

    if not enabled:
        yield
        return
    previous_precision = torch.get_float32_matmul_precision()
    autocast_context = (
        torch.autocast(device_type="cuda", enabled=False)
        if ref is not None and ref.is_cuda
        else nullcontext()
    )
    torch.set_float32_matmul_precision("highest")
    try:
        with autocast_context:
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def orthogonality_max_abs_fp64(basis):
    basis64 = basis.detach().to(device="cpu", dtype=torch.float64)
    gram = basis64.transpose(0, 1) @ basis64
    identity = torch.eye(basis64.shape[1], dtype=torch.float64)
    return float((gram - identity).abs().max().item())
