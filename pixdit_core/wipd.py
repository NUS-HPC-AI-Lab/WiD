"""WiD NoGate: the full-rank, balanced input/output route used in the XL runs.

Historical state-dict names are retained so existing experiment checkpoints
load without key conversion. No gates, alternative decoders or ablation paths.
Patch vectors use (channel, row, column), i.e. F.unfold order.
"""

import hashlib
import math
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from pixdit_core.modules import RMSNorm


@contextmanager
def spectral_fp32_context(ref):
    previous = torch.get_float32_matmul_precision()
    context = torch.autocast("cuda", enabled=False) if ref.is_cuda else nullcontext()
    torch.set_float32_matmul_precision("highest")
    try:
        with context:
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def load_patch_pca(path, patch_dim):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format_version") != "shared_patch_pca_uncentered_second_moment_v1"
            or payload.get("flatten_order") != "chw_unfold_v1"
            or payload.get("covariance_kind") != "uncentered_second_moment"
            or payload.get("patch_dim") != patch_dim):
        raise ValueError("WiD requires the uncentered CHW shared-patch PCA prior")
    u = payload["eigenvectors_fp32_desc"].float()
    lam = payload["eigenvalues_fp32_desc"].float()
    if u.shape != (patch_dim, patch_dim) or lam.shape != (patch_dim,):
        raise ValueError("Invalid PCA shape")
    if not torch.isfinite(u).all() or not torch.isfinite(lam).all() or (lam < 0).any():
        raise ValueError("Invalid PCA values")
    if (u.double().T @ u.double() - torch.eye(patch_dim)).abs().max() > 2e-4:
        raise ValueError("PCA columns are not orthonormal")
    return u, lam, payload


def dct2d_channel_basis(patch_size, channels):
    """Data-free DCT-II columns, ordered by (kh²+kw², kh, kw, channel)."""
    n = patch_size
    k = torch.arange(n, dtype=torch.float64).view(-1, 1)
    m = torch.arange(n, dtype=torch.float64).view(1, -1)
    dct = torch.cos(math.pi * (m + .5) * k / n) * math.sqrt(2 / n)
    dct[0] /= math.sqrt(2)
    columns, order = [], []
    for kh in range(n):
        for kw in range(n):
            spatial = torch.einsum("h,w->hw", dct[kh], dct[kw]).reshape(-1)
            for c in range(channels):
                column = torch.zeros(channels * n * n, dtype=torch.float64)
                column[c*n*n:(c+1)*n*n] = spatial
                columns.append(column)
                order.append((kh*kh + kw*kw, kh, kw, c))
    return torch.stack(columns, 1)[:, sorted(range(len(order)), key=order.__getitem__)].float()


def build_basis(kind, prior_path, patch_dim, patch_size, channels):
    if kind == "dct":
        payload = torch.load(prior_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") == "shared_patch_dct_uncentered_second_moment_v1":
            if (payload.get("covariance_kind") != "uncentered_second_moment"
                    or payload.get("flatten_order") != "chw_unfold_v1"
                    or payload.get("patch_size") != patch_size
                    or payload.get("channels") != channels
                    or payload.get("patch_dim") != patch_dim
                    or payload.get("ordering") != "kh^2+kw^2,kh,kw,channel"):
                raise ValueError("WiD requires matching uncentered CHW patch-DCT statistics")
            basis, spectrum = payload["basis"].float(), payload["spectrum"].float()
            expected = dct2d_channel_basis(patch_size, channels)
            if (basis.shape != (patch_dim, patch_dim) or spectrum.shape != (patch_dim,)
                    or not torch.isfinite(basis).all() or not torch.isfinite(spectrum).all()
                    or (spectrum < 0).any()
                    or not torch.allclose(basis, expected, rtol=0, atol=1e-6)):
                raise ValueError("Invalid DCT basis or second-moment spectrum")
            return basis, spectrum
    u, lam, _ = load_patch_pca(prior_path, patch_dim)
    if kind == "dct":
        # M = U diag(lam) U^T, not the centered covariance C.
        # DCT uses diag(D^T M D); it does not diagonalize the full M.
        basis = dct2d_channel_basis(patch_size, channels)
        with spectral_fp32_context(u):
            lam = ((u.T @ basis).square() * lam[:, None]).sum(0)
        u = basis
    elif kind != "pca":
        raise ValueError("basis must be pca or dct")
    return u, lam


def analytic_velocity_gain(lam, t, denominator_floor=1e-12):
    """E[v|z] gain for z=t*x+(1-t)*noise and v=x-noise."""
    t = t.view(-1, 1, 1).float()
    lam = lam.view(1, 1, -1).float()
    return (t * lam - (1-t)) / ((t*t)*lam + (1-t)*(1-t)).clamp_min(denominator_floor)


class AnchoredSpectralHead(nn.Module):
    def __init__(self, prior_path: str, patch_size: int = 16, channels: int = 3,
                 pixel_hidden_size: int = 16, basis: str = "pca",
                 eigenvalue_floor: float = 1e-6, denominator_floor: float = 1e-12):
        super().__init__()
        self.patch_size, self.channels = patch_size, channels
        self.pixel_hidden_size = pixel_hidden_size
        self.patch_dim = self.rank = channels * patch_size**2
        self.eigenvalue_floor, self.denominator_floor = eigenvalue_floor, denominator_floor
        u, lam = build_basis(basis, prior_path, self.patch_dim, patch_size, channels)
        self.register_buffer("eigenvalue_permutation", torch.arange(self.rank))
        self.basis = nn.Parameter(u.clone())
        self.decoder_basis = nn.Parameter(u.clone())
        log_lam = lam.clamp_min(eigenvalue_floor).log()
        self.log_lambda = nn.Parameter(log_lam.clone())
        self.register_buffer("log_lambda_init", log_lam.clone())
        self.delta_norm = RMSNorm(pixel_hidden_size, eps=1e-6)
        self.delta = nn.Linear(patch_size**2 * pixel_hidden_size, self.rank, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, pixel_features, x_patches, t):
        b, length, _ = x_patches.shape
        with spectral_fp32_context(x_patches):
            c = torch.einsum("bld,dr->blr", x_patches.float(), self.basis.float())
            lam = self.log_lambda.float().exp().clamp_min(self.eigenvalue_floor)
            coeff = analytic_velocity_gain(lam, t, self.denominator_floor) * c
        delta = self.delta(self.delta_norm(pixel_features).reshape(b * length, -1))
        with spectral_fp32_context(x_patches):
            coeff = coeff + delta.view(b, length, self.rank).float()
            velocity = torch.einsum("blr,dr->bld", coeff, self.decoder_basis.float())
        return velocity.to(x_patches.dtype)


class PiTSpectralAnchor(nn.Module):
    """Ungated raw PCA coefficients projected into the native pixel stream."""
    def __init__(self, prior_path: str, patch_size: int = 16, channels: int = 3,
                 pixel_hidden_size: int = 16, basis: str = "pca",
                 eigenvalue_floor: float = 1e-6):
        super().__init__()
        self.patch_size, self.channels = patch_size, channels
        self.pixel_hidden_size = pixel_hidden_size
        self.patch_dim = self.rank = channels * patch_size**2
        u, lam = build_basis(basis, prior_path, self.patch_dim, patch_size, channels)
        self.register_buffer("eigenvalue_permutation", torch.arange(self.rank))
        self.basis = nn.Parameter(u.clone())
        log_lam = lam.clamp_min(eigenvalue_floor).log()
        self.log_lambda = nn.Parameter(log_lam.clone(), requires_grad=False)
        self.register_buffer("log_lambda_init", log_lam.clone())
        self.expand_anchor = nn.Linear(self.rank, patch_size**2 * pixel_hidden_size, bias=False)

    @torch.no_grad()
    def initialize_expand_from_pixel_embedder(self, pixel_embedder, scale=.5):
        weight = pixel_embedder.proj.weight.detach().float()
        p2 = self.patch_size**2
        lift = torch.zeros(p2 * self.pixel_hidden_size, self.patch_dim,
                           device=weight.device, dtype=torch.float32)
        for p in range(p2):
            for c in range(self.channels):
                lift[p*self.pixel_hidden_size:(p+1)*self.pixel_hidden_size, c*p2+p] = weight[:, c]
        with spectral_fp32_context(lift):
            self.expand_anchor.weight.copy_(scale * (lift @ self.basis.detach().float()))

    def forward(self, x_patches, s_cond, t):
        b, length, _ = x_patches.shape
        with spectral_fp32_context(x_patches):
            coeff = torch.einsum("bld,dr->blr", x_patches.float(), self.basis.float())
            out = nn.functional.linear(coeff.reshape(b*length, self.rank), self.expand_anchor.weight.float())
        return out.view(b*length, self.patch_size**2, self.pixel_hidden_size).to(s_cond.dtype)
