"""Shared-patch PCA and Wiener operators for Wiener-informed Decoding (WiD).

The balanced input route preserves DeCo's initial dense query. The learned
output heads start at zero, leaving the analytic Wiener velocity at initialization.
Parameter names are retained for compatibility with the trained checkpoints.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.spectral_precision import (
    orthogonality_max_abs_fp64,
    spectral_fp32_context,
)


SUPPORTED_PRIOR_FORMATS = {
    "shared_patch_pca_uncentered_second_moment_v1",
    "deco_t2i_shared_patch_pca_pooled_uncentered_direct_v1",
}
SUPPORTED_FLATTEN_ORDER = "chw_unfold_v1"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prior(path: str, patch_size: int, channels: int):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    patch_dim = channels * patch_size * patch_size
    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_PRIOR_FORMATS:
        raise ValueError(
            f"PCA prior format_version={format_version!r}, expected one of "
            f"{sorted(SUPPORTED_PRIOR_FORMATS)!r}: {path}"
        )
    checks = {
        "patch_size": patch_size,
        "channels": channels,
        "patch_dim": patch_dim,
        "flatten_order": SUPPORTED_FLATTEN_ORDER,
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"PCA prior {key}={actual!r}, expected {expected!r}: {path}"
            )
    if payload.get("covariance_kind", "uncentered_second_moment_direct") != \
            "uncentered_second_moment_direct":
        raise ValueError("WiD requires a direct uncentered second moment prior")
    preprocessing = str(payload.get("preprocessing", ""))
    if format_version == "deco_t2i_shared_patch_pca_pooled_uncentered_direct_v1":
        required_preprocessing = (
            "active_DeCo_WebDataset",
            "center_crop",
            "Normalize_minus1_plus1",
        )
    else:
        required_preprocessing = ("center_crop_256", "x=2*img-1")
    if not all(token in preprocessing for token in required_preprocessing):
        raise ValueError(
            "PCA prior preprocessing is not compatible with DeCo: "
            f"{preprocessing!r}"
        )

    basis = payload["eigenvectors_fp32_desc"].to(torch.float32).contiguous()
    eigenvalues = payload["eigenvalues_fp32_desc"].to(torch.float32).contiguous()
    if basis.shape != (patch_dim, patch_dim):
        raise ValueError(f"basis shape {tuple(basis.shape)} != {(patch_dim, patch_dim)}")
    if eigenvalues.shape != (patch_dim,):
        raise ValueError(
            f"eigenvalue shape {tuple(eigenvalues.shape)} != {(patch_dim,)}"
        )
    return basis, eigenvalues, payload


def analytic_velocity_gain(
    eigenvalues: torch.Tensor,
    t: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    """Official-code orientation: z_t=t*x+(1-t)*eps and v=x-eps."""

    lam = eigenvalues.view(1, 1, -1).to(torch.float32)
    t = t.view(-1, 1, 1).to(torch.float32)
    one_minus_t = 1.0 - t
    numerator = t * lam - one_minus_t
    denominator = t.square() * lam + one_minus_t.square()
    return numerator / denominator.clamp_min(float(denominator_floor))


class SharedPatchWienerState(nn.Module):
    """Full-rank shared-patch PCA analysis and Wiener synthesis state."""

    def __init__(
        self,
        prior_path: str,
        patch_size: int = 16,
        channels: int = 3,
        rank: int = 768,
        train_encoder_basis: bool = True,
        train_decoder_basis: bool = True,
        train_gains: bool = True,
        eigenvalue_floor: float = 1e-6,
        denominator_floor: float = 1e-12,
    ):
        super().__init__()
        self.prior_path = str(prior_path)
        self.prior_sha256 = _sha256_file(self.prior_path)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.patch_dim = self.channels * self.patch_size * self.patch_size
        self.rank = int(rank)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.denominator_floor = float(denominator_floor)
        if self.rank != self.patch_dim:
            raise ValueError(
                "WiD for DeCo uses the full patch basis; "
                f"rank={self.rank}, patch_dim={self.patch_dim}."
            )

        basis, eigenvalues, payload = _load_prior(
            self.prior_path, self.patch_size, self.channels
        )
        basis = basis[:, : self.rank]
        eigenvalues = eigenvalues[: self.rank].clamp_min(self.eigenvalue_floor)
        orthogonality_error = orthogonality_max_abs_fp64(basis)
        if orthogonality_error > 1e-5:
            raise ValueError(
                f"PCA basis is not orthonormal in FP64: max_abs={orthogonality_error:.3e}"
            )

        self.encoder_basis = nn.Parameter(
            basis.clone(), requires_grad=bool(train_encoder_basis)
        )
        self.decoder_basis = nn.Parameter(
            basis.clone(), requires_grad=bool(train_decoder_basis)
        )
        self.log_lambda = nn.Parameter(
            eigenvalues.log().clone(), requires_grad=bool(train_gains)
        )
        self.register_buffer("encoder_basis_init", basis.clone(), persistent=False)
        self.register_buffer("decoder_basis_init", basis.clone(), persistent=False)
        self.register_buffer(
            "log_lambda_init", eigenvalues.log().clone(), persistent=True
        )
        self.prior_num_images = int(payload.get("num_images", -1))
        self.prior_preprocessing = str(payload.get("preprocessing", ""))

    def analyze(self, x_patches: torch.Tensor) -> torch.Tensor:
        with spectral_fp32_context(x_patches):
            return torch.einsum(
                "bld,dr->blr",
                x_patches.to(torch.float32),
                self.encoder_basis.to(torch.float32),
            )

    def analytic_coefficients(
        self, coefficients: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        with spectral_fp32_context(coefficients):
            eigenvalues = self.effective_eigenvalues()
            gain = analytic_velocity_gain(
                eigenvalues, t, self.denominator_floor
            )
            return gain * coefficients.to(torch.float32)

    def effective_eigenvalues(self) -> torch.Tensor:
        log_lambda = self.log_lambda.to(torch.float32)
        return log_lambda.exp().clamp_min(self.eigenvalue_floor)

    def decode(
        self, coefficients: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        with spectral_fp32_context(coefficients):
            patches = torch.einsum(
                "blr,dr->bld",
                coefficients.to(torch.float32),
                self.decoder_basis.to(torch.float32),
            )
        return patches.to(output_dtype)

class BalancedPCARoute(nn.Module):
    """Full-rank PCA-to-pixel input route, without semantic gating."""

    def __init__(
        self,
        context_dim: int,
        rank: int,
        patch_size: int,
        pixel_hidden_size: int,
        balanced_fraction: float = 0.5,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.rank = int(rank)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.balanced_fraction = float(balanced_fraction)
        if not 0.0 <= self.balanced_fraction <= 1.0:
            raise ValueError("balanced_fraction must be in [0,1]")
        self.expander = nn.Linear(
            self.rank,
            self.patch_size * self.patch_size * self.pixel_hidden_size,
            bias=False,
        )
        self.register_buffer(
            "native_rgb_weight_before_balance",
            torch.empty(0),
            persistent=False,
        )

    @torch.no_grad()
    def initialize_from_deco_pixel_embedder(
        self,
        nerf_embedder: nn.Module,
        encoder_basis: torch.Tensor,
        channels: int,
    ):
        linear = nerf_embedder.embedder[0]
        if not isinstance(linear, nn.Linear):
            raise TypeError("Expected NerfEmbedder.embedder[0] to be nn.Linear")
        if linear.in_features < int(channels):
            raise ValueError("NerfEmbedder has fewer input features than RGB channels")

        rgb_weight = linear.weight[:, :channels].detach().to(torch.float32).clone()
        self.native_rgb_weight_before_balance = rgb_weight.clone()
        p2 = self.patch_size * self.patch_size
        basis = encoder_basis.detach().to(torch.float32)
        # U has CHW-unfold rows.  For pixel q and hidden channel h:
        # W_A[q,h,j] = sum_c E_rgb[h,c] U[c,q,j].
        basis_cqr = basis.view(channels, p2, self.rank)
        target = torch.einsum("hc,cqr->qhr", rgb_weight, basis_cqr)
        target = target.reshape(p2 * self.pixel_hidden_size, self.rank)
        self.expander.weight.copy_(
            (self.balanced_fraction * target).to(
                device=self.expander.weight.device,
                dtype=self.expander.weight.dtype,
            )
        )
        # Only RGB columns are split.  DCT positional columns and bias remain
        # single-copy in the native DeCo embedder.
        linear.weight[:, :channels].mul_(1.0 - self.balanced_fraction)

    def forward(
        self,
        coefficients: torch.Tensor,
        semantic_context: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        bsz, length, rank = coefficients.shape
        if rank != self.rank:
            raise ValueError(f"coefficient rank {rank} != configured {self.rank}")
        # Retain the unit multiplication used during training.
        gate = coefficients.new_ones((bsz * length, self.rank))
        with spectral_fp32_context(coefficients):
            routed = gate.to(torch.float32) * coefficients.reshape(
                bsz * length, self.rank
            ).to(torch.float32)
            expanded = F.linear(
                routed, self.expander.weight.to(torch.float32), bias=None
            )
        p2 = self.patch_size * self.patch_size
        return expanded.view(
            bsz * length, p2, self.pixel_hidden_size
        ).to(output_dtype)
