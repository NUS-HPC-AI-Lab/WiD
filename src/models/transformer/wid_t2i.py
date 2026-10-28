"""Wiener-informed Decoding (WiD) for DeCo text-to-image generation.

The text refiner, semantic DiT and native pixel decoder are unchanged. WiD
adds a balanced PCA input route and an analytic-plus-learned coefficient
output route. Both learned output heads use the normalized decoder state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.dit_t2i_DeCo import PixNerDiT
from src.models.transformer.wid import (
    BalancedPCARoute,
    SharedPatchWienerState,
)


class DeCoWiD(PixNerDiT):
    """DeCo's native architecture with Wiener-informed Decoding."""

    def __init__(
        self,
        *args,
        prior_path: str,
        rank: int = 768,
        train_encoder_basis: bool = True,
        train_decoder_basis: bool = True,
        train_gains: bool = True,
        eigenvalue_floor: float = 1e-6,
        denominator_floor: float = 1e-12,
        balanced_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.patch_size != 16 or self.in_channels != 3:
            raise ValueError("WiD for DeCo requires RGB with native patch size 16")

        pixel_hidden_size = int(self.dec_net.model_channels)
        self.wiener_state = SharedPatchWienerState(
            prior_path=prior_path,
            patch_size=self.patch_size,
            channels=self.in_channels,
            rank=rank,
            train_encoder_basis=train_encoder_basis,
            train_decoder_basis=train_decoder_basis,
            train_gains=train_gains,
            eigenvalue_floor=eigenvalue_floor,
            denominator_floor=denominator_floor,
        )
        self.pca_route = BalancedPCARoute(
            context_dim=self.hidden_size,
            rank=rank,
            patch_size=self.patch_size,
            pixel_hidden_size=pixel_hidden_size,
            balanced_fraction=balanced_fraction,
        )
        self.coefficient_delta = nn.Linear(
            self.patch_size * self.patch_size * pixel_hidden_size,
            rank,
            bias=True,
        )
        nn.init.zeros_(self.coefficient_delta.weight)
        nn.init.zeros_(self.coefficient_delta.bias)
        self.pca_route.initialize_from_deco_pixel_embedder(
            self.x_embedder,
            self.wiener_state.encoder_basis,
            channels=self.in_channels,
        )

    def _patchify(self, image: torch.Tensor) -> torch.Tensor:
        return F.unfold(
            image, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)

    def _native_pixel_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = patches.shape
        p2 = self.patch_size * self.patch_size
        pixels = patches.reshape(
            bsz * length, self.in_channels, p2
        ).transpose(1, 2)
        return self.x_embedder(pixels)

    def _decoder_hidden(
        self, dense_query: torch.Tensor, semantic_context: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.dec_net.input_proj(dense_query)
        condition = self.dec_net.cond_embed(semantic_context).reshape(
            semantic_context.shape[0], self.patch_size * self.patch_size, -1
        )
        if self.dec_net.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.dec_net.res_blocks:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, condition, use_reentrant=False
                )
        else:
            for block in self.dec_net.res_blocks:
                hidden = block(hidden, condition)
        return hidden

    def forward(self, image: torch.Tensor, t: torch.Tensor, text: torch.Tensor):
        bsz, _, height, width = image.shape
        patches = self._patchify(image)
        position = self.fetch_pos(
            height // self.patch_size,
            width // self.patch_size,
            image.device,
        )

        t_emb = self.t_embedder(t.view(-1)).view(bsz, -1, self.hidden_size)
        text_tokens = self.y_embedder(text).view(bsz, -1, self.hidden_size)
        if text_tokens.shape[1] > self.y_pos_embedding.shape[1]:
            raise ValueError(
                f"text length {text_tokens.shape[1]} exceeds configured "
                f"maximum {self.y_pos_embedding.shape[1]}"
            )
        text_tokens = text_tokens + self.y_pos_embedding[
            :, : text_tokens.shape[1]
        ].to(text_tokens.dtype)

        condition = F.silu(t_emb)
        for block in self.text_refine_blocks:
            text_tokens = block(text_tokens, condition)

        semantic = self.s_embedder(patches)
        for block in self.blocks:
            semantic = block(semantic, text_tokens, condition, position)
        semantic = F.silu(t_emb + semantic)

        length = semantic.shape[1]
        semantic_context = semantic.reshape(bsz * length, self.hidden_size)
        coefficients = self.wiener_state.analyze(patches)

        native = self._native_pixel_tokens(patches)
        prior = self.pca_route(
            coefficients, semantic_context, output_dtype=native.dtype
        )
        hidden = self._decoder_hidden(native + prior, semantic_context)

        # Mirror the official FinalLayer: both learned heads consume the same
        # normalized per-pixel decoder state.  There is no norm-bypassing delta.
        normalized_hidden = self.dec_net.final_layer.norm_final(hidden)
        pixel_velocity = self.dec_net.final_layer.linear(normalized_hidden)
        coefficient_delta = self.coefficient_delta(
            normalized_hidden.reshape(bsz * length, -1)
        ).view(bsz, length, self.wiener_state.rank)

        analytic_coeff = self.wiener_state.analytic_coefficients(
            coefficients, t.view(-1)
        )
        coefficient_patches = self.wiener_state.decode(
            analytic_coeff + coefficient_delta.to(torch.float32),
            output_dtype=pixel_velocity.dtype,
        )

        p2 = self.patch_size * self.patch_size
        coefficient_pixels = coefficient_patches.view(
            bsz, length, self.in_channels, p2
        ).permute(0, 1, 3, 2)
        pixel_velocity = pixel_velocity.view(
            bsz, length, p2, self.in_channels
        )
        output_patches = (coefficient_pixels + pixel_velocity).permute(
            0, 1, 3, 2
        ).reshape(bsz, length, self.in_channels * p2)
        return F.fold(
            output_patches.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
