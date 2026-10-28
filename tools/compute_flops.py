#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pixdit_core.pixeldit_c2i import PixDiT
from pixdit_core.modules import RotaryAttention
import pixdit_core.modules as _pixdit_modules


def _install_sdpa_stub():
    stub = lambda q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False: v
    torch.nn.functional.scaled_dot_product_attention = stub
    _pixdit_modules.scaled_dot_product_attention = stub


class FlopsCounter:
    def __init__(self, model):
        self.total = 0
        self._handles = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, RotaryAttention):
                self._handles.append(m.register_forward_pre_hook(self._rotary_hook))

    def _linear_hook(self, mod, inp, out):
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() == 0:
            return
        n = x.numel() // x.shape[-1]
        self.total += 2 * n * mod.in_features * mod.out_features

    def _rotary_hook(self, mod, inp):
        if not inp:
            return
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            return
        B, N, C = x.shape
        self.total += 4 * B * N * N * C

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_model(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})

    if "denoiser" in model_cfg:
        init_args = model_cfg["denoiser"].get("init_args", {})
        model = PixDiT(**init_args)
        return model, "c2i", cfg

    raise ValueError(f"Expected a C2I config with model.denoiser: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute GFLOPs for upstream PixelDiT C2I models.")
    parser.add_argument("--config", required=True, help="Path to an upstream C2I YAML config.")
    parser.add_argument("--height", type=int, default=256, help="Input image height (default: 256).")
    parser.add_argument("--width", type=int, default=256, help="Input image width (default: 256).")
    args = parser.parse_args()

    _install_sdpa_stub()

    model, mode, cfg = build_model(args.config)
    H, W = args.height, args.width
    params = sum(p.numel() for p in model.parameters())

    model.eval().to("cpu")
    x = torch.randn(1, 3, H, W)
    t = torch.tensor([0.5])

    y = torch.zeros(1, dtype=torch.long)

    with FlopsCounter(model) as fc, torch.no_grad():
        model(x, t, y)
        gflops = fc.total / 1e9

    print(f"Model:      PixelDiT ({mode})")
    print(f"Config:     {os.path.basename(args.config)}")
    print(f"Parameters: {params / 1e6:.1f}M")
    print(f"Resolution: {H} x {W}")
    print(f"GFLOPs:     {gflops:.2f}")


if __name__ == "__main__":
    main()
