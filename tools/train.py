#!/usr/bin/env python3
"""Resolve data/model paths and train one stage of WiD on DeCo."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "statistics/pca_uncentered.pt"
STARTS = {"pretrain_256": 0, "pretrain_512": 200000, "sft_512": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STARTS)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pretrained-root", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=PRIOR, help="PCA statistics; defaults to the original bundled prior")
    p.add_argument("--sft-root", type=Path, help="Override the extracted SFT pair directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="200K full state for 512-pre; 280K weights for fresh SFT")
    p.add_argument("--resume", type=Path, help="Full-state SAME-stage restart; data cursor restarts as historically")
    p.add_argument("--steps", type=int, help="Override the number of updates from the stage start")
    p.add_argument("--write-config-only", action="store_true")
    args = p.parse_args()
    if args.checkpoint and args.resume:
        p.error("--checkpoint and --resume are mutually exclusive")
    if args.steps is not None and (args.steps <= 0 or args.resume):
        p.error("--steps must be positive and used at a stage start, not with --resume")
    if args.stage != "pretrain_256" and not (args.checkpoint or args.resume):
        p.error("This stage requires --checkpoint (stage transition) or --resume")
    if args.stage == "pretrain_256" and args.checkpoint:
        p.error("Fresh 256 starts from PCA initialization; use --resume for same-stage restart")
    args.prior = args.prior.resolve()
    if not args.prior.is_file():
        raise FileNotFoundError(args.prior)
    args.output = args.output.resolve()
    cfg = yaml.safe_load((ROOT / f"configs_t2i/wid/{args.stage}.yaml").read_text())
    cfg["trainer"]["default_root_dir"] = str(args.output / "runs")
    cfg["trainer"]["logger"]["init_args"]["save_dir"] = str(args.output / "metrics")
    model = cfg["model"]
    model["denoiser"]["init_args"]["prior_path"] = str(args.prior)
    pretrained = args.pretrained_root.resolve()
    model["conditioner"]["init_args"]["weight_path"] = str(pretrained / "Qwen/Qwen3-1.7B")
    encoder = model["diffusion_trainer"]["init_args"]["encoder"]["init_args"]
    encoder["weight_path"] = str(pretrained / "dinov2/dinov2_vitb14_pretrain.pth")
    encoder["local_repo_path"] = str(pretrained / "dinov2/facebookresearch_dinov2_main")
    data = cfg["data"]["train_dataset"]["init_args"]
    if args.stage == "sft_512":
        data["root"] = str(args.sft_root.resolve() if args.sft_root else
                           args.data_root.resolve() / "BLIP3o-60k-extracted")
        if args.checkpoint:
            cfg["init_ckpt_path"] = str(args.checkpoint.resolve())
    else:
        data["urls"] = [str(args.data_root.resolve() / Path(url).name) for url in data["urls"]]
        if args.checkpoint:
            cfg["ckpt_path"] = str(args.checkpoint.resolve())
    if args.resume:
        cfg["ckpt_path"] = str(args.resume.resolve())
        cfg["model"]["resume_rng_policy"] = "strict"
    if args.steps is not None:
        cfg["trainer"]["max_steps"] = STARTS[args.stage] + args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "resolved.yaml"
    if path.exists() and not args.resume:
        raise FileExistsError(f"Use a new output directory: {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(path)
    if args.write_config_only:
        return
    env = os.environ.copy()
    env.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8")
    subprocess.run([sys.executable, "-u", "main.py", "fit", "--config", str(path)],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
