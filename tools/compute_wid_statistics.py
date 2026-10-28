#!/usr/bin/env python3
"""Compute the shared-patch PCA statistics used by WiD on DeCo.

This intentionally mirrors the ImageNet Shared-PCA protocol:
  * naturally sample training images;
  * apply the active DeCo preprocessing;
  * use every non-overlapping 16x16 RGB patch;
  * accumulate the direct uncentered second moment E[p p^T];
  * produce one pooled basis/spectrum, with resolution recorded only as audit
    metadata rather than as separate priors.

Run with torchrun on eight GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset.blip3o_dataset import WebDatasetPackedDataset


PATCH_SIZE = 16
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE
FORMAT_VERSION = "deco_t2i_shared_patch_pca_pooled_uncentered_direct_v1"
FLATTEN_ORDER = "chw_unfold_v1"


def split_quota(total: int, rank: int, world_size: int) -> int:
    return total // world_size + int(rank < total % world_size)


def source_from_url(url: str) -> str:
    for source in (
        "BLIP3o-Pretrain-Long-Caption",
        "BLIP3o-Pretrain-Short-Caption",
        "BLIP3o-Pretrain-JourneyDB",
    ):
        if source in url:
            return source
    return "unknown"


def as_strings(value, length: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-images",
        type=int,
        default=51200,
        help="Total image views across all resolutions, matching ImageNet n=51200.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=(256,),
        help="Training preprocessing streams to pool into one patch distribution.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026090801)
    parser.add_argument("--progress-every", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; never overwrite existing statistics")
    if (int(os.environ.get("WORLD_SIZE", "1")), args.num_images,
            list(args.resolutions), args.batch_size, args.num_workers, args.seed) != (
            8, 51200, [256], 8, 8, 2026090801):
        raise ValueError("Historical PCA protocol requires 8 ranks, 51200 images, resolution256, batch8, workers8, seed2026090801")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal DDP statistic collection requires an existing GPU allocation.")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    source_dirs = [
        args.data_root / "BLIP3o-Pretrain-Long-Caption",
        args.data_root / "BLIP3o-Pretrain-Short-Caption",
        args.data_root / "BLIP3o-Pretrain-JourneyDB",
    ]
    missing = [str(path) for path in source_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing DeCo training sources: {missing}")

    # The total ImageNet-matched image budget is divided across active DeCo
    # preprocessing streams.  Resolution never changes the model family: all
    # emitted patches enter the same direct uncentered moment accumulator.
    resolution_totals = {}
    remaining = args.num_images
    for index, resolution in enumerate(args.resolutions):
        slots = len(args.resolutions) - index
        count = remaining // slots
        resolution_totals[int(resolution)] = count
        remaining -= count

    gram = torch.zeros((PATCH_DIM, PATCH_DIM), dtype=torch.float64, device=device)
    patch_sum = torch.zeros(PATCH_DIM, dtype=torch.float64, device=device)
    patch_count = torch.zeros((), dtype=torch.int64, device=device)
    image_count = torch.zeros((), dtype=torch.int64, device=device)
    image_count_by_resolution = torch.zeros(
        len(args.resolutions), dtype=torch.int64, device=device
    )
    source_order = (
        "BLIP3o-Pretrain-Long-Caption",
        "BLIP3o-Pretrain-Short-Caption",
        "BLIP3o-Pretrain-JourneyDB",
        "unknown",
    )
    source_counts = torch.zeros(len(source_order), dtype=torch.int64, device=device)

    for resolution_index, resolution in enumerate(args.resolutions):
        resolution_seed = seed + 100_003 * resolution_index
        random.seed(resolution_seed)
        np.random.seed(resolution_seed % (2**32))
        torch.manual_seed(resolution_seed)
        torch.cuda.manual_seed_all(resolution_seed)
        total_for_resolution = resolution_totals[int(resolution)]
        local_target = split_quota(total_for_resolution, rank, world_size)
        if local_target == 0:
            continue

        # This is the active, patched DeCo WDS path. It performs RGB conversion,
        # size filtering, BOX half-downsampling, bicubic short-side resize,
        # center crop, ToTensor, and Normalize(0.5, 0.5) to [-1,1].
        dataset = WebDatasetPackedDataset(
            urls=[str(path) for path in source_dirs],
            resolution=int(resolution),
            random_crop=False,
            shuffle_buffer=1000,
            sample_shuffle=True,
            repeat=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        accepted = 0
        for images, _captions, metadata in loader:
            keep = min(images.shape[0], local_target - accepted)
            if keep <= 0:
                break
            images = images[:keep].to(device=device, non_blocking=True)

            # All native non-overlapping patches are used. There is deliberately
            # no per-image K or patch subsampling.
            patches = F.unfold(
                images, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
            ).transpose(1, 2)
            q = patches.reshape(-1, PATCH_DIM).to(dtype=torch.float64)
            gram.addmm_(q.transpose(0, 1), q)
            patch_sum += q.sum(dim=0)
            patch_count += q.shape[0]
            image_count += keep
            image_count_by_resolution[resolution_index] += keep

            urls = as_strings(metadata.get("wds_url", ""), images.shape[0])[:keep]
            local_sources = Counter(source_from_url(url) for url in urls)
            for source_index, source in enumerate(source_order):
                source_counts[source_index] += local_sources.get(source, 0)

            accepted += keep
            if rank == 0 and accepted % args.progress_every < keep:
                print(
                    json.dumps(
                        {
                            "resolution": int(resolution),
                            "rank0_images": accepted,
                            "rank0_target": local_target,
                            "rank0_patches_total": int(patch_count.item()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if accepted >= local_target:
                break

    for tensor in (
        gram,
        patch_sum,
        patch_count,
        image_count,
        image_count_by_resolution,
        source_counts,
    ):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        expected_images = int(args.num_images)
        if int(image_count.item()) != expected_images:
            raise RuntimeError(
                f"Image count mismatch: {int(image_count.item())} != {expected_images}"
            )
        expected_patches = sum(
            resolution_totals[int(resolution)]
            * (int(resolution) // PATCH_SIZE) ** 2
            for resolution in args.resolutions
        )
        if int(patch_count.item()) != expected_patches:
            raise RuntimeError(
                f"Patch count mismatch: {int(patch_count.item())} != {expected_patches}"
            )

        # Direct uncentered second moment. patch_sum/mean is retained for audit
        # only and is never subtracted from this matrix.
        second_moment = gram / patch_count.to(torch.float64)
        second_moment = 0.5 * (second_moment + second_moment.transpose(0, 1))
        raw_eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        order = torch.arange(PATCH_DIM - 1, -1, -1, device=device)
        raw_eigenvalues = raw_eigenvalues[order].contiguous()
        eigenvectors = eigenvectors[:, order].contiguous()
        eigenvalues = raw_eigenvalues.clamp_min(0.0)
        mean = patch_sum / patch_count.to(torch.float64)

        payload = {
            "format_version": FORMAT_VERSION,
            "covariance_kind": "uncentered_second_moment_direct",
            "moment_formula": "M=sum_over_all_native_patches(p p^T)/num_patches; no_mean_subtraction",
            "img_size": 0,
            "resolutions": [int(value) for value in args.resolutions],
            "resolution_image_counts": {
                str(int(resolution)): int(image_count_by_resolution[index].item())
                for index, resolution in enumerate(args.resolutions)
            },
            "patch_size": PATCH_SIZE,
            "channels": 3,
            "patch_dim": PATCH_DIM,
            "flatten_order": FLATTEN_ORDER,
            "num_images": int(image_count.item()),
            "num_patches": int(patch_count.item()),
            "selection_seed": int(args.seed),
            "source_image_counts": {
                source: int(source_counts[index].item())
                for index, source in enumerate(source_order)
            },
            "preprocessing": (
                "active_DeCo_WebDataset_RGB_size_filter_BOX_bicubic_"
                "center_crop_ToTensor_Normalize_minus1_plus1"
            ),
            "mean_fp64_audit_only": mean.cpu(),
            "eigenvalues_fp64_desc": eigenvalues.cpu(),
            "raw_eigenvalues_fp64_desc": raw_eigenvalues.cpu(),
            "eigenvalues_fp32_desc": eigenvalues.float().cpu(),
            "eigenvectors_fp32_desc": eigenvectors.float().cpu(),
            "trace": float(eigenvalues.sum().item()),
            "min_raw_eigenvalue": float(raw_eigenvalues.min().item()),
            "max_raw_eigenvalue": float(raw_eigenvalues.max().item()),
            "negative_eigenvalue_count": int((raw_eigenvalues < 0).sum().item()),
            "world_size": int(world_size),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, args.output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "num_images": payload["num_images"],
                    "num_patches": payload["num_patches"],
                    "resolution_image_counts": payload["resolution_image_counts"],
                    "source_image_counts": payload["source_image_counts"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
