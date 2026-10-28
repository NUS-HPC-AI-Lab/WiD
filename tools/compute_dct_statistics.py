"""Global DCT RGB second moments, following the final DCT-M experiment.

Scan ImageFolder images, or reproduce the final matrix from archived sufficient
statistics. Preserve the historical FP32 mean/covariance storage followed by
FP64 C + mu mu^T conversion; directly storing the FP64 raw moment differs in
roundoff. The returned channel_second_moment, NOT channel_covariance, is the prior.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c2i"))
from src.wid_data import center_crop_arr

FORMAT_VERSION = "global_dct_channel_uncentered_second_moment_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_channel_moment(payload):
    """Exactly the matrix conversion used by the final DCT-M runs."""
    mean = payload["mean"].double().cpu()
    covariance = payload["channel_covariance"].double().cpu()
    n = int(payload["img_size"])
    if mean.shape != (3, n, n) or covariance.shape != (n, n, 3, 3):
        raise ValueError("Expected CHW mean and HW33 channel covariance")
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise ValueError("Non-finite DCT statistics")
    mean_hw = mean.permute(1, 2, 0)
    moment = covariance + mean_hw.unsqueeze(-1) * mean_hw.unsqueeze(-2)
    moment = .5 * (moment + moment.transpose(-1, -2))
    result = dict(payload)
    result.update(format_version=FORMAT_VERSION,
                  channel_matrix_kind="uncentered_second_moment",
                  channel_second_moment=moment.float().contiguous(),
                  derivation_formula="M_k=C_k+mu_k*mu_k^T",
                  build_dtype="float64_then_store_float32")
    return result


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return center_crop_arr(image, self.size)


def compute_from_images(args):
    dataset = datasets.ImageFolder(args.data_dir, transform=transforms.Compose([
        CenterCrop(args.resolution), transforms.ToTensor()]))
    if not 0 < args.num_images <= len(dataset):
        raise ValueError("Invalid requested sample count")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_images]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    device, n = torch.device(args.device), args.resolution
    # Historical DCT matrix construction is FP64 on CPU, even for a GPU scan.
    coords = torch.arange(n, dtype=torch.float64)
    freqs = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    dct = torch.cos(math.pi/n * (coords+.5).unsqueeze(0) * freqs) * math.sqrt(2.0/n)
    dct[0] /= math.sqrt(2.0)
    dct = dct.to(device)
    square = torch.zeros(3, n, n, dtype=torch.float64, device=device)
    total = torch.zeros_like(square)
    outer = torch.zeros(n, n, 3, 3, dtype=torch.float64, device=device)
    count = 0
    for images, _ in loader:
        # Match the original: ToTensor FP32 -> FP64 -> scale to [-1, 1].
        x = images.to(device=device, dtype=torch.float64) * 2.0 - 1.0
        coeff = (dct @ x.flatten(0, 1) @ dct.T).view(-1, 3, n, n)
        square += coeff.pow(2).sum(0)
        total += coeff.sum(0)
        hw = coeff.permute(0, 2, 3, 1).contiguous()
        outer += torch.einsum("bhwc,bhwd->hwcd", hw, hw)
        count += len(x)
        if count % 512 == 0:
            print(f"images={count}/{args.num_images}", flush=True)
    assert count == args.num_images
    spectrum, mean = (square/count).float().cpu(), (total/count).float().cpu()
    mean_hw = mean.permute(1, 2, 0).double().to(device)
    cov = outer/count - torch.einsum("hwc,hwd->hwcd", mean_hw, mean_hw)
    cov = (.5 * (cov+cov.transpose(-1, -2))).float().cpu()
    sufficient = dict(spectrum=spectrum, mean=mean,
        centered_spectrum=(spectrum.double()-mean.double().pow(2)).clamp_min(0).float(),
        channel_covariance=cov, img_size=n, num_images=count, channels="rgb",
        convention="x = 2*img - 1; orthonormal DCT-II; spectrum=E[x_k^2]; mean=E[x_k]; centered_spectrum=Var[x_k]; channel_covariance=Cov_rgb[x_k]",
        data_path=args.data_dir, seed=args.seed)
    selection = dict(dataset_length=len(dataset), selected_indices=indices.tolist(),
        selected_relative_paths=[Path(dataset.samples[i][0]).relative_to(args.data_dir).as_posix() for i in indices.tolist()],
        selection_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest())
    return sufficient, selection


def main():
    p = argparse.ArgumentParser(__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data-dir", help="ImageNet training split in ImageFolder order")
    inputs.add_argument("--input-statistics", help="Archived mean and centered RGB covariance")
    p.add_argument("--output", required=True)
    p.add_argument("--num-images", type=int, default=51200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cpu", help="CPU matches the original scan; CUDA may differ in roundoff")
    args = p.parse_args()
    out = Path(args.output)
    manifest_path = out.with_suffix(out.suffix+".json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(out)
    if args.input_statistics:
        source = Path(args.input_statistics)
        sufficient = torch.load(source, map_location="cpu", weights_only=True)
        result = derive_channel_moment(sufficient)
        result.update(source_stats_path=str(source.resolve()), source_stats_sha256=digest(source))
        selection = dict(source_stats_sha256=digest(source))
    else:
        sufficient, selection = compute_from_images(args)
        result = derive_channel_moment(sufficient)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    manifest = dict(config=vars(args), torch_version=torch.__version__,
        matrix_key="channel_second_moment", matrix_kind="uncentered_second_moment",
        output_sha256=digest(out), source_code_sha256=digest(__file__), **selection)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {out}; channel_matrix_kind={result['channel_matrix_kind']}", flush=True)


if __name__ == "__main__":
    main()
