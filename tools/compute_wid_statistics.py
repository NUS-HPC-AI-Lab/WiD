"""Reproduce the ImageNet statistics used by WiD. Run from the repository root.

PCA follows the original TWO eigendecompositions, including the intermediate
FP32 eigenvectors. A direct eigendecomposition of E[qq^T] is not identical.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "c2i"))
from src.wid_data import CustomINImageFolderDataset
from pixdit_core.wipd import dct2d_channel_basis, spectral_fp32_context


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def derive_uncentered(centered):
    """Historical centered-FP32-basis -> uncentered prior conversion."""
    mean = centered["mean_fp64"].double().cpu()
    values = centered["eigenvalues_fp64_desc"].double().cpu()
    basis = centered["eigenvectors_fp32_desc"].double().cpu()
    cov = (basis * values[None, :]) @ basis.T
    moment = cov + mean[:, None] * mean[None, :]
    values, basis = torch.linalg.eigh((moment + moment.T) * .5)
    raw_min, raw_max = float(values.min()), float(values.max())
    negative_count = int((values < 0).sum())
    values, basis = values.flip(0).clamp_min(0).contiguous(), basis.flip(1).contiguous()
    result = dict(centered)
    result.update(format_version="shared_patch_pca_uncentered_second_moment_v1",
                  covariance_kind="uncentered_second_moment",
                  eigenvalues_fp64_desc=values, eigenvalues_fp32_desc=values.float(),
                  eigenvectors_fp32_desc=basis.float(), trace=float(values.sum()),
                  explained_variance_ratio=(values / values.sum().clamp_min(torch.finfo(values.dtype).tiny)).float(),
                  max_raw_eigenvalue=raw_max, min_raw_eigenvalue=raw_min,
                  negative_eigenvalue_count=negative_count,
                  derivation_formula="M=C+mu_muT; eig_fp64; store_evec_fp32_v1")
    return result


def dct_payload(uncentered, source_sha256):
    """DCT coefficient second moments diag(D^T M D), using the PCA-stored M."""
    if uncentered.get("covariance_kind") != "uncentered_second_moment":
        raise ValueError("DCT initialization requires M, not centered covariance")
    n, c = uncentered["patch_size"], uncentered["channels"]
    basis = dct2d_channel_basis(n, c)
    with spectral_fp32_context(basis):
        rotation = uncentered["eigenvectors_fp32_desc"].T @ basis
        spectrum = (rotation.square() * uncentered["eigenvalues_fp32_desc"][:, None]).sum(0)
    return dict(format_version="shared_patch_dct_uncentered_second_moment_v1",
                covariance_kind="uncentered_second_moment", basis=basis, spectrum=spectrum,
                spectrum_definition="diag(D^T M D), M=E[q q^T]",
                patch_size=n, channels=c, patch_dim=c*n*n, flatten_order="chw_unfold_v1",
                ordering="kh^2+kw^2,kh,kw,channel", source_sha256=source_sha256,
                img_size=uncentered["img_size"], num_images=uncentered["num_images"],
                num_patches=uncentered["num_patches"], selection_seed=uncentered["selection_seed"],
                selection_sha256=uncentered["selection_sha256"])


def centered_payload(sum_q, sum_qq, count, metadata):
    mean = sum_q / count
    cov = sum_qq/count - torch.outer(mean, mean)
    cov = (cov + cov.T) * .5
    values, vectors = torch.linalg.eigh(cov)
    values, vectors = values.flip(0).contiguous(), vectors.flip(1).contiguous()
    return dict(metadata, format_version="shared_patch_pca_centered_population_v1",
                covariance_kind="centered_population", mean_fp64=mean.cpu(),
                eigenvalues_fp64_desc=values.cpu(), eigenvalues_fp32_desc=values.float().cpu(),
                eigenvectors_fp32_desc=vectors.float().cpu(), trace=float(cov.trace()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data-dir", required=True, help="ImageFolder training split (1000 class folders)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-images", type=int, default=51200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["pca_centered.pt", "pca_uncentered.pt", "dct_patch.pt", "manifest.json"]
    if any((out/n).exists() for n in names):
        raise FileExistsError("Statistics output already exists; use a new directory")
    if a.resolution % a.patch_size or a.num_images < 1:
        raise ValueError("Invalid image count or patch geometry")
    device = torch.device(a.device)
    dataset = CustomINImageFolderDataset(a.data_dir, a.resolution)
    if a.num_images > len(dataset):
        raise ValueError("num-images exceeds dataset length")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(a.seed))[:a.num_images]
    samples = dataset.dataset.samples
    selected_paths = [Path(samples[i][0]).relative_to(a.data_dir).as_posix() for i in indices.tolist()]
    selection_sha = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    metadata = dict(img_size=a.resolution, patch_size=a.patch_size, channels=3,
                    patch_dim=3*a.patch_size**2, flatten_order="chw_unfold_v1",
                    num_images=a.num_images, num_patches=a.num_images*(a.resolution//a.patch_size)**2,
                    selection_seed=a.seed, selected_indices=indices,
                    selection_sha256=selection_sha,
                    class_histogram=torch.bincount(torch.tensor([samples[i][1] for i in indices]), minlength=len(dataset.dataset.classes)),
                    preprocessing=f"center_crop_{a.resolution}; ToTensor; x=2*img-1; no augmentation")
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=a.batch_size,
                        shuffle=False, num_workers=a.workers, drop_last=False)
    dim = metadata["patch_dim"]
    sum_q = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_qq = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    count, images_seen = 0, 0
    for images, _, _ in loader:
        images = images.to(device)
        q = F.unfold(images, a.patch_size, stride=a.patch_size).transpose(1, 2).reshape(-1, dim).double()
        sum_q += q.sum(0)
        sum_qq += q.T @ q
        count += q.shape[0]
        images_seen += images.shape[0]
        print(f"images={images_seen}/{a.num_images} patches={count}", flush=True)
    if count != metadata["num_patches"] or images_seen != a.num_images:
        raise RuntimeError("Statistics sample count mismatch")
    centered = centered_payload(sum_q, sum_qq, count, metadata)
    torch.save(centered, out/"pca_centered.pt")
    uncentered = derive_uncentered(centered)
    uncentered["source_stats_sha256"] = digest(out/"pca_centered.pt")
    torch.save(uncentered, out/"pca_uncentered.pt")
    torch.save(dct_payload(uncentered, digest(out/"pca_uncentered.pt")), out/"dct_patch.pt")
    manifest = dict(protocol="wid_statistics_v1", selection_sha256=selection_sha,
                    selected_relative_paths=selected_paths, config=vars(a),
                    torch_version=torch.__version__, source_sha256=digest(__file__),
                    sha256={n:digest(out/n) for n in names if n.endswith(".pt")})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
