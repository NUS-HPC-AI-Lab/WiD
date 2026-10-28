<h2 align="center">Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models</h2>

<p align="center">
  Ziqiao Wang, &nbsp;
  <a href="https://www.comp.nus.edu.sg/~youy/">Yang You</a><sup></sup>
</p>
<p align="center">
  <a href="https://ai.comp.nus.edu.sg/">NUS HPC-AI Lab</a>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2610.36192"><img src="https://img.shields.io/badge/Arxiv-2610.36192-b31b1b.svg?logo=arXiv" alt="arXiv" /></a>
  &nbsp;
  <a href="https://huggingface.co/ziqiaow/Wiener-informed-Decoding/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model" /></a>
</p>


![t2i samples](assets/t2i_samples.png)
![C2i samples](assets/c2i_samples.png)

This branch implements **WiD** on [DeCo](https://github.com/Zehong-Ma/DeCo) for text-to-image generation.
Please switch to the [`c2i` branch](https://github.com/NUS-HPC-AI-Lab/WiD/tree/c2i) for class-conditional image generation code based on PixelDiT.

We propose Wiener-informed Decoding (WiD), which uses statistical knowledge both for direct prediction and to organize decoder computation without modifying native blocks. Across model scales and decoder architectures, WiD accelerates training and improves generation quality with modest parameter overhead.

- **1.53 FID** on ImageNet 256×256
- **1.64 FID** on ImageNet 512×512
- **0.86** on GenEval 512×512
- **82.6** on DPG-Bench 512×512


## Environment

Use Linux, Python 3.12.3 and eight H200 GPUs for the training configuration below.
The requirements pin PyTorch 2.7.1, CUDA 12.6 and Lightning 2.4.

```bash
pip install -r requirements.txt
```

## Data Preparation

Set these paths to your local directories:

```bash
export DATA_ROOT=/path/to/data
export PRETRAINED_ROOT=/path/to/pretrained
export OUTPUT_ROOT=/path/to/output
```

### Training Data

Pretraining uses the three BLIP3o WebDataset collections; SFT uses BLIP3o-60k.
Download the tar archives:

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

datasets = {
    "BLIP3o-Pretrain-Long-Caption": "e4d07091a466d1a1e35a9b0c61caddc78d14a059",
    "BLIP3o-Pretrain-Short-Caption": "e84d184540a6cc4fc3b1ba5528bb45dcc6b3fb0e",
    "BLIP3o-Pretrain-JourneyDB": "734b390be883b2021ae6d6dc1b24e6b97e7253a2",
    "BLIP3o-60k": "f7316b0aa446338ee1707484924aa59457b4bbf3",
}
for name, revision in datasets.items():
    snapshot_download(
        repo_id=f"BLIP3o/{name}", repo_type="dataset",
        revision=revision, allow_patterns=["*.tar"],
        local_dir=Path(os.environ["DATA_ROOT"]) / name,
    )
```

Extract the SFT image/text pairs without changing their bytes or filenames:

```bash
python tools/prepare_sft.py --input "$DATA_ROOT/BLIP3o-60k" \
  --output "$DATA_ROOT/BLIP3o-60k-extracted"
```

The expected directory layout is:

```text
data/
├── BLIP3o-Pretrain-Long-Caption/
├── BLIP3o-Pretrain-Short-Caption/
├── BLIP3o-Pretrain-JourneyDB/
├── BLIP3o-60k/
└── BLIP3o-60k-extracted/
```

### Frozen Encoders

Download Qwen3-1.7B at the specified revision:

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    local_dir=Path(os.environ["PRETRAINED_ROOT"]) / "Qwen/Qwen3-1.7B",
)
```

Download DINOv2 and its official ViT-B/14 weights:

```bash
DINO_ROOT="$PRETRAINED_ROOT/dinov2"
mkdir -p "$DINO_ROOT"
git clone https://github.com/facebookresearch/dinov2.git \
  "$DINO_ROOT/facebookresearch_dinov2_main"
git -C "$DINO_ROOT/facebookresearch_dinov2_main" \
  checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
curl -fL https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
  -o "$DINO_ROOT/dinov2_vitb14_pretrain.pth"
```

### Statistics Computation

The ready-to-use prior is [statistics/pca_uncentered.pt](statistics/pca_uncentered.pt).
To recompute it, run from the repository root:

```bash
torchrun --standalone --nproc_per_node=8 tools/compute_wid_statistics.py \
  --data-root "$DATA_ROOT" --output "$OUTPUT_ROOT/pca_uncentered.pt"
```
To train with recomputed statistics, pass `--prior /path/to/pca_uncentered.pt` to the training launcher below.

## Training

Text-to-image checkpoint available [here](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/tree/main).

| Stage | Updates | Resolution | Batch/GPU | Accumulation | Global batch | LR | Adam betas |
|---|---:|---:|---:|---:|---:|---:|---|
| Pretraining | 200K | 256 | 192 | 1 | 1536 | 2e-4 | (0.9, 0.95) |
| Continued pretraining | 80K | 512 | 16 | 4 | 512 | 1e-4 | (0.9, 0.95) |
| SFT | 40K | 512 | 16 | 4 | 512 | 1e-5 | (0.9, 0.999) |

All stages use BF16, AdamW with zero weight decay, gradient clipping at 1.0, EMA decay 0.9999 and REPA weight 0.5. Training time shift is 2.0 at 256 and 4.0 at 512.

### 256×256 Pretraining

```bash
python tools/train.py pretrain_256 --data-root "$DATA_ROOT" \
  --pretrained-root "$PRETRAINED_ROOT" --output "$OUTPUT_ROOT/pretrain256"
```

### 512×512 Continued Pretraining

```bash
python tools/train.py pretrain_512 --data-root "$DATA_ROOT" \
  --pretrained-root "$PRETRAINED_ROOT" --output "$OUTPUT_ROOT/pretrain512" \
  --checkpoint /path/to/stage1-200k.ckpt
```

### 512×512 SFT

```bash
python tools/train.py sft_512 --data-root "$DATA_ROOT" \
  --pretrained-root "$PRETRAINED_ROOT" --output "$OUTPUT_ROOT/sft512" \
  --checkpoint /path/to/stage2-280k.ckpt
```

## Acknowledgements
We thank the authors of [PixelDiT](https://github.com/NVlabs/PixelDiT) and [DeCo](https://github.com/Zehong-Ma/DeCo) for sharing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026estimatewienerinformeddecodingeasier,
      title={Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models},
      author={Ziqiao Wang and Yang You},
      year={2026},
      eprint={2610.36192},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.36192},
}
```
