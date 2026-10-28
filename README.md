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

This branch implements **WiD** on [PixelDiT](https://github.com/NVlabs/PixelDiT) for class conditional generation.
Please switch to the [`t2i` branch](https://github.com/NUS-HPC-AI-Lab/WiD/tree/t2i) for text-to-image generation code based on DeCo.

We propose Wiener-informed Decoding (WiD), which uses statistical knowledge both for direct prediction and to organize decoder computation without modifying native blocks. Across model scales and decoder architectures, WiD accelerates training and improves generation quality with modest parameter overhead.

- **1.53 FID** on ImageNet 256×256
- **1.64 FID** on ImageNet 512×512
- **0.86** on GenEval 512×512
- **82.6** on DPG-Bench 512×512

## Environment

Use Python 3.12.8 on Linux. The requirements pin the H200 training environment (PyTorch 2.5.1, CUDA 12.4).

```bash
pip install -r requirements.txt
```

For B200 hardware, please use a Blackwell-compatible PyTorch/CUDA environment instead.

## Class-to-Image Generation (ImageNet)

PixelDiT-XL+WiD class-conditioned image generation on ImageNet.

### Pre-trained Models

The following EMA FID-50K results and sampling settings are reported in [our paper](https://arxiv.org/abs/2610.36192). Checkpoints are available in our [Hugging Face repository](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/tree/main).

| Model | Epochs | Resolution | gFID | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|
| PixelDiT-XL+WiD | 80 | 256×256 | 2.09 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet256_wid_pixeldit_xl_epoch80.ckpt) |
| PixelDiT-XL+WiD | 160 | 256×256 | 1.87 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet256_wid_pixeldit_xl_epoch160.ckpt) |
| PixelDiT-XL+WiD | 260 | 256×256 | 1.61 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet256_wid_pixeldit_xl_epoch260.ckpt) |
| PixelDiT-XL+WiD | 320 | 256×256 | 1.58 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet256_wid_pixeldit_xl_epoch320.ckpt) |
| PixelDiT-XL+WiD | 500 | 256×256 | 1.53 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet256_wid_pixeldit_xl_epoch500.ckpt) |
| PixelDiT-XL+WiD | 260+50 | 512×512 | 1.64 | [🤗 HuggingFace](https://huggingface.co/ziqiaow/Wiener-informed-Decoding/resolve/main/imagenet512_wid_pixeldit_xl_epoch50.ckpt) |


### Data Preparation

#### Statistics Computation

Ready-to-use statistics are included in [statistics](statistics/).

| File | Statistic | Use |
|---|---|---|
| [pca_uncentered.pt](statistics/pca_uncentered.pt) | Eigendecomposition of the shared 768×768 patch moment | Exact prior used to initialize the released 256/512 WiD models |
| [dct_image.pt](statistics/dct_image.pt) | Full-image DCT-II, one 3×3 RGB moment per frequency | Exact prior used by the DCT baseline experiments |

To recompute PCA statistics from ImageNet, run from the repository root:

```bash
python tools/compute_wid_statistics.py \
  --data-dir /path/to/imagenet/train \
  --output-dir statistics_recomputed/pca \
  --num-images 51200 --seed 0 --resolution 256 --patch-size 16 \
  --batch-size 32 --workers 12 --device cuda
```

To use a recomputed prior, set `prior_path` for both `spectral_head` and `pit_spectral_anchor` in the WiD config.

#### Data Preprocessing
We follow the preprocessing instructions in [PixelDiT](https://github.com/NVlabs/PixelDiT).

### Training

#### ImageNet 256×256

 Training uses 8 GPUs, batch size 32 per GPU (global batch 256), BF16, EMA decay 0.9999, and REPA weight 0.5.
 We also provide the [training logs](training_logs/imagenet256_0_to_2500k) on ImageNet 256×256. You will be able to reproduce our training with the same hardware stated in our paper.

**0 → 800K:**

```bash
cd c2i/
torchrun --standalone --nproc_per_node=8 main.py fit \
  -c configs/wid/pix256_xl.yaml \
  --data.train_dataset.init_args.data_dir=/path/to/imagenet/train \
  --seed=0 --per_run_seed=false \
  --trainer.max_steps=800000 \
  --tags.exp=WiD-XL-256-initial
```

**800K → 2.5M:** reduce the learning rate to 1e-5 and gradient clipping to 0.5.

```bash
torchrun --standalone --nproc_per_node=8 main.py fit \
  -c configs/wid/pix256_xl.yaml \
  --data.train_dataset.init_args.data_dir=/path/to/imagenet/train \
  --ckpt_path=/path/to/step800000.ckpt \
  --seed=0 --per_run_seed=false \
  --model.optimizer.init_args.lr=1e-5 --model.override_lr_on_resume=1e-5 \
  --trainer.gradient_clip_val=0.5 --trainer.limit_val_batches=0 \
  --trainer.max_steps=2500000 \
  --tags.exp=WiD-XL-256-from800K
```

#### ImageNet 512×512

```bash
cd c2i/
torchrun --standalone --nproc_per_node=8 main.py fit \
  -c configs/wid/pix512_xl.yaml \
  --data.train_dataset.init_args.data_dir=/path/to/imagenet/train \
  --model.init_from=/path/to/step1300000.ckpt \
  --seed=0 --per_run_seed=false \
  --trainer.max_steps=250000 \
  --tags.exp=WiD-XL-512-from1300K
```


#### Resume from Checkpoint

```bash
cd c2i/
torchrun --standalone --nproc_per_node=8 main.py fit \
  -c configs/wid/pix512_xl.yaml \
  --data.train_dataset.init_args.data_dir=/path/to/imagenet/train \
  --ckpt_path=/path/to/512_stage_checkpoint.ckpt \
  --per_run_seed=false
```

### Evaluation

Evaluation generates 50K images via `main.py predict`, then computes FID using the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).

#### Step 1: Generate Samples

Prediction uses EMA weights by default. Run each command from `c2i/`.

**ImageNet 256×256 (Epoch 500):**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/wid/pix256_xl.yaml \
  --ckpt_path=imagenet256_wid_pixeldit_xl_epoch500.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=4.0 \
  --model.diffusion_sampler.init_args.timeshift=1.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.14 \
  --model.diffusion_sampler.init_args.guidance_interval_max=0.55 \
  --data.pred_batch_size=32 --data.pred_num_workers=1 \
  --per_run_seed=false --seed_everything=1600 \
  --tags.exp=WiD-XL-256-epoch500-cfg4-i014-055
```

**ImageNet 512×512:**

```bash
cd c2i/
torchrun --nproc_per_node=8 main.py predict \
  -c configs/wid/pix512_xl.yaml \
  --ckpt_path=imagenet512_wid_pixeldit_xl_epoch50.ckpt \
  --model.diffusion_sampler.class_path=src.diffusion.FlowDPMSolverSampler \
  --model.diffusion_sampler.init_args.num_steps=100 \
  --model.diffusion_sampler.init_args.guidance=3.5 \
  --model.diffusion_sampler.init_args.timeshift=2.0 \
  --model.diffusion_sampler.init_args.guidance_interval_min=0.1 \
  --model.diffusion_sampler.init_args.guidance_interval_max=0.9 \
  --data.pred_batch_size=16 --data.pred_num_workers=1 \
  --per_run_seed=false --seed_everything=10000 \
  --tags.exp=WiD-XL-512-epoch50-cfg35-i01-09
```

##### Sampler Settings Summary

| Setting | 80 ep (256) | 160 ep (256) | 260 ep (256) | 320 ep (256) | 500 ep (256) | 260+50 ep (512) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CFG Scale | 2.90 | 2.60 | 3.25 | 4.00 | 4.00 | 3.50 |
| CFG Interval | [0.10, 1.00] | [0.10, 1.00] | [0.12, 0.70] | [0.14, 0.55] | [0.14, 0.55] | [0.10, 0.90] |
| Time Shift | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 |
| Seed | 1600 | 1600 | 1600 | 1600 | 1600 | 10000 |



#### Step 2: Compute FID

After generating samples, compute FID with the [ADM evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Samples are written to `output/wid/exp_<tag>/validation/predict/output.npz`. 
Use the matching reference resolution: [ImageNet-256 reference](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) or [ImageNet-512 reference](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz).


```bash
# Run from guided-diffusion/evaluations; use the actual generated NPZ path.
python evaluator.py /path/to/VIRTUAL_imagenet256_labeled.npz /path/to/256/output.npz
python evaluator.py /path/to/VIRTUAL_imagenet512.npz /path/to/512/output.npz
```

The evaluator reports FID, sFID, Inception Score, Precision, and Recall.

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
