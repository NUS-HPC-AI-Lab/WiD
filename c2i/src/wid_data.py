"""ImageNet preprocessing and consumed-batch sampler state for WiD runs."""

import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def center_crop_arr(pil_image, image_size):
    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), Image.Resampling.BOX)
    scale = image_size / min(pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    y, x = (arr.shape[0] - image_size)//2, (arr.shape[1] - image_size)//2
    return arr[y:y+image_size, x:x+image_size]


class CustomINImageFolderDataset(Dataset):
    def __init__(self, data_dir: str, resolution: int = 256):
        self.resolution = resolution
        self.dataset = datasets.ImageFolder(data_dir)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        raw = self.to_tensor(center_crop_arr(image.convert("RGB"), self.resolution))
        return raw * 2 - 1, int(label), {"raw_image": raw, "class": int(label)}


class ResumableDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Skip consumed rank-local samples once; keep structural length constant."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume = None

    def set_resume(self, epoch, sample_offset):
        if not 0 <= sample_offset <= self.num_samples:
            raise ValueError("Invalid rank-local sampler offset")
        self._resume = (int(epoch), int(sample_offset))
        self.epoch = int(epoch)

    def set_epoch(self, epoch):
        self.epoch = self._resume[0] if self._resume is not None else int(epoch)

    def __iter__(self):
        if self._resume is None:
            return super().__iter__()
        self.epoch, offset = self._resume
        indices = list(super().__iter__())[offset:]
        self._resume = None
        return iter(indices)


class ExactResumeDataMixin:
    def _local_train_batch_size(self):
        micro = getattr(self.train_dataset, "micro_batch_size", 1)
        return self.train_batch_size // micro

    def _batches_per_epoch(self, world):
        return math.ceil(len(self.train_dataset)/world) // self._local_train_batch_size()

    def state_dict(self):
        if self.train_dataset is None:
            return {}
        world = self.trainer.world_size
        n = self._batches_per_epoch(world)
        count = self._train_batches_seen_total
        return dict(format="pixeldit_resumable_sampler_v1", world_size=world,
                    dataset_length=len(self.train_dataset), local_batch_size=self._local_train_batch_size(),
                    batches_per_epoch=n, sampler_epoch=count//n, consumed_batches_in_epoch=count%n,
                    global_step=self.trainer.global_step, consumed_batches_total=count,
                    accumulate_grad_batches=self.trainer.accumulate_grad_batches,
                    sampler_seed=int(self.seed), dataloader_drop_last=True)

    def load_state_dict(self, state):
        if not state:
            return
        # Standalone prediction must not restore training position.
        if self.trainer.state.fn in ("predict", "validate", "test"):
            return
        if self.train_dataset is None:
            raise RuntimeError("Training dataset missing during resume")
        expected = self.state_dict()
        for key in ("format", "world_size", "dataset_length", "local_batch_size",
                    "batches_per_epoch", "accumulate_grad_batches", "sampler_seed", "dataloader_drop_last"):
            if state.get(key) != expected[key]:
                raise RuntimeError(f"Resume contract changed: {key}: {state.get(key)} != {expected[key]}")
        total = int(state["consumed_batches_total"])
        epoch, offset = divmod(total, expected["batches_per_epoch"])
        if (epoch, offset) != (state["sampler_epoch"], state["consumed_batches_in_epoch"]):
            raise RuntimeError("Inconsistent consumed-batch counters")
        self._train_batches_seen_total = total
        self._pending_sampler_resume = (epoch, offset * self._local_train_batch_size())
        self._apply_pending_sampler_resume()

    def _apply_pending_sampler_resume(self):
        if self._pending_sampler_resume is not None and self._train_dataloader is not None:
            self._train_dataloader.sampler.set_resume(*self._pending_sampler_resume)
            self._pending_sampler_resume = None
