"""Stage transitions and checkpointing for WiD text-to-image training.

All optimizer groups retain their historical ordering. The third sampler group
is empty in these experiments but is retained for checkpoint compatibility.
"""

import os

import lightning.pytorch as pl
import torch

def optimizer_steps(optimizer):
    return {int(v["step"].item()) if isinstance(v["step"], torch.Tensor)
            else int(v["step"]) for v in optimizer.state.values()}


def check_geometry(trainer, batch):
    if (trainer.world_size, trainer.accumulate_grad_batches) != (8, 4):
        raise RuntimeError("Historical 512 stages require 8 ranks and accumulation 4")
    if tuple(batch[0].shape) != (16, 3, 512, 512):
        raise RuntimeError(f"Expected local batch (16,3,512,512), got {batch[0].shape}")
    if trainer.precision != "bf16-mixed":
        raise RuntimeError("Historical stages require bf16-mixed")


class Pretrain512Transition(pl.Callback):
    """Full-state continuation, then override the restored 256-stage LR."""

    def __init__(self):
        super().__init__()
        self.checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.checked:
            return
        check_geometry(trainer, batch)
        step = int(trainer.global_step)
        opt = trainer.optimizers[0]
        if step < 200_000 or len(opt.state) != 290 or optimizer_steps(opt) != {step}:
            raise RuntimeError("512 pretraining requires a full 200K-or-later pretraining checkpoint")
        if [len(g["params"]) for g in opt.param_groups] != [284, 6, 0]:
            raise RuntimeError("Optimizer parameter ordering differs from the historical run")
        for group in opt.param_groups[:2]:
            if tuple(group["betas"]) != (0.9, 0.95):
                raise RuntimeError("Unexpected pretraining Adam betas")
            group["lr"] = 1e-4
        self.checked = True


class SFTTransition(pl.Callback):
    """New Adam state and local counter; preserve loaded online/EMA/REPA weights."""

    def __init__(self):
        super().__init__()
        self.checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.checked:
            return
        check_geometry(trainer, batch)
        opt = trainer.optimizers[0]
        if [len(g["params"]) for g in opt.param_groups] != [284, 6, 0]:
            raise RuntimeError("Unexpected SFT parameter groups")
        if [float(g["lr"]) for g in opt.param_groups] != [1e-5, 1e-5, 1e-3]:
            raise RuntimeError("Unexpected SFT learning rates")
        if any(tuple(g["betas"]) != (0.9, 0.999) for g in opt.param_groups):
            raise RuntimeError("Unexpected SFT Adam betas")
        if int(trainer.global_step) == 0:
            if opt.state or not pl_module._warmstart_loaded_ema:
                raise RuntimeError("Fresh SFT needs warm-started weights and a fresh optimizer")
            # Compatibility with the historical 280K pre-EMA checkpoint. This
            # does not affect training losses (the objective uses the online net).
            if getattr(pl_module, "_wid_init_ema_pending", True):
                pl_module.ema_tracker.ema_step()
                torch.cuda.current_stream().wait_stream(pl_module.ema_tracker._stream)
        elif len(opt.state) != 290 or optimizer_steps(opt) != {int(trainer.global_step)}:
            raise RuntimeError("Incomplete SFT resume state")
        self.checked = True


class TrainingCheckpoint(pl.Callback):
    """Save full state only at newly completed optimizer steps, on every rank."""

    def __init__(self, every_n_train_steps: int = 10_000):
        super().__init__()
        if int(every_n_train_steps) <= 0:
            raise ValueError("every_n_train_steps must be positive")
        self.every_n_train_steps = int(every_n_train_steps)
        self.dirpath = None
        self._last_saved_step = -1

    def setup(self, trainer, pl_module, stage):
        self.dirpath = trainer.default_root_dir
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
        trainer.strategy.barrier()

    def on_train_start(self, trainer, pl_module):
        self._last_saved_step = max(self._last_saved_step, int(trainer.global_step))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= self._last_saved_step or step % self.every_n_train_steps:
            return
        path = os.path.join(self.dirpath, f"step={step:09d}-train.ckpt")
        # Decide on rank zero and broadcast before any writer starts, avoiding
        # a slow rank mistaking this very save for a pre-existing partial file.
        collision = trainer.strategy.broadcast(
            (os.path.exists(path) or os.path.exists(path + ".partial"))
            if trainer.is_global_zero else False
        )
        if collision:
            raise FileExistsError(path)
        self._last_saved_step = step
        trainer.save_checkpoint(path + ".partial", weights_only=False)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            os.replace(path + ".partial", path)
        trainer.strategy.barrier()

    def state_dict(self):
        return {"last_saved_step": int(self._last_saved_step)}

    def load_state_dict(self, state_dict):
        self._last_saved_step = int(state_dict.get("last_saved_step", -1))
