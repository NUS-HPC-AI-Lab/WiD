import os
import shutil
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class CheckpointHook(Callback):
    """Save canonical checkpoints only after scheduled validation."""

    def __init__(
        self,
        every_n_train_steps: int,
        save_top_k: int = -1,
        save_last: bool = True,
        **_ignored,
    ):
        super().__init__()
        if int(every_n_train_steps) <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if int(save_top_k) not in {-1, 0}:
            raise ValueError("Post-validation checkpointing supports save_top_k=-1 or 0")
        self.every_n_train_steps = int(every_n_train_steps)
        self.save_enabled = int(save_top_k) != 0
        self.save_last = bool(save_last)
        self.dirpath: Optional[str] = None
        self._last_saved_step = -1

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        self.dirpath = trainer.default_root_dir
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
        pl_module.strict_loading = False

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        trainer_fn = getattr(trainer.state.fn, "value", trainer.state.fn)
        if trainer_fn != "fit" or trainer.sanity_checking or not self.save_enabled:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_train_steps != 0:
            return
        if step == self._last_saved_step:
            return

        restore = getattr(pl_module, "restore_training_rng_after_validation", None)
        if callable(restore):
            restore()
        path = os.path.join(self.dirpath, f"step={step:09d}-postval.ckpt")
        trainer.save_checkpoint(path, weights_only=False)
        trainer.strategy.barrier()
        if trainer.is_global_zero and self.save_last:
            last_path = os.path.join(self.dirpath, "last-postval.ckpt")
            temporary = last_path + ".tmp"
            shutil.copy2(path, temporary)
            os.replace(temporary, last_path)
            print(f"[deco-checkpoint] canonical post-val checkpoint: {path}", flush=True)
        trainer.strategy.barrier()
        self._last_saved_step = step

    def state_dict(self):
        return {"last_saved_step": int(self._last_saved_step)}

    def load_state_dict(self, state_dict):
        self._last_saved_step = int(state_dict.get("last_saved_step", -1))
