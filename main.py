import os
import json
import torch
import time
from typing import Any, Union

from src.utils.patch_bugs import *
from lightning import Trainer, LightningModule

from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
import lightning.pytorch as pl

import logging
logger = logging.getLogger("lightning.pytorch")
# log_path = os.path.join( f"log.txt")
# logger.addHandler(logging.FileHandler(log_path))


class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class ReWriteRootDirCli(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})

        # predict path & logger check
        if self.subcommand == "predict":
            config_trainer.logger = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)
        parser.add_argument(
            "--fixed_run_seed",
            type=int,
            default=0,
            help=(
                "Pre-registered experiment seed. It must not depend on timestamps or job ids."
            ),
        )
        parser.add_argument("--segment_id", type=int, default=0)
        parser.add_argument(
            "--init_ckpt_path",
            type=str,
            default=None,
            help=(
                "Stage-transition warm start: load model, EMA and trainer weights only, "
                "while constructing a new optimizer and resetting the stage-local step. "
                "Use ckpt_path instead for an exact same-stage resume."
            ),
        )

    @staticmethod
    def _distributed_rank() -> int:
        for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            value = os.environ.get(key)
            if value is not None:
                return int(value)
        return 0

    @staticmethod
    def _derive_segment_seed(fixed_run_seed: int, segment_id: int) -> int:
        # Segment boundaries are part of the protocol.  Never use timestamps,
        # PBS ids or process start order as scientific randomness.
        seed = (int(fixed_run_seed) + 1_000_003 * int(segment_id)) % (2**31 - 1)
        if not 0 <= seed < 2**31 - 1:
            raise ValueError(f"derived segment seed is out of range: {seed}")
        return seed

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        # Lightning's DDP subprocesses all execute this method.  Only global
        # rank zero may enforce the fresh-run directory contract; otherwise
        # ranks 1..N race with rank zero after it creates the directory and
        # incorrectly abort a valid multi-GPU launch with FileExistsError.
        distributed_rank = self._distributed_rank()
        if (
            distributed_rank == 0
            and os.path.exists(default_root_dir)
            and "debug" not in default_root_dir
        ):
            if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume:
                raise FileExistsError(f"{default_root_dir} already exists")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)

        # Lightning's model initialization remains common across ranks through
        # top-level seed_everything. Only after all classes have been built do
        # we create rank-distinct training RNG streams for t/noise/dropout and
        # dataloader workers.
        fixed_run_seed = int(
            self._get(self.config_init, "fixed_run_seed", default=0)
        )
        segment_id = int(self._get(self.config_init, "segment_id", default=0))
        base_seed = self._derive_segment_seed(fixed_run_seed, segment_id)
        rank = self._distributed_rank()
        rank_seed = base_seed + rank
        pl.seed_everything(rank_seed, workers=True)
        os.environ["DECO_FIXED_RUN_SEED"] = str(fixed_run_seed)
        os.environ["DECO_SEGMENT_ID"] = str(segment_id)
        os.environ["DECO_RANK_TRAINING_SEED"] = str(rank_seed)

        if trainer.is_global_zero:
            manifest = {
                "base_seed": base_seed,
                "fixed_run_seed": fixed_run_seed,
                "segment_id": segment_id,
                "world_size": int(trainer.world_size),
                "note": "rank r uses base_seed + r",
            }
            manifest_path = os.path.join(
                default_root_dir, f"rng-segment-{segment_id:04d}.json"
            )
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info(
            "Applied rank-distinct training seed %d (base=%d, rank=%d)",
            rank_seed,
            base_seed,
            rank,
        )
        return trainer

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()
        init_ckpt_path = self._get(self.config_init, "init_ckpt_path", default=None)
        exact_resume_path = self._get(self.config_init, "ckpt_path", default=None)
        if init_ckpt_path and exact_resume_path:
            raise ValueError("init_ckpt_path and ckpt_path are mutually exclusive")
        if init_ckpt_path and self.subcommand == "fit":
            checkpoint = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" not in checkpoint:
                raise RuntimeError(f"No state_dict in warm-start checkpoint: {init_ckpt_path}")
            incompatible = self.model.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            critical_missing = [
                key for key in incompatible.missing_keys
                if key.startswith(("denoiser.", "ema_denoiser.", "diffusion_trainer.proj."))
            ]
            required_source_prefixes = (
                "denoiser.", "ema_denoiser.", "diffusion_trainer.proj."
            )
            absent_source = [
                prefix for prefix in required_source_prefixes
                if not any(key.startswith(prefix) for key in checkpoint["state_dict"])
            ]
            if critical_missing or incompatible.unexpected_keys or absent_source:
                raise RuntimeError(
                    "Unsafe stage warm start: "
                    f"critical_missing={critical_missing}, "
                    f"unexpected={incompatible.unexpected_keys}, "
                    f"absent_source={absent_source}"
                )
            self.model._warmstart_loaded_ema = True
            print(
                f"[deco-stage-init] loaded model+EMA+trainer weights from "
                f"{init_ckpt_path}; optimizer/global_step intentionally reset",
                flush=True,
            )

if __name__ == "__main__":

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})
