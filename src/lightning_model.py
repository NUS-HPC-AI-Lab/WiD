from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback


from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.conditioner.base import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: BaseAE,
                 conditioner: BaseConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 checkpoint_per_rank_rng: bool = True,
                 resume_rng_policy: str = "strict",
                 assert_rank_distinct_rng: bool = True,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.checkpoint_per_rank_rng = bool(checkpoint_per_rank_rng)
        self.resume_rng_policy = str(resume_rng_policy).lower()
        self.assert_rank_distinct_rng = bool(assert_rank_distinct_rng)
        if self.resume_rng_policy not in {"auto", "strict", "reseed", "off"}:
            raise ValueError(f"Invalid resume_rng_policy={resume_rng_policy!r}")
        self._pending_per_rank_rng = None
        self._pre_validation_rng = None
        self._warmstart_loaded_ema = False

        self._strict_loading = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        if not self._warmstart_loaded_ema:
            copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        # torch.compile
        self.denoiser.compile()
        self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
            {"params": params_sampler, "lr": 1e-3},
        ]
        # optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def on_validation_start(self) -> None:
        # Validation sampling must not advance training RNG.  The post-val
        # checkpoint callback calls restore_training_rng_after_validation()
        # before saving; on_validation_end is an idempotent fallback.
        trainer_fn = getattr(self.trainer.state.fn, "value", self.trainer.state.fn)
        if trainer_fn == "fit" and not self.trainer.sanity_checking:
            self._pre_validation_rng = self._capture_local_rng_state()
        self.ema_denoiser.to(torch.float32)

    def on_validation_end(self) -> None:
        self.restore_training_rng_after_validation()

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)
        self._restore_checkpoint_rng_if_available()
        self._assert_rank_distinct_training_rng()
        if self.trainer.is_global_zero:
            optimizer_report = []
            for optimizer in self.trainer.optimizers:
                optimizer_report.append(
                    [
                        {
                            "lr": float(group["lr"]),
                            "betas": tuple(group.get("betas", ())),
                            "weight_decay": float(group.get("weight_decay", 0.0)),
                        }
                        for group in optimizer.param_groups
                    ]
                )
            print(f"[deco-optimizer-contract] {optimizer_report}", flush=True)

    @staticmethod
    def _capture_local_rng_state() -> Dict[str, Any]:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        )
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "rank": int(rank),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state().cpu()
        return state

    @staticmethod
    def _restore_local_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"].cpu())

    def restore_training_rng_after_validation(self) -> None:
        if self._pre_validation_rng is None:
            return
        self._restore_local_rng_state(self._pre_validation_rng)
        self._pre_validation_rng = None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.checkpoint_per_rank_rng:
            return
        # Never serialize a validation-advanced stream.
        self.restore_training_rng_after_validation()
        local_state = self._capture_local_rng_state()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            states = [None] * world_size
            torch.distributed.all_gather_object(states, local_state)
        else:
            world_size, states = 1, [local_state]
        checkpoint["deco_per_rank_rng_v2"] = {
            "world_size": int(world_size),
            "global_step": int(getattr(self, "global_step", 0) or 0),
            "fixed_run_seed": int(os.environ.get("DECO_FIXED_RUN_SEED", "-1")),
            "segment_id": int(os.environ.get("DECO_SEGMENT_ID", "-1")),
            "states": states,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        payload = checkpoint.get("deco_per_rank_rng_v2")
        if payload is None:
            payload = checkpoint.get("deco_per_rank_rng_v1")
        if self.checkpoint_per_rank_rng and self.resume_rng_policy == "strict" and payload is None:
            raise RuntimeError("Strict resume requires per-rank RNG state in checkpoint")
        self._pending_per_rank_rng = payload

    def _restore_checkpoint_rng_if_available(self) -> None:
        payload = self._pending_per_rank_rng
        self._pending_per_rank_rng = None
        if payload is None or self.resume_rng_policy in {"off", "reseed"}:
            return
        saved_world_size = int(payload.get("world_size", -1))
        current_world_size = int(getattr(self.trainer, "world_size", 1) or 1)
        rank = int(getattr(self, "global_rank", 0) or 0)
        saved_fixed_seed = int(
            payload.get("fixed_run_seed", payload.get("base_seed", -1))
        )
        current_fixed_seed = int(os.environ.get("DECO_FIXED_RUN_SEED", "-1"))
        saved_segment = int(payload.get("segment_id", -1))
        current_segment = int(os.environ.get("DECO_SEGMENT_ID", "-1"))
        if saved_fixed_seed not in {-1, current_fixed_seed}:
            raise RuntimeError(
                f"fixed_run_seed mismatch checkpoint={saved_fixed_seed}, "
                f"current={current_fixed_seed}"
            )
        if saved_segment != current_segment:
            message = (
                f"planned RNG segment transition checkpoint={saved_segment}, "
                f"current={current_segment}"
            )
            if self.resume_rng_policy == "strict":
                raise RuntimeError(message)
            print(f"[deco-rng-resume] {message}; using deterministic segment reseed", flush=True)
            return
        if saved_world_size != current_world_size:
            message = (
                f"per-rank RNG world-size mismatch checkpoint={saved_world_size}, "
                f"current={current_world_size}"
            )
            if self.resume_rng_policy == "strict":
                raise RuntimeError(message)
            print(f"[deco-rng-resume] {message}; keeping launch reseed", flush=True)
            return
        states = payload.get("states", [])
        if len(states) != saved_world_size or rank >= len(states):
            raise RuntimeError("Malformed per-rank RNG payload")
        self._restore_local_rng_state(states[rank])
        print(
            f"[deco-rng-resume] restored rank={rank}/{current_world_size} "
            f"saved_step={payload.get('global_step')} segment={payload.get('segment_id')}",
            flush=True,
        )

    def _assert_rank_distinct_training_rng(self) -> None:
        if not self.assert_rank_distinct_rng:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            if int(getattr(self.trainer, "world_size", 1) or 1) > 1:
                raise RuntimeError("Rank RNG audit requires initialized DDP")
            return
        world_size = torch.distributed.get_world_size()
        device = self.device
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device)
        try:
            cpu_probe = torch.randint(0, 2**31 - 1, (16,), dtype=torch.int64)
            cuda_probe = torch.randint(
                0, 2**31 - 1, (16,), dtype=torch.int64, device=device
            )
        finally:
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state, device)
        probe = torch.cat([cpu_probe.to(device), cuda_probe])
        gathered = [torch.empty_like(probe) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, probe)
        if int(torch.unique(torch.stack(gathered), dim=0).shape[0]) != world_size:
            raise RuntimeError("Cross-rank RNG streams are not distinct")
        parameters = [p for p in self.denoiser.parameters() if p.numel()]
        sample = torch.cat([p.detach().reshape(-1)[:8].float() for p in parameters[:16]])
        gathered_params = [torch.empty_like(sample) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_params, sample)
        if not all(torch.equal(gathered_params[0], p) for p in gathered_params[1:]):
            raise RuntimeError("Model initialization differs across ranks")
        print(
            f"[deco-rng-audit] PASS unique={world_size}/{world_size} "
            "model_parameters_identical=true",
            flush=True,
        )


    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        with torch.no_grad():
            x = self.vae.encode(x)
            condition, uncondition = self.conditioner(y, metadata)
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        nonfinite = {
            key: value.detach()
            for key, value in loss.items()
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        }
        if nonfinite:
            raise FloatingPointError(
                f"non-finite training objective at global_step={self.global_step}: "
                f"keys={list(nonfinite)}"
            )
        # to be do! fix the bug in tqdm iteration when enabling accumulate_grad_batches>1
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        # sample images
        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        samples = self.vae.decode(samples)
        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination
