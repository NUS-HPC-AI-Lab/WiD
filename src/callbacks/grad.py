"""Log gradient norms at optimizer boundaries, before gradient clipping."""

import lightning.pytorch as pl
import torch


class GradientMonitor(pl.Callback):
    """A read-only monitor; it never changes or clears gradients."""

    def __init__(self, norm_type: int = 2, every_n_optimizer_steps: int = 50):
        super().__init__()
        if float(norm_type) != 2.0 or every_n_optimizer_steps <= 0:
            raise ValueError("Use L2 norms and a positive logging interval")
        self.every_n_optimizer_steps = int(every_n_optimizer_steps)
        self._last_logged_step = -1

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        step = int(trainer.global_step)
        if (not trainer.is_global_zero or step == self._last_logged_step
                or step % self.every_n_optimizer_steps):
            return
        self._last_logged_step = step
        total_sq = torch.zeros((), device=pl_module.device, dtype=torch.float64)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    norm = torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
                    total_sq.add_(norm.double().square())
        pl_module.log("train/grad_norm/preclip_total", total_sq.sqrt().float(),
                      on_step=True, on_epoch=False, prog_bar=False,
                      sync_dist=False, rank_zero_only=True)
