"""Small CPU tests for WiD mathematics, stage transitions and data handling."""

import io
import random
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from train import PRIOR
from prepare_sft import extract_archive, paired_index
from src.callbacks.grad import GradientMonitor
from src.callbacks.wid_stages import Pretrain512Transition, TrainingCheckpoint
from src.data.dataset.wid_sft import VerifiedImageTextSFT
from src.data.dataset.image_txt import ImageText
from src.lightning_model import LightningModel
from src.models.transformer.wid import SharedPatchWienerState, analytic_velocity_gain
from src.utils.spectral_precision import spectral_fp32_context


class TinyNet(torch.nn.Linear):
    def compile(self):
        pass


class WiDTests(unittest.TestCase):
    def test_stage_configs(self):
        for stage, gbs, lr, steps, segment in [
            ("pretrain_256", 1536, 2e-4, 200000, 0),
            ("pretrain_512", 512, 1e-4, 280000, 0),
            ("sft_512", 512, 1e-5, 40000, 1),
        ]:
            c = yaml.safe_load((ROOT / f"configs_t2i/wid/{stage}.yaml").read_text())
            t, m, d = c["trainer"], c["model"], c["data"]
            self.assertEqual(t["devices"] * t["accumulate_grad_batches"] * d["train_batch_size"], gbs)
            self.assertEqual(m["optimizer"]["init_args"]["lr"], lr)
            self.assertEqual(t["max_steps"], steps)
            self.assertEqual(c["segment_id"], segment)
            self.assertEqual(d["train_num_workers"], 8)
            self.assertEqual(m["denoiser"]["class_path"], "src.models.transformer.wid_t2i.DeCoWiD")

    def test_prior_and_wiener_formula(self):
        state = SharedPatchWienerState(str(PRIOR))
        x = torch.randn(2, 4, 768, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(state.decode(state.analyze(x), torch.float32), x,
                                   atol=2e-6, rtol=2e-6)
        lam = state.effective_eigenvalues()
        t = torch.tensor([0.2, 0.7])
        a = t[:, None, None]
        expected = (a * lam - (1 - a)) / (a.square() * lam + (1 - a).square())
        torch.testing.assert_close(analytic_velocity_gain(lam, t, 1e-12), expected, atol=0, rtol=0)

    def test_spectral_precision_restored_on_error(self):
        old = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with self.assertRaises(ValueError):
                with spectral_fp32_context(torch.ones(1)):
                    self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                    raise ValueError("test")
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            torch.set_float32_matmul_precision(old)

    def test_rng_roundtrip(self):
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        state = LightningModel._capture_local_rng_state()
        draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
        LightningModel._restore_local_rng_state(state)
        self.assertEqual(draws, (random.random(), float(np.random.rand()), float(torch.rand(()))))

    def test_stage2_lr_is_changed_after_restore(self):
        opt = SimpleNamespace(
            state={i: {"step": torch.tensor(200000)} for i in range(290)},
            param_groups=[{"params": list(range(n)), "lr": lr, "betas": (0.9, 0.95)}
                          for n, lr in [(284, 2e-4), (6, 2e-4), (0, 1e-3)]],
        )
        trainer = SimpleNamespace(world_size=8, accumulate_grad_batches=4,
                                  precision="bf16-mixed", global_step=200000, optimizers=[opt])
        batch = [torch.empty(16, 3, 512, 512, device="meta")]
        Pretrain512Transition().on_train_batch_start(trainer, None, batch, 0)
        self.assertEqual([g["lr"] for g in opt.param_groups], [1e-4, 1e-4, 1e-3])
        opt.state.pop(0)
        with self.assertRaises(RuntimeError):
            Pretrain512Transition().on_train_batch_start(trainer, None, batch, 0)

    def test_warmstarted_ema_is_not_overwritten(self):
        model = LightningModel(TinyNet(2, 2), TinyNet(2, 2), TinyNet(2, 2),
                               TinyNet(2, 2), TinyNet(2, 2))
        model._trainer = SimpleNamespace(strategy=SimpleNamespace(barrier=lambda: None))
        with torch.no_grad():
            model.ema_denoiser.weight.fill_(9)
        model._warmstart_loaded_ema = True
        model.configure_model()
        self.assertTrue(torch.equal(model.ema_denoiser.weight, torch.full((2, 2), 9.)))
        with self.assertRaises(RuntimeError):
            model.on_load_checkpoint({})

    def test_checkpoint_at_new_optimizer_step_only(self):
        callback = TrainingCheckpoint(5000)
        trainer = SimpleNamespace(global_step=200000)
        callback.on_train_start(trainer, None)
        callback.on_train_batch_end(trainer, None, None, None, 0)

    def test_checkpoint_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = TrainingCheckpoint(5)
            callback.dirpath = tmp
            trainer = SimpleNamespace(
                global_step=5, is_global_zero=True,
                strategy=SimpleNamespace(broadcast=lambda x: x, barrier=lambda: None),
                save_checkpoint=lambda path, weights_only: Path(path).write_bytes(b"full state"),
            )
            callback.on_train_batch_end(trainer, None, None, None, 0)
            self.assertEqual((Path(tmp) / "step=000000005-train.ckpt").read_bytes(), b"full state")
            self.assertFalse((Path(tmp) / "step=000000005-train.ckpt.partial").exists())
            callback._last_saved_step = -1
            with self.assertRaises(FileExistsError):
                callback.on_train_batch_end(trainer, None, None, None, 0)

    def test_gradient_monitor_does_not_modify_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3., 4.])
        values = []
        module = SimpleNamespace(device=torch.device("cpu"), log=lambda k, v, **kw: values.append(float(v)))
        trainer = SimpleNamespace(global_step=0, is_global_zero=True)
        GradientMonitor().on_before_optimizer_step(trainer, module,
            SimpleNamespace(param_groups=[{"params": [parameter]}]))
        self.assertEqual(values, [5.])
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([3., 4.])))

    def test_sft_extraction_and_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(root / "data.tar", "w") as tar:
                for name, data in {"1.jpg": b"original bytes", "1.txt": b"caption"}.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            extract_archive(root / "data.tar", root / "out")
            self.assertEqual(paired_index(root / "out"), [root / "out/1.jpg"])
            self.assertEqual((root / "out/1.jpg").read_bytes(), b"original bytes")
            with tarfile.open(root / "unsafe.tar", "w") as tar:
                tar.addfile(tarfile.TarInfo("../escape"))
            with self.assertRaises(ValueError):
                extract_archive(root / "unsafe.tar", root / "unsafe")

    def test_sft_sorted_index_and_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("2", "10"):
                Image.new("RGB", (540, 600), (80, 120, 30)).save(root / f"{name}.jpg")
                (root / f"{name}.txt").write_text(name + " caption", encoding="utf-8")
            old = ImageText(str(root), 512)
            new = VerifiedImageTextSFT(str(root), expected_pairs=2)
            self.assertEqual([Path(p).name for p in new.image_paths], ["10.jpg", "2.jpg"])
            for i, path in enumerate(new.image_paths):
                x, y, _ = new[i]
                a, b, _ = old[old.image_paths.index(path)]
                torch.testing.assert_close(x, a, atol=0, rtol=0)
                self.assertEqual(y, b)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
