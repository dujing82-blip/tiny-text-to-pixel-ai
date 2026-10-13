"""Run: python -m unittest discover -s tests -v (from monster_diffusion)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import HELD_OUT, build_dataset, caption, draw_monster, parse_prompt
from model import Diffusion, MonsterUNet
from training import Trainer

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_prompts(self):
        for attrs in np.ndindex(4, 3, 2, 2):
            np.testing.assert_array_equal(parse_prompt(caption(attrs)), attrs)
        np.testing.assert_array_equal(parse_prompt('green monster with 3 eyes and horns'), [0, 2, 1, 0])
        np.testing.assert_array_equal(parse_prompt('sad blue monster with one eye without horns'), [1, 0, 0, 1])
        for bad in ('orange monster with two eyes', 'green blue monster with one eye',
                    'happy sad green monster with one eye', 'green flying monster with two eyes'):
            with self.assertRaises(ValueError):
                parse_prompt(bad)

    def test_render_and_splits(self):
        a = draw_monster([0, 2, 1, 0], 42)
        self.assertEqual(a.shape, (32, 32, 3))
        self.assertEqual(a.dtype, np.uint8)
        np.testing.assert_array_equal(a, draw_monster([0, 2, 1, 0], 42))
        with tempfile.TemporaryDirectory() as tmp:
            meta = build_dataset(tmp, train_per_combo=3, val_per_combo=1, held_per_combo=2)
            with np.load(Path(tmp)/'monsters32.npz') as archive:
                data = {key: archive[key] for key in archive.files}
            for split in (0, 1):
                for attrs in data['labels'][data['splits'] == split]:
                    self.assertNotIn(tuple(attrs), HELD_OUT)
            self.assertEqual({tuple(a) for a in data['labels'][data['splits'] == 2]}, set(HELD_OUT))
            self.assertEqual(meta['counts'], dict(train=132, validation=44, unseen_combination=8))
            self.assertEqual(meta, json.loads((Path(tmp)/'metadata.json').read_text()))


class DiffusionTests(unittest.TestCase):
    def test_noise_math(self):
        diffusion = Diffusion()
        self.assertTrue(torch.all(diffusion.alpha_bar[1:] < diffusion.alpha_bar[:-1]))
        self.assertLess(diffusion.alpha_bar[-1], 1e-6)
        clean = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(clean)
        t = torch.tensor([0, 150])
        xt = diffusion.add_noise(clean, t, noise)
        a = diffusion.alpha_bar[t][:, None, None, None]
        torch.testing.assert_close((xt-(1-a).sqrt()*noise)/a.sqrt(), clean)

    def test_sampling_and_seed(self):
        model = MonsterUNet(base=8)
        labels = torch.tensor([[0, 2, 1, 0]])
        output = model(torch.randn(1, 3, 32, 32), torch.tensor([10]), labels)
        self.assertEqual(output.shape, (1, 3, 32, 32))
        diffusion = Diffusion(16)
        first, frames = diffusion.sample(model, labels, steps=4, guidance=2, seed=9)
        second, _ = diffusion.sample(model, labels, steps=4, guidance=2, seed=9)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(first.abs().max().item(), 1.)
        self.assertEqual((frames[0][0], frames[-1][0]), (16, 0))
        self.assertTrue(model.training)

    def test_oracle_denoising(self):
        diffusion = Diffusion(32)
        clean = torch.full((1, 3, 32, 32), 0.2)
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            def forward(self, x, t, labels):
                a = diffusion.alpha_bar[t][:, None, None, None]
                return (x-a.sqrt()*clean)/(1-a).sqrt()
        sample, _ = diffusion.sample(Oracle(), [[0, 0, 0, 0]], steps=8, guidance=1)
        torch.testing.assert_close(sample, clean, atol=2e-4, rtol=2e-4)

    def test_train_resume_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, train_per_combo=2, val_per_combo=1, held_per_combo=1)
            path = Path(tmp)/'monsters32.npz'
            checkpoint = Path(tmp)/'checkpoint.pt'
            trainer = Trainer(path, device='cpu', base=8, batch_size=4, timesteps=16)
            self.assertTrue(np.isfinite(trainer.validate(count=8)))
            trainer.train(3, max_minutes=1, checkpoint=checkpoint, log_every=3)
            restored = Trainer.resume(checkpoint, path, device='cpu')
            self.assertEqual(restored.step, 3)
            for a, b in zip(trainer.ema.parameters(), restored.ema.parameters()):
                torch.testing.assert_close(a, b)
            trainer.train(1, 1, checkpoint, 1)
            restored.train(1, 1, checkpoint, 1)
            for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
