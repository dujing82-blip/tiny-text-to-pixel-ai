"""Noise-prediction training, EMA sampling, validation and resumable checkpoints."""
from copy import deepcopy
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F

from model import MonsterUNet, Diffusion


class Trainer:
    def __init__(self, data_path, device=None, base=32, batch_size=128,
                 learning_rate=2e-4, timesteps=256, seed=42):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.config = dict(base=base, batch_size=batch_size, learning_rate=learning_rate,
                           timesteps=timesteps, seed=seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.data_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        with np.load(data_path, allow_pickle=False) as data:
            self.images = torch.from_numpy(data['images'].copy()).permute(0, 3, 1, 2).contiguous()
            self.labels = torch.from_numpy(data['labels'].copy()).long()
            self.splits = torch.from_numpy(data['splits'].copy()).long()
        self.train_ids = torch.where(self.splits == 0)[0]
        if len(self.train_ids) == 0:
            raise ValueError('Dataset has no training images.')
        self.train_images = self.images[self.train_ids].to(self.device)
        self.train_labels = self.labels[self.train_ids].to(self.device)
        self.model = MonsterUNet(base).to(self.device)
        self.ema = deepcopy(self.model).eval().requires_grad_(False)
        self.diffusion = Diffusion(timesteps, self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.step = 0
        self.history = []

    def _noise(self, x, generator):
        t = torch.randint(self.diffusion.timesteps, (len(x),), device=self.device,
                          generator=generator)
        noise = torch.randn(x.shape, generator=generator, device=self.device)
        return self.diffusion.add_noise(x, t, noise), t, noise

    @torch.no_grad()
    def validate(self, split=1, count=128):
        ids = torch.where(self.splits == split)[0][:count]
        if not len(ids):
            raise ValueError('Requested validation split is empty.')
        generator = torch.Generator(device=self.device).manual_seed(12345)
        total = 0.
        for chunk in ids.split(32):
            x = self.images[chunk].to(self.device).float()/127.5-1
            labels = self.labels[chunk].to(self.device)
            xt, t, noise = self._noise(x, generator)
            pred = self.ema(xt, t, labels)
            total += F.mse_loss(pred, noise).item()*len(chunk)
        return total/len(ids)

    def train(self, additional_steps=4000, max_minutes=24, checkpoint='runs/monster.pt',
              log_every=100, callback=None):
        """max_minutes limits this call; additional_steps resumes from current step."""
        if additional_steps < 1 or max_minutes <= 0 or log_every < 1:
            raise ValueError('Steps, time budget and logging interval must be positive.')
        started = time.monotonic()
        start_step = self.step
        target = self.step+additional_steps
        self.model.train()
        loss_sum, loss_count = 0., 0
        print(f'{self.device}: {sum(p.numel() for p in self.model.parameters()):,} parameters; '
              f'training from step {self.step} toward {target}.', flush=True)
        try:
            while self.step < target:
                if time.monotonic()-started >= max_minutes*60:
                    print('Time budget reached; saving current training state.', flush=True)
                    break
                idx = torch.randint(len(self.train_images), (self.config['batch_size'],),
                                    device=self.device, generator=self.generator)
                x = self.train_images[idx].float()/127.5-1
                labels = self.train_labels[idx]
                xt, t, noise = self._noise(x, self.generator)
                drop = torch.rand(len(x), device=self.device, generator=self.generator) < 0.1
                lr = self.config['learning_rate']*min(1., (self.step+1)/100)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.use_amp):
                    prediction = self.model(xt, t, labels, drop)
                    loss = F.mse_loss(prediction.float(), noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss: checkpoint saved; try a smaller learning rate.')
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                with torch.no_grad():
                    decay = min(0.995, (1+self.step)/(10+self.step))
                    for average, current in zip(self.ema.parameters(), self.model.parameters()):
                        average.lerp_(current, 1-decay)
                loss_sum += loss.detach().item()
                loss_count += 1
                if self.step % log_every == 0 or self.step == target:
                    elapsed = time.monotonic()-started
                    rate = (self.step-start_step)/max(elapsed, 1e-6)
                    record = dict(step=self.step, train_noise_mse=loss_sum/loss_count,
                                  elapsed_seconds=elapsed, steps_per_second=rate)
                    if self.step % 500 == 0 or self.step == target:
                        record['validation_noise_mse'] = self.validate()
                    self.history.append(record)
                    print(f'Step {self.step:5d} | noise MSE {record["train_noise_mse"]:.4f} | '
                          f'{rate:.2f} steps/s | estimated {(target-self.step)/rate/60:.1f} min left',
                          flush=True)
                    loss_sum, loss_count = 0., 0
                    if callback is not None:
                        callback(self, record)
                if self.step % 500 == 0:
                    self.save(checkpoint)
        finally:
            self.save(checkpoint)
        print(f'Saved step {self.step} to {checkpoint}.', flush=True)
        return self.history

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(format_version=1, config=self.config, step=self.step,
                     model=self.model.state_dict(), ema=self.ema.state_dict(),
                     optimizer=self.optimizer.state_dict(), scaler=self.scaler.state_dict(),
                     history=self.history, data_sha256=self.data_sha256,
                     generator_state=self.generator.get_state(), device_type=self.device.type)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def resume(cls, checkpoint, data_path, device=None):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        trainer = cls(data_path, device=device, **state['config'])
        if trainer.data_sha256 != state['data_sha256']:
            raise ValueError('Checkpoint dataset checksum differs from the current dataset.')
        trainer.model.load_state_dict(state['model'])
        trainer.ema.load_state_dict(state['ema'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        if trainer.use_amp and state['scaler']:
            trainer.scaler.load_state_dict(state['scaler'])
        trainer.step, trainer.history = state['step'], state['history']
        if state['device_type'] == trainer.device.type:
            trainer.generator.set_state(state['generator_state'])
        else:
            print('Device changed; weights/optimizer restored, random stream restarted.')
        return trainer
