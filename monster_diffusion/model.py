"""Small conditional U-Net and transparent DDPM / DDIM equations."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def time_embedding(t, dim=128):
    frequencies = torch.exp(-math.log(10000) *
                            torch.arange(dim//2, device=t.device)/(dim//2-1))
    angles = t.float()[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, context=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.condition = nn.Linear(context, outgoing*2)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, context):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(context)).chunk(2, dim=1)
        h = self.norm2(h)*(1+scale[:, :, None, None])+shift[:, :, None, None]
        return self.skip(x)+self.conv2(F.silu(h))


class MonsterUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.base = base
        # Each attribute has its own learnable 32-number word embedding.
        self.embeddings = nn.ModuleList([nn.Embedding(n, 32) for n in (4, 3, 2, 2)])
        self.text_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv2d(3, base, 3, padding=1)
        self.enc1 = ResidualBlock(base, base)
        self.down1 = nn.Conv2d(base, base*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base*2, base*2)
        self.down2 = nn.Conv2d(base*2, base*3, 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(base*3, base*3)
        self.middle2 = ResidualBlock(base*3, base*3)
        self.up2 = nn.Conv2d(base*3, base*2, 3, padding=1)
        self.dec2 = ResidualBlock(base*4, base*2)
        self.up1 = nn.Conv2d(base*2, base, 3, padding=1)
        self.dec1 = ResidualBlock(base*2, base)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                    nn.Conv2d(base, 3, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy_image, timestep, labels, drop_mask=None):
        text = self.text_mlp(torch.cat([e(labels[:, i]) for i, e in
                                        enumerate(self.embeddings)], dim=1))
        # Zero the entire text condition for classifier-free guidance training.
        if drop_mask is not None:
            text = text*(~drop_mask[:, None]).to(text.dtype)
        context = self.time_mlp(time_embedding(timestep))+text
        a = self.enc1(self.input(noisy_image), context)       # 32 x 32
        b = self.enc2(self.down1(a), context)                # 16 x 16
        h = self.middle2(self.middle1(self.down2(b), context), context)  # 8 x 8
        h = self.up2(F.interpolate(h, scale_factor=2, mode='nearest'))
        h = self.dec2(torch.cat([h, b], dim=1), context)
        h = self.up1(F.interpolate(h, scale_factor=2, mode='nearest'))
        return self.output(self.dec1(torch.cat([h, a], dim=1), context))


class Diffusion:
    def __init__(self, timesteps=256, device='cpu'):
        self.timesteps = timesteps
        # Cosine schedule: alpha_bar approximately goes from 1 to 0.
        s = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float64)
        curve = torch.cos(((s/timesteps+0.008)/1.008)*math.pi/2).square()
        curve = curve/curve[0]
        self.betas = (1-curve[1:]/curve[:-1]).clamp(0, 0.999).float().to(device)
        self.alpha_bar = torch.cumprod(1-self.betas, dim=0)

    def add_noise(self, clean, timestep, noise):
        a = self.alpha_bar[timestep][:, None, None, None]
        return a.sqrt()*clean+(1-a).sqrt()*noise

    @torch.no_grad()
    def sample(self, model, labels, steps=64, guidance=2.0, seed=42, keep_frames=9):
        """Deterministic DDIM (eta=0), starting from seeded Gaussian noise.

        Training predicts epsilon. Inference visits a subset of the same timesteps.
        Frame labels use mathematical t=1..T; clean output is t=0.
        """
        if not 2 <= steps <= self.timesteps:
            raise ValueError(f'Sampling steps must be in [2, {self.timesteps}].')
        device = next(model.parameters()).device
        if not isinstance(labels, torch.Tensor):
            labels = np.asarray(labels)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if labels.ndim != 2 or labels.shape[1] != 4 or not len(labels):
            raise ValueError('labels must have shape (N, 4), N >= 1')
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((len(labels), 3, 32, 32), generator=gen, device=device)
        times = torch.linspace(self.timesteps-1, 0, steps).round().long().tolist()
        snapshots = [(self.timesteps, x[:1].cpu().clone())]
        capture = set(torch.linspace(0, steps-1, max(1, keep_frames-1)).long().tolist())
        was_training = model.training
        model.eval()
        try:
            for i, t in enumerate(times):
                ts = torch.full((len(x),), t, device=device, dtype=torch.long)
                if guidance == 1:
                    epsilon = model(x, ts, labels)
                else:
                    doubled = torch.cat([x, x])
                    drop = torch.cat([torch.ones(len(x), device=device, dtype=torch.bool),
                                      torch.zeros(len(x), device=device, dtype=torch.bool)])
                    uncond, cond = model(doubled, torch.cat([ts, ts]),
                                        torch.cat([labels, labels]), drop).chunk(2)
                    epsilon = uncond+guidance*(cond-uncond)
                a = self.alpha_bar[t]
                # Estimate the original clean image, then move to a lower noise level.
                x0 = ((x-(1-a).sqrt()*epsilon)/a.sqrt()).clamp(-1, 1)
                prev = times[i+1] if i+1 < len(times) else -1
                ap = self.alpha_bar[prev] if prev >= 0 else x.new_tensor(1.)
                # Recompute epsilon after clipping x0 for a consistent DDIM step.
                epsilon = (x-a.sqrt()*x0)/(1-a).sqrt().clamp_min(1e-8)
                x = ap.sqrt()*x0+(1-ap).sqrt()*epsilon
                if i in capture:
                    snapshots.append((prev+1, x[:1].cpu().clone()))
        finally:
            model.train(was_training)
        return x.clamp(-1, 1).cpu(), snapshots
