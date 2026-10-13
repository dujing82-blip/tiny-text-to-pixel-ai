"""Optional local CLI. The Colab notebook is the recommended student interface."""
import argparse
from pathlib import Path
import numpy as np

from dataset import parse_prompt, caption
from training import Trainer
from visuals import show_grid, save_gif


def main():
    parser = argparse.ArgumentParser(description='Train or sample the classroom monster diffusion model.')
    parser.add_argument('mode', choices=['train', 'sample'])
    parser.add_argument('--data', default=str(Path(__file__).parent/'data/monsters32.npz'))
    parser.add_argument('--checkpoint', default='runs/monster.pt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=4000, help='Additional optimization steps')
    parser.add_argument('--minutes', type=float, default=24)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--prompt', default='happy green monster with three eyes and horns')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-steps', type=int, default=64)
    parser.add_argument('--guidance', type=float, default=2)
    parser.add_argument('--output', default='runs')
    args = parser.parse_args()
    if args.mode == 'sample' or args.resume:
        trainer = Trainer.resume(args.checkpoint, args.data)
    else:
        trainer = Trainer(args.data, batch_size=args.batch_size, seed=args.seed)
    if args.mode == 'train':
        trainer.train(args.steps, args.minutes, args.checkpoint)
    attrs = parse_prompt(args.prompt)
    samples, frames = trainer.diffusion.sample(trainer.ema, np.repeat(attrs[None], 8, axis=0),
        steps=args.sample_steps, guidance=args.guidance, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    show_grid(samples, heading=caption(attrs), save_path=output/'monsters.png')
    save_gif(frames, output/'denoising.gif')


if __name__ == '__main__':
    main()
