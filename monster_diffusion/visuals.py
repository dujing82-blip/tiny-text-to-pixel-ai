"""Classroom displays: actual data, generated samples, and denoising frames."""
from pathlib import Path
import math
import textwrap
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def to_rgb(images):
    if hasattr(images, 'detach'):
        return ((images.detach().cpu().permute(0, 2, 3, 1).numpy()+1)*127.5).clip(0, 255).astype('uint8')
    return np.asarray(images, dtype=np.uint8)


def show_grid(images, titles=None, columns=4, heading=None, save_path=None):
    images = to_rgb(images)
    columns = min(columns, len(images))
    rows = math.ceil(len(images)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*2.8, rows*3), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < len(images):
            ax.imshow(images[i], interpolation='nearest')
            if titles is not None:
                title = '\n'.join(textwrap.fill(line, width=29) for line in titles[i].split('\n'))
                ax.set_title(title, fontsize=9)
    if heading:
        fig.suptitle(heading, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if heading else (0, 0, 1, 1))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def save_gif(frames, path='runs/denoising.gif', scale=8):
    rendered = []
    for timestep, tensor in frames:
        picture = Image.fromarray(to_rgb(tensor)[0]).resize((32*scale, 32*scale), Image.Resampling.NEAREST)
        canvas = Image.new('RGB', (picture.width, picture.height+28), 'white')
        canvas.paste(picture, (0, 28))
        ImageDraw.Draw(canvas).text((8, 7), f't = {timestep}', fill='black')
        rendered.append(canvas)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(path, save_all=True, append_images=rendered[1:], duration=350, loop=0)
    return str(path)
