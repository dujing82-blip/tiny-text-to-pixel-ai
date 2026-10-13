"""Original, reproducible 32x32 monster art. No third-party images required."""
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw

COLORS = ('green', 'blue', 'red', 'purple')
EYES = ('one', 'two', 'three')
HORNS = ('no horns', 'horns')
MOODS = ('happy', 'sad')
PALETTE = ((91, 194, 99), (69, 157, 231), (235, 101, 103), (171, 117, 220))
# Every individual attribute occurs in training; only these full tuples are new.
HELD_OUT = ((0, 2, 1, 0), (1, 0, 0, 1), (2, 1, 1, 1), (3, 2, 0, 0))
VOCAB = (COLORS, EYES, HORNS, MOODS)


def caption(attrs):
    color, eyes, horns, mood = map(int, attrs)
    return (f'{MOODS[mood]} {COLORS[color]} monster with {EYES[eyes]} '
            f'{"eye" if eyes == 0 else "eyes"} and {HORNS[horns]}')


def parse_prompt(prompt):
    """Strict controlled vocabulary. Never silently ignore an unsupported concept."""
    text = re.sub(r'[^a-z0-9\s-]', ' ', prompt.lower()).replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    for digit, word in (('1', 'one'), ('2', 'two'), ('3', 'three')):
        text = re.sub(rf'\b{digit}\b', word, text)
    words = set(text.split())
    allowed = set('a an the monster with and eye eyes horn horns no without '
                  'one two three happy sad smiling frowning green blue red purple'.split())
    unknown = words - allowed
    if unknown:
        raise ValueError('Unsupported words: ' + ', '.join(sorted(unknown)) +
                         '. Use green/blue/red/purple, one/two/three eyes, '
                         'horns/no horns, happy/sad.')
    color = [i for i, name in enumerate(COLORS) if name in words]
    eyes = [i for i, name in enumerate(EYES) if name in words]
    if len(color) != 1 or len(eyes) != 1 or not words.intersection({'eye', 'eyes'}):
        raise ValueError('Specify one color and one eye count, e.g. happy green '
                         'monster with three eyes and horns.')
    happy = bool(words.intersection({'happy', 'smiling'}))
    sad = bool(words.intersection({'sad', 'frowning'}))
    if happy and sad:
        raise ValueError('Choose happy OR sad.')
    if words.intersection({'no', 'without'}) and not words.intersection({'horn', 'horns'}):
        raise ValueError('no/without must describe horns.')
    horns = int(bool(words.intersection({'horn', 'horns'})) and
                not bool(words.intersection({'no', 'without'})))
    # Deliberately explicit defaults: happy; no horns if horns are omitted.
    return np.array([color[0], eyes[0], horns, int(sad)], dtype=np.int64)


def draw_monster(attrs, seed):
    """Draw at native resolution so small eyes and mouths remain distinct."""
    rng = np.random.default_rng(seed)
    color, eye_id, horns, mood = map(int, attrs)
    im = Image.new('RGB', (32, 32), (247, 247, 242))
    d = ImageDraw.Draw(im)
    cx = 16 + int(rng.integers(-1, 2))
    cy = 18 + int(rng.integers(-1, 2))
    rx = int(rng.integers(9, 12))
    ry = int(rng.integers(9, 11))
    body = tuple(np.clip(np.array(PALETTE[color]) + rng.integers(-10, 11, 3), 0, 255))
    ink = (40, 43, 52)
    # Short limbs and symmetrical horns keep the task learnable in a short lesson.
    d.ellipse((cx-rx-3, cy-1, cx-rx+2, cy+5), fill=body, outline=ink)
    d.ellipse((cx+rx-2, cy-1, cx+rx+3, cy+5), fill=body, outline=ink)
    for x in (cx-5, cx+3):
        d.rounded_rectangle((x, cy+ry-3, x+3, min(31, cy+ry+2)), radius=1,
                            fill=body, outline=ink)
    if horns:
        for sign in (-1, 1):
            x = cx+sign*6
            d.polygon([(x-3, cy-ry+3), (x+sign, cy-ry-5), (x+3, cy-ry+3)],
                      fill=(249, 206, 100), outline=ink)
    d.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=body, outline=ink, width=1)
    n = eye_id+1
    positions = {1: [cx], 2: [cx-4, cx+4], 3: [cx-6, cx, cx+6]}[n]
    ey = cy-3
    for ex in positions:
        d.ellipse((ex-2, ey-3, ex+2, ey+2), fill='white', outline=ink)
        d.rectangle((ex, ey-1, ex+1, ey+1), fill=ink)
    if mood == 0:
        d.arc((cx-4, cy+1, cx+4, cy+7), 0, 180, fill=ink, width=2)
    else:
        d.arc((cx-4, cy+3, cx+4, cy+9), 180, 360, fill=ink, width=2)
    return np.asarray(im, dtype=np.uint8)


def build_dataset(destination, seed=2026, train_per_combo=300,
                  val_per_combo=20, held_per_combo=100):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images, labels, splits = [], [], []
    for attrs in product(range(4), range(3), range(2), range(2)):
        groups = [(2, held_per_combo)] if attrs in HELD_OUT else [
            (0, train_per_combo), (1, val_per_combo)]
        for split, count in groups:
            for _ in range(count):
                images.append(draw_monster(attrs, int(rng.integers(2**32))))
                labels.append(attrs)
                splits.append(split)
    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits, dtype=np.uint8)
    order = rng.permutation(len(images))
    path = destination / 'monsters32.npz'
    np.savez_compressed(path, images=images[order], labels=labels[order], splits=splits[order])
    meta = {
        'seed': seed, 'shape': list(images.shape), 'dtype': 'uint8',
        'attributes': {name: list(values) for name, values in
                       zip(('color', 'eyes', 'horns', 'mood'), VOCAB)},
        'split_ids': {'train': 0, 'validation': 1, 'unseen_combination': 2},
        'counts': {name: int((splits == i).sum()) for name, i in
                   [('train', 0), ('validation', 1), ('unseen_combination', 2)]},
        'held_out': [caption(a) for a in HELD_OUT],
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'license': 'MIT; original procedurally generated teaching artwork',
    }
    (destination/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).parent/'data'))
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output), indent=2))
