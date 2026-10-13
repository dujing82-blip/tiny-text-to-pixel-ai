"""Execute all notebook cells using a small CPU model; not a quality benchmark."""
import json
import os
from pathlib import Path
import tempfile

project = Path(__file__).resolve().parents[1]
os.environ['MONSTER_SMOKE'] = '1'
os.environ['MONSTER_LOCAL_DIR'] = str(project)
os.environ['MPLBACKEND'] = 'Agg'
notebook = json.loads((project/'Monster_Generator_Colab.ipynb').read_text())
scope = {'__name__': '__main__'}
# The notebook writes only reproducible smoke outputs to runs/. Original data stays untouched.
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        print(f'Executing code cell {index}', flush=True)
        exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), scope)
print('All notebook code cells passed the CPU smoke check.')
