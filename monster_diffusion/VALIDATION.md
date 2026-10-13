# Validation record — 2026-10-13

## Checks completed

- Six automated unit tests passed: prompt parsing, reproducible artwork and held-out split exclusions, forward-noise equations, oracle reverse denoising, deterministic sampling, and training/checkpoint resume.
- Every Colab notebook code cell executed in CPU smoke mode with a small U-Net and two training steps, including plots, image export, and denoising GIF export.
- The optional widget panel was instantiated separately and its generation callback ran successfully. Browser rendering in the Colab frontend was not tested.
- The published notebook, dataset, metadata and all four downloaded Python modules were fetched from GitHub and matched the local SHA-256 checksums.

## Additional full-size-model training check

A separate CPU run used the actual 947,363-parameter model, batch 64, seed 42, 256 noise levels, and learning rate 0.0002. At 2,000 optimizer steps:

| Diagnostic | Value |
|---|---:|
| Recent training noise MSE (100-step average) | 0.03338 |
| Seen-combination validation noise MSE | 0.03485 |
| Held-out-combination noise MSE | 0.03669 |
| Average CPU training throughput | approximately 3.26 steps/second |

This check verifies that the full model learns and can generate samples and resume. Samples from this partial training run had recognizable colors and monster silhouettes, with noisy edges and imperfect eyes/expressions. These losses do not establish prompt adherence or image quality.

Environment: Python 3.12, PyTorch 2.14.1+cpu, NumPy 2.3.5.

The notebook's default 4,000-step, batch-128 GPU run was **not benchmarked on Colab A100 or T4**. The roughly 30-minute A100 figure is a classroom planning target. Use the notebook's measured rate and actual sample quality to decide whether to continue training.
