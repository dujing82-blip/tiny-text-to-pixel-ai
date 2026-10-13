# Monster Generator — teacher guide

Audience: high-school students who have seen a neural network, a loss function and gradient descent. No calculus is needed for the core lesson. Use a GPU runtime and a reliable internet connection. Test the Colab runtime before class and download a checkpoint after your rehearsal.

## A 60-minute lesson

| Minutes | Activity | Main question |
|---|---|---|
| 0–8 | Open GPU Colab; inspect data and captions | Which features describe a monster? |
| 8–12 | Forward noising; start training | How can a clean image become random noise? |
| 12–23 | Explain noise supervision and embeddings while training runs | Where does the correct answer for the noise come from? |
| 23–35 | Explain U-Net, loss, EMA and reverse denoising | How can many small corrections create an image? |
| 35–44 | Generate and animate; compare seeds | What does randomness control? |
| 44–52 | Change one word with the same seed | Did the requested feature actually change? |
| 52–58 | Test unseen combinations and record attributes | Can the model combine familiar ideas in a new way? |
| 58–60 | Save results; exit question | What did the model learn, and what is it unable to do? |

If training takes longer, use the live loss and preliminary samples as a discussion. Continue the saved run during the next session. Do not present a procedural reference drawing as the model's generated result.

## A concrete explanation of the equation

For one pixel, suppose the clean value is 0.8 and the sampled noise is −0.4. If alpha_bar = 0.64, then the signal multiplier is 0.8 and the noise multiplier is 0.6. The noisy pixel is `0.8 × 0.8 + 0.6 × (−0.4) = 0.40`.

Alpha_bar describes how much signal variance remains after repeated noising. Its square root scales the original image. It is a schedule value, not a learned parameter or the learning rate. Training pixels are scaled to −1 to 1, so negative numbers are expected.

During training, the network receives 0.40, the noise level and the caption; its desired noise answer is −0.4. This example is only one pixel: the real model predicts all 32×32×3 noise values jointly using spatial context. It does not receive the original image as an extra input.

## How to explain the stages

**Forward process:** We deliberately spoil clean monster images with noise. Because we sampled the noise, we know the answer the network should learn to predict.

**Training:** Each batch uses randomly selected images, noise levels and new noise. The U-Net learns one noise predictor shared across all levels. We do not train 256 separate networks.

**Prompt:** Each supported word selects a small vector of trainable numbers. These embeddings tell the network which monster attributes should be present. They are learned with the U-Net; they do not come from a large language model.

**Reverse process:** Start from fresh random noise. Predict noise, estimate a clean image, and move to a lower noise level. Repeat. The model creates images by using what it learned, not by calling the drawing script or copying a selected dataset image.

**DDPM versus DDIM:** The forward process and noise-prediction objective are the DDPM setup. We use a deterministic DDIM inference rule to visit fewer noise levels. With eta=0, the initial seed supplies the randomness; the sampler does not add fresh noise at each step.

**Guidance:** During some training examples, hide the prompt. At inference, compare a prediction with the prompt to one without it. Increasing their difference can emphasize requested attributes. Large guidance can also harm the picture. At guidance 0, changing only the prompt should not change the image when the seed stays fixed.

## Student recording sheet

Use at least 10 seeds per selected prompt. Do not discard unsuccessful generations. Record “unclear” if a feature cannot be judged.

| Prompt / seen or unseen | Seed | Sampling steps | Guidance | Color | Eye count | Horns | Expression | Notes |
|---|---:|---:|---:|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

For each attribute, report `correct / total generated`, including unclear cases in the denominator. Also report the unclear count. Compare seen and unseen full combinations under the same sampling settings and training checkpoint. This is a classroom diagnostic, not a rigorous benchmark.

## Suggested experiments and expected reasoning

1. **Change the seed only.** Attributes should remain the same; shape and details may vary. A poorly trained model can still miss an attribute.
2. **Change the color only with fixed noise.** This isolates the effect of the color condition. Other details may also move because the model does not enforce perfect disentanglement.
3. **Guidance 0 versus 2.** At 0 the prompt is removed from the final prediction. At 2 the difference from the unconditional prediction is amplified.
4. **16 versus 64 sampling steps.** Fewer network evaluations are faster; visual quality can change. More steps do not compensate for an untrained network.
5. **Train another 1,000 steps.** Compare samples with fixed prompts, seeds and sampler settings. Do not compare two different seeds and attribute the difference to training.
6. **Unseen combinations.** The model has seen the components but not the complete tuple. Success suggests useful composition in this small task; it does not demonstrate general English understanding or unrestricted reasoning.

## Misconceptions to address

- A decreasing noise MSE does not guarantee accurate eye counts or expressions. Inspect outputs and count attributes.
- The U-Net output is predicted noise, not the final monster at every training step.
- Gaussian noise is not a semantic latent code with two interpretable coordinates.
- An embedding is trainable numerical input; “32 dimensions” does not mean 32 vocabulary words.
- The training loop samples a noise level directly. It does not run every previous forward step for every example.
- DDIM inference with 64 steps is not retraining the network for only 64 training levels.
- Training data, validation data, reference drawings and neural-network outputs must remain clearly labeled in classroom demonstrations.

## Rehearsal and practical settings

The notebook defaults to batch 128, 4,000 additional steps and a 24-minute training budget. The A100 30-minute figure is a target; use the measured rate printed by the notebook to plan your class. T4 can take longer. If GPU memory is limited, start fresh with batch 64 or 32. A resumed run retains its saved settings.

For continuity, use the same local checkpoint and RESUME=True. After a runtime reset, restore the downloaded checkpoint into the `runs/monster.pt` path printed by the notebook before enabling resume. Checkpoints include a dataset checksum to prevent accidentally resuming on incompatible data. GPU numerical results can differ across software and device versions even with the same seed.

This project uses only original synthetic monster artwork. The repository MIT license covers both code and data.
