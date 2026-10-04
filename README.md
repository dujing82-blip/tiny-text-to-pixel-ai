# Tiny Text-to-Pixel AI

A tiny, from-scratch **text-conditioned generative AI** project for teaching embeddings, latent representations, and image generation.

Students train a Conditional Variational Autoencoder (CVAE) in Google Colab to generate **16×16 pixel art** from prompts such as `red small robot`, `blue big spaceship`, and `green big heart`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/tiny-text-to-pixel-ai/blob/main/Tiny_Text_to_Pixel_AI_Colab.ipynb)

![Dataset preview](dataset_preview.png)

## The idea

The project keeps the full generative-AI story visible:

**text concepts → embeddings → latent representation → generator → image**

The dataset uses three composable attributes:

- **Color:** red, blue, green, yellow
- **Size:** small, big
- **Object:** heart, robot, spaceship, tree, house

That gives 40 possible semantic combinations.

## The generalization experiment

Five combinations are deliberately excluded from training:

- `blue big spaceship`
- `yellow small robot`
- `green big heart`
- `red small tree`
- `blue small house`

The model sees every individual concept, but never these exact combinations. Students can therefore ask:

> **Can AI create something it has never seen before?**

## Dataset

The included dataset contains **3,050 procedurally generated 16×16 RGB pixel-art images**:

- 2,800 training images
- 250 images from the five held-out combinations
- a CSV file containing filename, caption, color, size, object, and split

## Run the lesson

Click **Open in Colab** above and run the notebook from top to bottom. The notebook downloads the dataset directly from this GitHub repository, so students do not need to upload any files manually.

## What students learn

1. Computers need numerical representations of words.
2. Embeddings represent semantic concepts as learned vectors.
3. A latent variable can encode variation not specified by the prompt.
4. Conditioning tells a generative model *what* to generate.
5. Sampling different latent vectors can produce different outputs for the same prompt.
6. A model can be tested on combinations deliberately withheld from training.

## Important limitation

This is intentionally **not a full natural-language model**. The notebook parses a controlled vocabulary of color, size, and object words and learns embeddings for them.

That simplification makes the mechanism visible. It also creates a natural next question:

> How do modern systems understand a prompt such as “a large blue spaceship flying through space”?

That leads naturally to tokenization, text encoders, Transformers, and modern text-to-image generation.

## Model

During training:

```text
IMAGE ─────────────────┐
                       │
COLOR  ── embedding ───┤
SIZE   ── embedding ───┼──> ENCODER ──> latent z
OBJECT ── embedding ───┘
```

During generation:

```text
random latent z ───────┐
                       │
COLOR  ── embedding ───┤
SIZE   ── embedding ───┼──> DECODER / GENERATOR ──> 16×16 IMAGE
OBJECT ── embedding ───┘
```

## Files

```text
tiny-text-to-pixel-ai/
├── README.md
├── LICENSE
├── .gitignore
├── Tiny_Text_to_Pixel_AI_Colab.ipynb
├── tiny_pixel_dataset.zip
└── dataset_preview.png
```

## Requirements

The notebook is designed for Google Colab and uses TensorFlow/Keras, NumPy, pandas, Matplotlib, Pillow, and ipywidgets. No pretrained generative model is required.

## License

Code and original procedurally generated teaching data are released under the MIT License.
