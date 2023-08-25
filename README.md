# Text Recognizer
Implementing the text recognizer project from the course ["Full Stack Deep Learning Course"](https://fullstackdeeplearning.com/march2019) (FSDL) in PyTorch in order to learn best practices when building a deep learning project. I have expanded on this project by adding additional feature and ideas given by Claudio Jolowicz in ["Hypermodern Python"](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).

## Prerequisite

- [pyenv](https://github.com/pyenv/pyenv) (or similar) and python 3.9.\* installed.

- [nox](https://nox.thea.codes/en/stable/index.html) for linting, formatting, and testing.

- [Poetry](https://python-poetry.org/) is a project manager for python.

## Installation

Install poetry and pyenv.

```sh
pyenv local 3.9.*
make install
```

## Generate Datasets

Download and generate datasets by running:

```sh
make download
make generate
```

## Train

Use, modify, or create a new experiment found at `training/conf/experiment/`.
To run an experiment we first need to enter the virtual env by running:

```sh
poetry shell
```

Then we can train a new model by running:

```sh
python main.py +experiment=conv_transformer_paragraphs
```

## Network

Create a picture of the network and place it here

## Graveyard

Ideas of mine that did not work unfortunately:

* Efficientnet was apparently a terrible choice of an encoder
  - A ConvNext module heavily copied from lucidrains [x-unet](https://github.com/lucidrains/x-unet)
  was incredibly much better at encoding the images to a better representation.

* Use VQVAE to create pre-train a good latent representation
  - Tests with various compressions did not show any performance increase compared to training directly e2e, more like decrease to be honest
  - This is very unfortunate as I really hoped that this idea would work :(
  - I still really like this idea, and I might not have given up just yet...
  - I have now given up... :( ConvNext ftw

* Axial Transformer Encoder
  - Added a lot of extra parameters with no gain in performance
  - Cool idea, but on a single GPU

* Word Pieces
  - Might have worked better, but liked the idea of single character recognition more.

## Todo
- [ ] remove einops (try)
- [ ] Tests
- [ ] Evaluation
- [ ] Wandb artifact fetcher
- [ ] fix linting
- [x] Modularize the decoder
- [ ] Add kv cache
- [ ] Train with Laprop
- [x] Fix stems
- [x] residual attn
- [x] single kv head
- [x] fix rotary embedding
- [ ] simplify attention with norm
- [ ] tie embeddings
- [ ] cnn -> tf encoder -> tf decoder
