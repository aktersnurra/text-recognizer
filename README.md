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
make check
make install
```

## Generate Datasets

Download and generate datasets by running:

```sh
make download
make generate
```


## TODO

## Todo
- [ ] remove einops
- [ ] Tests
- [ ] Evaluation
- [ ] Wandb artifact fetcher
- [ ] fix linting
