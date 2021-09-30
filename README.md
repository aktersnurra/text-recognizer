# Text Recognizer
Implementing the text recognizer project from the course ["Full Stack Deep Learning Course"](https://fullstackdeeplearning.com/march2019) (FSDL) in PyTorch in order to learn best practices when building a deep learning project. I have expanded on this project by adding additional feature and ideas given by Claudio Jolowicz in ["Hypermodern Python"](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).

## Installation

Install poetry and pyenv.

```sh
pyenv local 3.9.1
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
- [ ] Local attention for target sequence
- [ ] Rotary embedding for target sequence
- [ ] Tests
- [ ] Evaluation
- [ ] Wandb artifact fetcher
- [ ] Refactor once more
- [ ] fix linting
- [ ] transducer loss function
- [ ] make for install, build datasets
- [ ] Fix artifact uploading to wandb
- [ ] Train lines with word pieces
- [ ] Train with Smoothloss
- [ ] Train with SWA
- [ ] VqTransformer without the quantization
- [ ] VqTransformer with extra layer


## Run Sweeps (old stuff)
 Run the following commands to execute hyperparameter search with W&B:

```
wandb sweep training/sweep_emnist_resnet.yml
export SWEEP_ID=...
wandb agent $SWEEP_ID

```

(TODO: Not working atm, needed for GTN loss function)
Optionally, build a transition graph for word pieces:
```
python build-transitions --tokens iamdb_1kwp_tokens_1000.txt --lexicon iamdb_1kwp_lex_1000.txt --blank optional --self_loops --save_path 1kwp_prune_0_10_optblank.bin --prune 0 10
```
