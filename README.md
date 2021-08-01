# Text Recognizer
Implementing the text recognizer project from the course ["Full Stack Deep Learning Course"](https://fullstackdeeplearning.com/march2019) (FSDL) in PyTorch in order to learn best practices when building a deep learning project. I have expanded on this project by adding additional feature and ideas given by Claudio Jolowicz in ["Hypermodern Python"](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).


## Setup

TBC


### Build word piece dataset

Extract text from the iam dataset:
```
python extract-iam-text --use_words --save_text train.txt --save_tokens letters.txt
```

Create word pieces from the extracted training text:
```
python make-wordpieces --output_prefix iamdb_1kwp --text_file train.txt --num_pieces 100
```

Optionally, build a transition graph for word pieces:
```
python build-transitions --tokens iamdb_1kwp_tokens_1000.txt --lexicon iamdb_1kwp_lex_1000.txt --blank optional --self_loops --save_path 1kwp_prune_0_10_optblank.bin --prune 0 10
```
(TODO: Not working atm, needed for GTN loss function)

## Todo
- [x] Efficient-net b0 + transformer decoder
- [ ] Load everything with hydra, get it to work
- [ ] Train network
- [ ] Tests
- [ ] Evaluation
- [ ] Wandb artifact fetcher
- [ ] attr refactor
- [ ] Refactor once more
- [ ] fix linting
- [ ] fix loading of transform iam paragraph


## Run Sweeps
 Run the following commands to execute hyperparameter search with W&B:

```
wandb sweep training/sweep_emnist_resnet.yml
export SWEEP_ID=...
wandb agent $SWEEP_ID

```
