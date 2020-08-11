# Text Recognizer
Implementing the text recognizer project from the course ["Full Stack Deep Learning Course"](https://fullstackdeeplearning.com/march2019) in PyTorch in order to learn best practices when building a deep learning project. I have expanded on this project by adding additional feature and ideas given by Claudio Jolowicz in ["Hypermodern Python"](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).


## Setup

TBC

## Todo
- [x] subsampling
- [x] Be able to run experiments
- [x] Train models
- [x] Fix input size in base model
- [x] Fix s.t. the best weights are saved
- [x] Implement total training time
- [x] Fix tqdm and logging output
- [x] Fix basic test to load model
- [x] Fix loading previous experiments
- [x] Able to set verbosity level on the logger to terminal output
- [ ] Implement Callbacks for training
    - [x] Implement early stopping
    - [x] Implement wandb
    - [x] Implement lr scheduler as a callback
    - [x] Implement save checkpoint callback
    - [ ] Implement TQDM progress bar (Low priority)
- [ ] Check that dataset exists, otherwise download it form the web. Do this in run_experiment.py.
- [x] Create repr func for data loaders
- [ ] Be able to restart with lr scheduler (May skip this BS)
- [ ] Implement population based training
- [ ] Implement Bayesian hyperparameter search (with W&B maybe)
- [x] Try to fix shell cmd security issues S404, S602
- [x] Change prepare_experiment.py to print statements st it can be run with tasks/prepare_sample_experiments.sh | parallel -j1
- [x] Fix caption in WandbImageLogger
- [x] Rename val_accuracy in metric
- [x] Start implementing callback list stuff in train.py
- [x] Fix s.t. callbacks can be loaded in run_experiment.py
- [x] Lift out Emnist dataset out of Emnist dataloaders
- [x] Finish Emnist line dataset
- [x] SentenceGenerator
- [x] Write a Emnist line data loader
- [ ] Implement ctc line model
    - [ ] Implement CNN encoder (ResNet style)
    - [ ] Implement the RNN + output layer
    - [ ] Construct/implement the CTC loss
- [ ] Sweep base config yaml file
- [ ] sweep.py
- [ ] sweep.yaml
