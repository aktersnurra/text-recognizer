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
- [ ] Check how to pass arguments to unittest (remove B009 then)
- [x] Able to set verbosity level on the logger to terminal output
- [ ] Implement Callbacks for training
    - [ ] Implement early stopping
    - [ ] Implement wandb
    - [ ] Implement lr scheduler as a callback
- [ ] Continuing reimplementing labs
- [ ] New models and datasets
- [ ] Check that dataset exists, otherwise download it form the web. Do this in run_experiment.py.
- [ ] Create repr func for data loaders
- [ ] Be able to restart with lr scheduler
- [ ] Implement Bayesian hyperparameter search
- [ ] Try to fix shell cmd security issues S404, S602
