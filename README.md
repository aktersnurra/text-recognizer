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
- [x] Implement Callbacks for training
    - [x] Implement early stopping
    - [x] Implement wandb
    - [x] Implement lr scheduler as a callback
    - [x] Implement save checkpoint callback
    - [x] Implement TQDM progress bar (Low priority)
- [ ] Check that dataset exists, otherwise download it form the web. Do this in run_experiment.py.
- [x] Create repr func for data loaders
- [ ] Be able to restart with lr scheduler (May skip this)
- [ ] Implement population based training
- [x] Implement Bayesian hyperparameter search (with W&B maybe)
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
- [x] Implement ctc line model
    - [x] Implement CNN encoder (ResNet style)
    - [x] Implement the RNN + output layer
    - [x] Construct/implement the CTC loss
- [x] Sweep base config yaml file
- [x] sweep.py
- [x] sweep.yaml
- [x] Fix dataset splits.
- [x] Implement predict on image
- [x] CTC decoder
- [x] IAM dataset
- [x] IAM Lines dataset
- [x] IAM paragraphs dataset
- [ ] CNN + Transformer
- [ ] fix nosec problem
- [x] common Dataset class
- [x] Fix CTC blank stuff and varying length

## Run Sweeps
 Run the following commands to execute hyperparameter search with W&B:

```
wandb sweep training/sweep_emnist_resnet.yml
export SWEEP_ID=...
wandb agent $SWEEP_ID

```

## PyTorch Performance Guide
Tips and tricks from ["PyTorch Performance Tuning Guide - Szymon Migacz, NVIDIA"](https://www.youtube.com/watch?v=9mS1fIYj1So&t=125s):

* Always better to use `num_workers > 0`, allows asynchronous data processing
* Use `pin_memory=True` to allow data loading and computations to happen on the GPU in parallel.
* Have to tune `num_workers` to use based on the problem, too many and data loading becomes slower.
* For CNNs use `torch.backends.cudnn.benchmark=True`, allows cuDNN to select the best algorithm for convolutional computations (autotuner).
* Increase batch size to max out GPU memory.
* Use optimizer for large batch training, e.g. LARS, LAMB etc.
* Set `bias=False` for convolutions directly followed by BatchNorm.
* Use `for p in model.parameters(): p.grad = None` instead of `model.zero_grad()`.
* Careful with disable debug APIs in prod (detect_anomaly, profiler, gradcheck).
* Use `DistributedDataParallel` not `DataParallel`, uses 1 CPU core for each GPU.
* Important to load balance compute on all GPUs, if variably-sized inputs or GPUs will idle.
* Use an apex fused optimizer
* Use checkpointing to recompute memory-intensive compute-efficient ops in backward pass (e.g. activations, upsampling), `torch.utils.checkpoint`.
* Use `@torch.jit.script`, especially to fuse long sequences of pointwise operations like GELU.
