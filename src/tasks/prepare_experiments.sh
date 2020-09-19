#!/bin/bash
experiments_filename=${1:-training/experiments/sample_experiment.yml}
python training/prepare_experiments.py --experiments_filename $experiments_filename
