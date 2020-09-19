#!/bin/bash
experiments_filename=${1:-training/experiments/line_ctc_experiment.yml}
exec ./prepare_experiments.sh experiments_filename=experiments_filename
