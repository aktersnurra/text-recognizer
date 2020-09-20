#!/bin/bash
experiments_filename=${1:-training/experiments/embedding_experiment.yml}
OUTPUT=$(./tasks/prepare_experiments.sh $experiments_filename)
echo $OUTPUT
eval $OUTPUT
