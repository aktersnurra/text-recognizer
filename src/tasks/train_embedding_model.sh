#!/bin/bash
experiments_filename=${1:-training/experiments/embedding_encoder.yml}
OUTPUT=$(./tasks/prepare_experiments.sh $experiments_filename)
eval $OUTPUT
