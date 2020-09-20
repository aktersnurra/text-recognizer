#!/bin/bash
experiments_filename=${1:-training/experiments/line_ctc_experiment.yml}
OUTPUT=$(./tasks/prepare_experiments.sh $experiments_filename)
echo $OUTPUT
eval $OUTPUT
