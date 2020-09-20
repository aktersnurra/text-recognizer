#!/bin/bash
experiments_filename=${1:-training/experiments/sample_experiment.yml}
poetry run prepare-experiments --experiments_filename $experiments_filename
