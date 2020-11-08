#!/bin/bash
command="python text_recognizer/datasets/emnist_lines_dataset.py --max_length 34 --min_overlap 0.0 --max_overlap 0.33 --num_train 100000 --num_test 10000"
echo $command
eval $command
