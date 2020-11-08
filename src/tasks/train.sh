#!/bin/bash

# Add checkpoint and resume experiment
usage() {
    cat << EOF
    usage: ./tasks/train_crnn_line_ctc_model.sh
    -f | --experiment_config                    Name of the experiment config.
    -c | --checkpoint           (Optional)      The experiment name to continue from.
    -p | --pretrained_weights   (Optional)      Path to pretrained weights.
    -n | --notrain              (Optional)      Evaluates a trained model.
    -t | --test                 (Optional)      If set, evaluates the model on test set.
    -v | --verbose              (Optional)      Sets the verbosity.
    -h | --help                                 Shows this message.
EOF
exit 1
}

experiment_config=""
checkpoint=""
pretrained_weights=""
notrain=""
test=""
verbose=""
train_command=""

while getopts 'f:c:p:nthv' flag; do
  case "${flag}" in
    f) experiment_config="${OPTARG}" ;;
    c) checkpoint="${OPTARG}" ;;
    p) pretrained_weights="${OPTARG}" ;;
    n) notrain="--notrain" ;;
    t) test="--test" ;;
    v) verbose="${verbose}v" ;;
    h) usage ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done


if [ -z ${experiment_config} ];
then
    echo "experiment_config not specified!"
    usage
    exit 1
fi

experiments_filename="training/experiments/${experiment_config}"
train_command=$(./tasks/prepare_experiments.sh $experiments_filename)

if [ ${checkpoint} ];
then
    train_command="${train_command} --checkpoint $checkpoint"
fi

if [ ${pretrained_weights} ];
then
    train_command="${train_command} --pretrained_weights $pretrained_weights"
fi

if [ ${verbose} ];
then
    train_command="${train_command} -$verbose"
fi

train_command="${train_command} $test $notrain"
echo $train_command
eval $train_command
