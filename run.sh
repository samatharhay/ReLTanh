#!/bin/bash

set -ex

# The following parameters are:
# - initialization parameters
# - model parameters
# - optimizer parameters
# - training parameters
# - use GPU
# - run on test set

python3 -u driver.py \
    \
    --split_seed 42 \
    --init_seed 42 \
    \
    --hidden_dims 128 \
    --hidden_layers 10 \
    --activation_function_type reltanh \
    --reltanh_alpha 0.0 \
    --reltanh_beta -1.5 \
    \
    --optimizer sgd \
    --learning_rate 1e-3 \
    --momentum 0.9 \
    \
    --batch_size 64 \
    --epochs 100 \
    \
    --cuda \
    \
    --run_test \
| tee train_log.txt
