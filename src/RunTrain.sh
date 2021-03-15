#!/usr/bin/env bash

python train.py --model="$1" --batch_size="$2" --eval_batch_size="$2" --n_epoch="$3"

