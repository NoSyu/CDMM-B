#!/usr/bin/env bash

python test.py --model="$1" --batch_size="$2" --eval_batch_size="$2" --checkpoint="$3"
