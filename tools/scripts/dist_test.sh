#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch $@
