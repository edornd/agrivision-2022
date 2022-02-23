#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

DATE=$(date +%Y-%m-%d)
SALT=$(openssl rand -base64 6)
echo "${DATE}_${SALT}"

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --suffix "${DATE}_${SALT}" ${@:3}
