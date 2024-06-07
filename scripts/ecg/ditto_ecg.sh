#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
input_path="$current_dir"/../../../../data
output_path="$current_dir"/../../../../output
seed=42
batch_size=32
lr=0.1
mu=0.01
max_epoch=1
communication_round=50

python "$current_dir"/../../trainers/ditto_ecg.py --batch_size $batch_size \
                                                  --lr $lr \
                                                  --mu $mu \
                                                  --seed $seed \
                                                  --input_path "$input_path" \
                                                  --output_path "$output_path" \
                                                  --max_epoch $max_epoch \
                                                  --communication_round $communication_round