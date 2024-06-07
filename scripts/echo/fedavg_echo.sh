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
max_epoch=1
communication_round=50

case_name="fedavg-batch_size=${batch_size}-lr=${lr}-seed=${seed}"
python "$current_dir"/../../trainers/fedavg_echo.py --lr "$lr" \
                                                    --case_name "$case_name" \
                                                    --batch_size "$batch_size" \
                                                    --seed "$seed" \
                                                    --max_epoch "$max_epoch" \
                                                    --input_path "$input_path" \
                                                    --output_path "$output_path" \
                                                    --communication_round "$communication_round"