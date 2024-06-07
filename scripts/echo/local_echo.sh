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
max_epoch=50
case_name="centralized-batch_size=${batch_size}-lr=${lr}-seed=${seed}-max_epoch=${max_epoch}"
python "$current_dir"/../../trainers/local_echo.py  --case_name "$case_name" \
                                                    --batch_size "$batch_size" \
                                                    --lr "$lr" \
                                                    --seed "$seed" \
                                                    --max_epoch "$max_epoch" \
                                                    --input_path "$input_path" \
                                                    --output_path "$output_path"