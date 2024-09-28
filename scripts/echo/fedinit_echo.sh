#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
#input_path="$current_dir"/../../../../data
input_path="/data/zyk/data/dataset/"
output_path="$current_dir"/../../../output/echo
seed=42
batch_size=32
betas=(0.01 0.05 0.1)
lrs=(0.1 0.03162 0.01 0.001 0.0001)
max_epoch=1
communication_round=50

for lr in "${lrs[@]}"; do
    for beta in "${betas[@]}"; do
      case_name="fedinit-batch_size=${batch_size}-lr=${lr}-beta=${beta}-seed=${seed}"
      python "$current_dir"/../../trainers/fedinit_echo.py  --lr "$lr" \
                                                          --beta "$beta" \
                                                          --case_name "$case_name" \
                                                          --batch_size "$batch_size" \
                                                          --seed "$seed" \
                                                          --max_epoch "$max_epoch" \
                                                          --input_path "$input_path" \
                                                          --output_path "$output_path" \
                                                          --communication_round "$communication_round"
  done
done