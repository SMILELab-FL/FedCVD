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
seeds=(3 9 15 55)
batch_size=32
lambdas=(0.1)
gammas=(0)
#lambdas=(0.1 0.3 0.7 0.9)
#gammas=(0 0.5 0.9)
#lrs=(0.1 0.03162 0.01 0.001 0.0001)
lrs=(0.1)
max_epoch=1
communication_round=50

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
      for lambda in "${lambdas[@]}"; do
        for gamma in "${gammas[@]}"; do
          case_name="fedsm-batch_size=${batch_size}-lr=${lr}-lambda=${lambda}-gamma=${gamma}-seed=${seed}"
          python "$current_dir"/../../trainers/fedsm_echo.py  --lr "$lr" \
                                                              --lambda_ "$lambda" \
                                                              --gamma "$gamma" \
                                                              --case_name "$case_name" \
                                                              --batch_size "$batch_size" \
                                                              --seed "$seed" \
                                                              --max_epoch "$max_epoch" \
                                                              --input_path "$input_path" \
                                                              --output_path "$output_path" \
                                                              --communication_round "$communication_round"
        done
    done
  done
done