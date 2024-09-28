#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
#input_path="$current_dir"/../../../../data
input_path="/data/zyk/data/dataset/"
output_path="$current_dir"/../../../output/
seed=42
batch_size=32
lrs=(0.1)
#lambdas=(0.1 0.3 0.7 0.9)
lambdas=(0.5)
gammas=(0 0.5 0.9)
max_epoch=1
communication_round=50

for lr in "${lrs[@]}"; do
    for lambda in "${lambdas[@]}"; do
      for gamma in "${gammas[@]}"; do
      if [ "$lambda" == 0.1 ] && [ "$gamma" == 0 ]; then
        continue
      fi
        case_name="fedsm-batch_size=${batch_size}-lr=${lr}-lambda=${lambda}-gamma=${gamma}-seed=${seed}"
        python "$current_dir"/../../trainers/fedsm_ecg.py  --batch_size $batch_size \
                                                            --lr $lr \
                                                            --lambda_ $lambda \
                                                            --gamma $gamma \
                                                            --case_name $case_name \
                                                            --seed $seed \
                                                            --input_path "$input_path" \
                                                            --output_path "$output_path" \
                                                            --max_epoch $max_epoch \
                                                            --communication_round $communication_round
      done
  done
done