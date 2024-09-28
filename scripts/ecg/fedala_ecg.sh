#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
#input_path="$current_dir"/../../../../data
input_path="/data/zyk/data/dataset/"
output_path="$current_dir"/../../../output
seed=42
batch_size=32
lrs=(0.1)
rand_percents=(5 80 50)
layer_idx=1
eta=1.0
threshold=0.1
num_pre_loss=10
max_epoch=1
communication_round=50

for lr in "${lrs[@]}"; do
    for rand_percent in "${rand_percents[@]}"; do
      case_name="fedala-batch_size=${batch_size}-lr=${lr}-rand_percent=${rand_percent}-layer_idx=${layer_idx}-eta=${eta}-threshold=${threshold}-num_pre_loss=${num_pre_loss}-seed=${seed}"
      python "$current_dir"/../../trainers/fedala_ecg.py  --batch_size $batch_size \
                                                    --lr $lr \
                                                    --rand_percent $rand_percent \
                                                    --layer_idx $layer_idx \
                                                    --eta $eta \
                                                    --threshold $threshold \
                                                    --num_pre_loss $num_pre_loss \
                                                    --case_name $case_name \
                                                    --seed $seed \
                                                    --input_path "$input_path" \
                                                    --output_path "$output_path" \
                                                    --max_epoch $max_epoch \
                                                    --communication_round $communication_round
  done
done