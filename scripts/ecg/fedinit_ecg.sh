#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
#input_path="$current_dir"/../../../../data
input_path="/data/zyk/data/dataset/"
output_path="$current_dir"/../../../../output
seed=42
batch_size=32
lr=0.1
beta=0.01
max_epoch=1
communication_round=50

case_name="fedinit-batch_size=${batch_size}-lr=${lr}-beta=${beta}-seed=${seed}"
python "$current_dir"/../../trainers/fedinit_ecg.py  --batch_size $batch_size \
                                                    --lr $lr \
                                                    --beta $beta \
                                                    --case_name $case_name \
                                                    --seed $seed \
                                                    --input_path "$input_path" \
                                                    --output_path "$output_path" \
                                                    --max_epoch $max_epoch \
                                                    --communication_round $communication_round