#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
input_path="$current_dir"/../../../../data
output_path="$current_dir"/../../../../data
dataset_names=("sph" "ptb" "sxph" "g12ec" "camus" "echonet" "hmcqu")

for dataset_name in "${dataset_names[@]}"; do
  python "$current_dir"/../../preprocess/preprocessor.py --dataset_name "$dataset_name" \
                                        --input_path "$input_path" \
                                        --output_path "$output_path"
done