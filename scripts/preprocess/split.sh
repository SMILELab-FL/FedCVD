#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi

base_path="$current_dir"/../../../../data
seed=2
sample_ratio=0.8
data_types=("ECG" "ECHO")

for data_type in "${data_types[@]}"; do
  input_path="$base_path"/"$data_type"/preprocessed
  python "$current_dir"/../../preprocess/splitter.py  --seed "$seed" \
                                                      --data_type "$data_type" \
                                                      --input_path "$input_path" \
                                                      --sample_ratio "$sample_ratio"
done