seeds=(42)
models=("unet")
batch_sizes=(32)
lrs=(0.1 0.03162 0.01 0.001 0.0001)
input_path=""
output_path=""
max_epoch=50

cd ../../trainers/

for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr in "${lrs[@]}"; do
        case_name="centralized-model=${model}-batch_size=${batch_size}-lr=${lr}"
        python local_echo.py \
          --case_name "$case_name" \
          --batch_size "$batch_size" \
          --model "$model" \
          --lr "$lr" \
          --seed "$seed" \
          --max_epoch "$max_epoch" \
          --input_path "$input_path" \
          --output_path "$output_path"
      done
    done
  done
done