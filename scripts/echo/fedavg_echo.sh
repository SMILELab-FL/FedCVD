seeds=(42)
batch_sizes=(32)
lrs=(0.1 0.01 0.03162 0.001 0.0001)
input_path=""
output_path=""
max_epoch=1
communication_round=50

cd ../../trainers/

for seed in ${seeds[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for lr in ${lrs[@]}; do
      case_name="fedavg-model=unet-batch_size=${batch_size}-lr=${lr}-seed=${seed}"
      python fedavg_echo.py --lr "$lr" \
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