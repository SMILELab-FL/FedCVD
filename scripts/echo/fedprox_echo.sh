seeds=(42)
batch_sizes=(32)
mus=(0.01 0.1 1.0)
lrs=(0.1 0.01 0.03162 0.001 0.0001)
input_path=""
output_path=""
max_epoch=50

cd ../../trainers/

for seed in ${seeds[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for mu in ${mus[@]}; do
      for lr in ${lrs[@]}; do
          case_name="fedprox-model=unet-batch_size=${batch_size}-lr=${lr}-mu=${mu}-seed=${seed}"
          python fedprox_echo.py --lr "$lr" \
          --mu "$mu" \
          --case_name "$case_name" \
          --batch_size "$batch_size" \
          --seed "$seed" \
          --max_epoch "$max_epoch" \
          --input_path "$input_path" \
          --output_path "$output_path"
      done
    done
  done
done