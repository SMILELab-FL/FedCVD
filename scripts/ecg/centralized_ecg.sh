seeds=(42)
batch_sizes=(32)
lrs=(0.1 0.01 0.001 0.0001 0.00001)

input_path=""
output_path=""
max_epoch=50

cd ../../trainers/

for seed in ${seeds[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for lr in ${lrs[@]}; do
      python centralized_ecg.py --batch_size $batch_size --lr $lr --seed $seed --input_path $input_path --output_path $output_path --max_epoch $max_epoch
    done
  done
done