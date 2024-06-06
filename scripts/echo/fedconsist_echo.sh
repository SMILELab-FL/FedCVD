seeds=(42)
batch_sizes=(32)
taus=(0.9 0.7 0.5)
lrs=(0.01 0.001 0.0001)
unlabeled_lrs=(0.001 0.0001 0.00001 0.000005 0.000001)

input_path=""
output_path=""
max_epoch=1
communication_round=50

cd ../../trainers/

for seed in ${seeds[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for tau in ${taus[@]}; do
      for lr in ${lrs[@]}; do
        for unlabeled_lr in ${unlabeled_lrs[@]}; do
          # skip if unlabeled_lr >= lr
          if [ $(echo "$unlabeled_lr > $lr" | bc) -eq 1 ]; then
            continue
          fi
          case_name="fedconsist-model=unet-batch_size=${batch_size}-lr=${lr}-unlabeled_lr=${unlabeled_lr}-tau=${tau}-seed=${seed}"
          python fedconsist_echo.py --lr "$lr" \
          --unlabeled_lr "$unlabeled_lr" \
          --tau "$tau" \
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
done