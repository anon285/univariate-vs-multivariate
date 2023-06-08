#!/usr/bin/env bash
val=50
patience=20
num_layers=1
batch_size=100
experiment_prepend=independence_univariate
model_name=VanillaTransformer
model_dim=64

# univariate_multi_model equals the first argument passed to this script.
function run_experiment(){
    python main.py \
    --experiment $experiment \
    --experiment_batch_name $experiment_prepend\
    --model_name $model_name \
    --num_layers $num_layers \
    --model_dim $model_dim \
    --batch_size $batch_size \
    --learning_rate 1e-4\
    --log \
    --validate_every $val \
    --log_every_n_steps 1 \
    --patience $patience\
    --dataset $dataset \
    --history_length $history_length \
    --horizon_length $horizon_length \
    --merge_splits \
    --disable_progress_bar \
    --independence $independent\
    --univariate \

}
# Is merge splits still needed in univariate mode?  I would have thought so.

history_length=96
dataset=synthetic
for independent in 0 0.5 1; do
    for horizon_length in 720; do
        experiment="${experiment_prepend}_${history_length}_${horizon_length}_synthetic_${independent}"
        run_experiment  
    done
done
