#!/usr/bin/env bash
val=1000
patience=20
num_layers=1
# Set to 50 if in univariate setting with forecast of 720 and less than 24GB
batch_size=100
experiment_batch_name=trad_vt_illness
experiment_prepend=trad_vt_illness
model_name=VanillaTransformer
model_dim=64

# univariate_multi_model equals the first argument passed to this script.
function run_experiment(){
    python main.py \
    --experiment $experiment \
    --experiment_batch_name $experiment_batch_name\
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
    --univariate_multi_model $dim \
    --merge_splits \

}
# Is merge splits still needed in univariate mode?  I would have thought so.

# illness (7 dims)
history_length=36
dataset=autoformer_illness
for dim in $(seq 0 6); do
    for horizon_length in 24 36 48 60; do
        experiment="${experiment_prepend}_illness_${history_length}_${horizon_length}_d_${dim}"
        run_experiment
    done
done
