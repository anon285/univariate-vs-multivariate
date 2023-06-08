#!/usr/bin/env bash
val=1000
patience=20
num_layers=1
# Set to 50 if in univariate setting with forecast of 720 and less than 24GB
batch_size=100
experiment_batch_name=vt_multivariate
experiment_prepend=vt_multivariate
model_name=VanillaTransformer
model_dim=64


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
    --merge_splits \

}
# Is merge splits still needed in univariate mode?  I would have thought so.

# Weather (21 dims)
history_length=96
dataset=autoformer_weather
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_weather_${history_length}_${horizon_length}"
    run_experiment
done



# Traffic (862 dims)
history_length=96
dataset=autoformer_traffic
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_traffic_${history_length}_${horizon_length}"
    run_experiment
done


# Illness (7 dims)
history_length=36
dataset=autoformer_illness
for horizon_length in 24 36 48 60; do
    experiment="${experiment_prepend}_illness_${history_length}_${horizon_length}"
    run_experiment
done


# Electricity (321 dims)
history_length=96
dataset=autoformer_electricity
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_electricity_${history_length}_${horizon_length}"
    run_experiment
done

# Exchange (8 dims)
history_length=96
dataset=autoformer_exchange_rate
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_exchange_${history_length}_${horizon_length}"
    run_experiment
done


# ETDm2 (7 dims)
history_length=96
dataset=autoformer_ETD_m2
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_ettm2_${history_length}_${horizon_length}"
    run_experiment
done
