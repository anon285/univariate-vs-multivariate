#!/usr/bin/env bash
### Test how training data size impacts loss
val=1000
patience=20
num_layers=1
# Set to 50 if in univariate setting with forecast of 720 and less than 24GB
batch_size=100
experiment_batch_name=size_requirements_vt_univariate
experiment_prepend=size_requirements_vt_univariate
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
    --univariate \
    --merge_splits \
    --crop_data $crop \
    --disable_progress_bar\

}

dataset=autoformer_ETD_m2
horizon_length=96
history_length=96
for crop in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.984 0.993 0.995; do
    experiment="${experiment_prepend}_ett_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done

dataset=autoformer_exchange_rate
horizon_length=96
history_length=96
for crop in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.94 0.96; do
    experiment="${experiment_prepend}_exchange_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done


dataset=autoformer_illness
horizon_length=36
history_length=24
for crop in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.87; do
    experiment="${experiment_prepend}_illness_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done


dataset=autoformer_traffic
horizon_length=96
history_length=96
for crop in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.97 0.983; do
    experiment="${experiment_prepend}_traffic_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done


dataset=autoformer_weather
horizon_length=96
history_length=96
for crop in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.97 0.992 0.994; do
    experiment="${experiment_prepend}_weather_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done


dataset=autoformer_electricity
horizon_length=96
history_length=96
for crop in 0.9895 0.989 0.9885 0.988 0.9875 0.987 0.986 0.985 0.984 0.982 0.98 0.975 0.97 0.96 0.95 0.9 0.85 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0; do
    experiment="${experiment_prepend}_electricity_${history_length}_${horizon_length}_crop_${crop}"
    run_experiment
done
