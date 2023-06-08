#!/usr/bin/env bash
val=1000
patience=20
num_layers=1
decoder=BasicDecoder
batch_size=100
experiment_batch_name=dimension_count
experiment_prepend=dimension_count
model_name=VanillaTransformer

function run_experiment(){
    python main.py \
        --experiment $experiment\
        --experiment_batch_name $experiment_batch_name\
        --model_name $model_name\
        --dataset $dataset\
        --num_layers $num_layers\
        --model_dim $model_dim\
        --history_length $history_length\
        --horizon_length $horizon_length\
        --batch_size $batch_size\
        --validate_every $val\
        --log_every_n_steps 1\
        --patience $patience\
        --log\
        --keep_dims $1\

}


# Weather (21 dims)
dataset=autoformer_weather
model_dim=64
history_length=96
horizon_length=96
for keep_dims in 1 2 3 4 5 8 10 12 15 20 21; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done



# Traffic (862 dims)
model_dim=8
history_length=96
horizon_length=96
dataset=autoformer_traffic
for keep_dims in 1 2 3 4 5 10 15 20 30 50 80 100 200 400 600 800 862; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done


# Illness (7 dims)
model_dim=64
history_length=36
horizon_length=24
dataset=autoformer_illness
for keep_dims in 1 2 3 4 5 6 7; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done


# Electricity (321 dims)
model_dim=24
history_length=96
horizon_length=96
dataset=autoformer_electricity
for keep_dims in 1 2 3 4 5 6 8 10 15 20 30 40 50 70 100 150 200 250 300 321; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done

# Exchange (8 dims)
model_dim=64
history_length=96
horizon_length=96
dataset=autoformer_exchange_rate
for keep_dims in 1 2 3 4 5 6 7 8; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done

# ETDm2 (7 dims)
model_dim=64
history_length=96
horizon_length=96
dataset=autoformer_ETD_m2
for keep_dims in 1 2 3 4 5 6 7; do
    experiment="${experiment_prepend}_${dataset}_${history_length}_${horizon_length}_KD-${keep_dims}"
    run_experiment $keep_dims
done
