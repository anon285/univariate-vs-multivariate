#!/usr/bin/env bash
val=1000
patience=20
num_layers=1
# Set to 50 if in univariate setting with forecast of 720 and less than 24GB
batch_size=100
experiment_batch_name=fedformer_univariate
experiment_prepend=fedformer_univariate
model_name=FEDformer

function run_experiment(){
    python main.py \
    --experiment $experiment \
    --model_name $model_name \
    --batch_size $batch_size \
    --validate_every $val \
    --log_every_n_steps 1 \
    --dataset $dataset \
    --patience $patience\
    --experiment_batch_name $experiment_batch_name\
    --data custom \
    --model FEDformer \
    --d_model 512\
    --features M \
    --history_length $history_length \
    --label_len $label_len \
    --horizon_length $horizon_length \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --learning_rate 1e-4\
    --merge_splits\
    --univariate\
    --log\

}

# Weather (21 dims)
history_length=96
label_len=48
dataset=fedformer_dataloader_weather
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_weather_${history_length}_${horizon_length}"
    run_experiment
done



# Traffic (862 dims)
history_length=96
label_len=48
dataset=fedformer_dataloader_traffic
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_traffic_${history_length}_${horizon_length}"
    run_experiment
done


# Illness (7 dims)
history_length=36
label_len=18
dataset=fedformer_dataloader_illness
for horizon_length in 24 36 48 60; do
    experiment="${experiment_prepend}_illness_${history_length}_${horizon_length}"
    run_experiment
done


# Electricity (321 dims)
history_length=96
label_len=48
dataset=fedformer_dataloader_electricity
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_electricity_${history_length}_${horizon_length}"
    run_experiment
done

# Exchange (8 dims)
history_length=96
label_len=48
dataset=fedformer_dataloader_exchange
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_exchange_${history_length}_${horizon_length}"
    run_experiment
done


# ETDm2 (7 dims)
history_length=96
label_len=48
dataset=fedformer_dataloader_ettm2
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_ettm2_${history_length}_${horizon_length}"
    run_experiment
done


