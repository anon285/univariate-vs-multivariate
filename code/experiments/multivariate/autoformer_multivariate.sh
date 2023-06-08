#!/usr/bin/env bash
val=1000
patience=20
num_layers=1
# Set to 50 if in univariate setting with forecast of 720 and less than 24GB
batch_size=100
experiment_batch_name=autoformer_multivariate
experiment_prepend=autoformer_multivariate
model_name=Autoformer

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
    --is_training 1 \
    --data custom \
    --model_id default \
    --model Autoformer \
    --d_model 512\
    --features M \
    --history_length $history_length \
    --label_len $label_len \
    --horizon_length $horizon_length \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --learning_rate 1e-4\
    --log\

}

# Weather (21 dims)
history_length=96
label_len=48
enc_in=21
dec_in=21
c_out=21
dataset=autoformer_dataloader_weather
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_weather_${history_length}_${horizon_length}"
    run_experiment
done



# Traffic (862 dims)
history_length=96
label_len=48
enc_in=862
dec_in=862
c_out=862
dataset=autoformer_dataloader_traffic
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_traffic_${history_length}_${horizon_length}"
    run_experiment
done


# Illness (7 dims)
history_length=36
label_len=18
enc_in=7
dec_in=7
c_out=7
dataset=autoformer_dataloader_illness
for horizon_length in 24 36 48 60; do
    experiment="${experiment_prepend}_illness_${history_length}_${horizon_length}"
    run_experiment
done


# Electricity (321 dims)
history_length=96
label_len=48
enc_in=321
dec_in=321
c_out=321
dataset=autoformer_dataloader_electricity
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_electricity_${history_length}_${horizon_length}"
    run_experiment
done

# Exchange (8 dims)
history_length=96
label_len=48
enc_in=8
dec_in=8
c_out=8
dataset=autoformer_dataloader_exchange
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_exchange_${history_length}_${horizon_length}"
    run_experiment
done


# ETDm2 (7 dims)
history_length=96
label_len=48
enc_in=7
dec_in=7
c_out=7
dataset=autoformer_dataloader_ettm2
for horizon_length in 96 192 336 720; do
    experiment="${experiment_prepend}_ettm2_${history_length}_${horizon_length}"
    run_experiment
done
