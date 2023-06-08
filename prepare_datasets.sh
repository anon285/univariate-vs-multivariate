#!/usr/bin/env bash
set -e

echo extracting dataset CSVs...
bsdtar xf all_six_datasets.zip
echo Cleaning up...
rm __MACOSX -r

# Copy the csv files to their correct locations
echo Moving CSVs...
mv all_six_datasets/electricity/*.csv   data/autoformer/electricity/
mv all_six_datasets/ETT-small/*.csv     data/autoformer/ETT-small/
mv all_six_datasets/exchange_rate/*.csv data/autoformer/exchange_rate/
mv all_six_datasets/illness/*.csv       data/autoformer/illness/
mv all_six_datasets/traffic/*.csv       data/autoformer/traffic/
mv all_six_datasets/weather/*.csv       data/autoformer/weather/


# Generate pickle datasets
echo Preparing electricity dataset...
cd data/autoformer/electricity/
python prep_electricity.py
cd ../../..
echo Preparing ETT dataset...
cd data/autoformer/ETT-small/
python prep_etd.py
cd ../../..
echo Preparing exchange_rate dataset...
cd data/autoformer/exchange_rate/
python prep_exchange_rate.py
cd ../../..
echo Preparing illness dataset...
cd data/autoformer/illness/
python prep_illness.py
cd ../../..
echo Preparing traffic dataset...
cd data/autoformer/traffic/
python prep_traffic.py
cd ../../..
echo Preparing weather dataset...
cd data/autoformer/weather/
python prep_weather.py
cd ../../..

# Generate synthetic dataset
echo Generating synthetic dataset...
cd data/synthetic
python generate_synthetic.py
cd ../..

# Cleanup
echo Cleaning up...
rm all_six_datasets -r

echo
echo ---Done---
