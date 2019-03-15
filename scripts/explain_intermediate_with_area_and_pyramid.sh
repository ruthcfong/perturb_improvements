#!/usr/bin/env bash
image="images/flute.jpg"
lr="1e-1"
tv_lambda="0."
area_lambda="1e3"
l1_lambda="0."
area="0.05"
ease_in_area=1
ease_rate="1.1"
gpu=1
area_delay=100
loss="dual"
dimension="channel"
interpolate=1
debug=1
layer="features.11"
use_softmax=0
mask_size=28
epochs=500
noise="0.05"
perturbation="intensity"
num_perturbations=10

python3 explain.py --learning_rate ${lr} --tv_lambda ${tv_lambda} \
    --area_lambda ${area_lambda} --l1_lambda ${l1_lambda} \
    --area ${area} --ease_in_area ${ease_in_area} --ease_rate ${ease_rate} \
    --image ${image} \
    --debug ${debug} \
    --area_delay ${area_delay} \
    --layer ${layer} \
    --dimension ${dimension} \
    --loss ${loss} \
    --interpolate ${interpolate} \
    --use_softmax ${use_softmax} \
    --gpu ${gpu} \
    --mask_size ${mask_size} \
    --epochs ${epochs} \
    --noise ${noise} \
    --perturbation ${perturbation} \
    --num_perturbations ${num_perturbations}


