#!/usr/bin/env bash
image="images/flute.jpg"
lr="1e-1"
tv_lambda="1e1"
area_lambda="1e1"
l1_lambda="0."
area="0.98"
ease_in_area=1
ease_rate="2."
python3 explain.py --learning_rate ${lr} --tv_lambda ${tv_lambda} \
    --area_lambda ${area_lambda} --l1_lambda ${l1_lambda} \
    --area ${area} --ease_in_area ${ease_in_area} --ease_rate ${ease_rate} \
    --image ${image}