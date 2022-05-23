#!/usr/bin/env bash

set -x

NAME='ast_summer2winteryosemite'
TASK='AST'
DATA='summer2winteryosemite'

# fisheye timecity day2night experiments
CROOT='/home/ubuntu/trungpq3/data/val_timecity_fisheye_weather_1024/content/'
SROOT='/home/ubuntu/trungpq3/data/val_timecity_fisheye_weather_1024/style/'
RESROOT='/home/ubuntu/trungpq3/output/TSIT_day2night/timecity_fisheye_weather_1024/'

CKPTROOT='./checkpoints'
WORKER=4
EPOCH='latest'

CUDA_VISIBLE_DEVICES=6 python3 test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH \
    --show_input
