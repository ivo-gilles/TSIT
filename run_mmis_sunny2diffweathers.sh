#!/usr/bin/env bash

set -x

NAME='mmis_sunny2diffweathers'
TASK='MMIS'
DATA='sunny2diffweathers'

# Downtown day2night experiments
# CROOT='../data/val_Downtown/content'
# SROOT='../data/val_Downtown/style'
# RESROOT='../output/TSIT_day2night/Downtown'

# fisheye timecity day2night experiments
CROOT='../data/val_timecity_fisheye_weather_1024/content'
SROOT='../data/val_timecity_fisheye_weather_1024/style'
RESROOT='../output/TSIT_day2night/timecity_fisheye_weather_1024/night'

# pinhole timecity day2night experiments
# CROOT='../data/val_timecity_pinhole/content'
# SROOT='../data/val_timecity_pinhole/style'
# RESROOT='../output/TSIT_day2night/timecity_pinhole'

CKPTROOT='./checkpoints'
WORKER=4
EPOCH='latest'
MODE=${1:-'night'}

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
    --show_input \
    --test_mode 'night'
    # --test_mode $MODE
