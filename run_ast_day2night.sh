#!/usr/bin/env bash

set -x

NAME='ast_day2night'
TASK='AST'
DATA='day2night'

# Downtown day2night experiments
# CROOT='../data/val_Downtown/content'
# SROOT='../data/val_Downtown/style'
# RESROOT='../output/TSIT_day2night/Downtown'

# fisheye timecity day2night experiments
# CROOT='../data/val_timecity_fisheye/content'
# SROOT='../data/val_timecity_fisheye/style'
# RESROOT='../output/TSIT_day2night/timecity_fisheye'

# fisheye MMIS
# CROOT='../data/val_timecity_fisheye_weather_1024/content'
# SROOT='../data/val_timecity_fisheye_weather_1024/style'
# RESROOT='../output/TSIT_day2night/timecity_fisheye_weather_1024'

# pinhole timecity day2night experiments
# CROOT='../data/val_timecity_pinhole/content'
# SROOT='../data/val_timecity_pinhole/style'
# RESROOT='../output/TSIT_day2night/timecity_pinhole'

# fisheye day2night with both fisheye content and fisheye style
CROOT="/home/ubuntu/trungpq3/data/val_timecity_fisheye_2_dim_light/content"
SROOT="/home/ubuntu/trungpq3/data/val_timecity_fisheye_2_dim_light/style"
RESROOT='../output/svm_low_light_0518/timecity2dimlight'


# CROOT="/home/ubuntu/trungpq3/data/svm_low_light_0518/val_normal/"
# SROOT="/home/ubuntu/trungpq3/data/svm_low_light_0518/val_normal_like/"
# RESROOT='../output/svm_low_light_0518/normal2normallike'

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
