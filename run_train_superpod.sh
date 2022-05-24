#!/usr/bin/env bash

set -x

NAME='ast_day2night'
TASK='AST'
DATA='scale_day2night'

# Scale data
DATABASE='/vinai-autopilot/data/lane/scale_combined/train_672w/images/'
CROOT='/home/ubuntu/trungpq3/data/Downtown/Clear/'
SROOT='/home/ubuntu/trungpq3/data/Downtown/Night/'

# Dataroot for training
DATADIR='./datasets'
SCALE_CROOT=${DATADIR}/scale/Clear
SCALE_SROOT=${DATADIR}/scale/Night

mkdir -p ${DATADIR}/scale/
cp -r ${CROOT} ${SCALE_CROOT}
cp -r ${SROOT} ${SCALE_SROOT}

CKPTROOT='./checkpoints'
WORKER=4

python3 train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 5 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 2 \
    --niter 10 \
    --lambda_vgg 2 \
    --lambda_feat 1 \
    --scale_croot ${SCALE_CROOT} \
    --scale_sroot ${SCALE_SROOT} \

