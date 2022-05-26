#!/usr/bin/env bash

set -x

NAME='ast_day2night'
TASK='AST'
DATA='scale_day2night'

# Database, change to where you keep data
BDD_DATABASE=/lustre/scratch/client/vinai/users/trungpq3/datasets/bdd100k
SCALE_DATABASE=/lustre/scratch/client/vinai/users/trungpq3/datasets/scale_combined

# Dataroot for training
HOME=$(pwd)
DATADIR=$HOME/datasets
SCALE_ROOT=${DATADIR}/scale_combined
BDD_ROOT=${DATADIR}/bdd100k

# Get BDD datasets
ln -s ${BDD_DATABASE} ${DATADIR}
cp -r ${DATADIR}/bdd100k_lists/ ${BDD_ROOT}/

# Get Scale data
ln -s ${SCALE_DATABASE} ${DATADIR}
# for name in 'Downtown'  'Highway'  'Rural'  'Sub-urban'
# do
#     cp -r $DATABASE/$name/Clear/* ${SCALE_CROOT}
#     cp -r $DATABASE/$name/Cloudy/* ${SCALE_CROOT}
#     cp -r $DATABASE/$name/Night/* ${SCALE_SROOT}
# done

CKPTROOT='./checkpoints'
WORKER=4
# CROOT='/home/ubuntu/trungpq3/data/Downtown/Clear/'
# SROOT='/home/ubuntu/trungpq3/data/Downtown/Night/'

python3 train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
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
    --use_bdd100k \
    --scale_root ${SCALE_ROOT} \
    --bdd_root ${BDD_ROOT} \

