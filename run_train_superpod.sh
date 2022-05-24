#!/usr/bin/env bash

set -x

NAME='ast_day2night'
TASK='AST'
DATA='scale_day2night'

# Dataroot for training
HOME=$(pwd)
DATADIR=$HOME/datasets
SCALE_CROOT=${DATADIR}/scale/Day
SCALE_SROOT=${DATADIR}/scale/Night
BDD_CROOT=${DATADIR}/bdd100k
BDD_SROOT=${DATADIR}/bdd100k

mkdir -p ${SCALE_CROOT} ${SCALE_SROOT} ${BDD_CROOT} ${BDD_SROOT}

# Get BDD datasets
cp /vinai-public-dataset/BDD100K/bdd100k_images_100k.zip ${DATADIR}
cd ${DATADIR} && unzip bdd100k_images_100k.zip
cp -r ${DATADIR}/bdd100k_lists/ ${BDD_CROOT}/
cd ..

# Get Scale data
DATABASE='/vinai-autopilot/data/lane/scale_combined/train_672w/images/'

for name in 'Downtown'  'Highway'  'Rural'  'Sub-urban'
do
    cp -r $DATABASE/$name/Clear/* ${SCALE_CROOT}
    cp -r $DATABASE/$name/Cloudy/* ${SCALE_CROOT}
    cp -r $DATABASE/$name/Night/* ${SCALE_SROOT}
done

CKPTROOT='./checkpoints'
WORKER=4
# CROOT='/home/ubuntu/trungpq3/data/Downtown/Clear/'
# SROOT='/home/ubuntu/trungpq3/data/Downtown/Night/'

python3 train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 5 \
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
    --scale_croot ${SCALE_CROOT} \
    --scale_sroot ${SCALE_SROOT} \
    --bdd_croot ${BDD_CROOT} \
    --bdd_sroot ${BDD_SROOT} \

