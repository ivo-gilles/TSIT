#!/bin/bash
#SBATCH --job-name=TSIT
#SBATCH --output=/lustre/scratch/client/vinai/users/trungpq3/TSIT_sbatch_outputs/output%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/trungpq3/TSIT_sbatch_outputs/output%A.err
#SBATCH --partition=applied
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-gpu=19GB
#SBATCH --cpus-per-task=16

export HTTP_PROXY=http://proxytc.vingroup.net:9090/
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
export http_proxy=http://proxytc.vingroup.net:9090/
export https_proxy=http://proxytc.vingroup.net:9090/

export MYHOME=/lustre/scratch/client/vinai/users/trungpq3

# Get codebase

# git clone https://github.com/trngpg/TSIT
cp -r $MYHOME/TSIT/ .
cd ./TSIT/
git checkout dev

# Setting up environment
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"

# conda env create -f conda_env.yml
# if there is Pillow error, reinstall pillow 6.1.0
# conda remove pillow
# conda install pillow=6.1.0

conda activate tsit

# run experiment
time bash run_train_superpod.sh

echo "DONE TRAINING"
