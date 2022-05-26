# TSIT: A Simple and Versatile Framework for Image-to-Image Translation

based on [EndlessSora/TSIT](https://github.com/EndlessSora/TSIT)

## Installation

Following [original code], or just

```
git clone https://github.com/trngpg/TSIT.git
cd TSIT
conda env create -f conda_env.yml
```

The code requires batch-norm (consult original code)
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
rm -rf Synchronized-BatchNorm-PyTorch
cd ../../
```

## Tasks and Datasets

We are going to train TSIT on day2night unpaired data. The two datasets in used are BDD100K and VinAI's Scale.

### Dataset Preparation

1. Get BDD100K data to your machine
   ```
   MYHOME=lustre/scratch/client/vinai/users/trungpq3/datasets
   cp /vinai-public-dataset/BDD100K/bdd100k_images_100k.zip $MYHOME
   cd $MYHOME && unzip bdd100k_images_100k.zip
   ```
   After unzipping, there is a folder `bdd100k` with the following structure:
   ```
   ./bdd100k/images/100k/{train-val-test}/{image_files}
   ```
2. Get Scale data
   ```
   cp /vinai-autopilot/data/lane/scale_combined $MYHOME
   ```
   After unzipping, the `scale_combined` folder has the following structure:
   ```
   ./scale_combined/train_672w/images/{Downtown-Highway-Rural-Sub-urban}/{Clear-CLoudy-Night-etc}/{image_files}
   ```
These data will be read by data loader in `./data/scale_day2night_dataset.py`

### Training

Training file is `run_train_superpod.sh`.
1. First, provide bdd and scale database paths
2. Run with
   ```bash run_train_superpod.sh```

To train with `sbatch` on superpod (submit queued job), see `run_sbatch_tsit.sh`
