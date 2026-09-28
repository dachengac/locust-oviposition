#!/bin/bash
#SBATCH --job-name=locust_train
#SBATCH --partition=wzhdtest
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.log

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/24.04
export LD_LIBRARY_PATH=$HOME/openssl11/lib:$LD_LIBRARY_PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate locust

yolo detect train \
    model=yolo11s.pt \
    data=dataset/locust.yaml \
    epochs=100 imgsz=640 batch=32 \
    patience=30 project=runs name=v2 device=0 workers=8
