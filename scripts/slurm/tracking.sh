#!/bin/bash
#SBATCH --job-name=locust_track
#SBATCH --partition=wzhdtest
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/track_%j.log

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/24.04
export LD_LIBRARY_PATH=$HOME/openssl11/lib:$LD_LIBRARY_PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate locust

python scripts/detect_oviposition.py \
    --weights runs/detect/runs/v2/weights/best.pt \
    --video-dir video/Training_Set \
    --stages stages.csv \
    --cache-tracks track_cache \
    --device 0 \
    --win-sec 180 --frag-min-duration 900 \
    --stitch-gap 60 --stitch-dist 100 \
    --out results.csv
