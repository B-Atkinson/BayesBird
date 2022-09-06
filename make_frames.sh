#!/bin/bash
#SBATCH --mem=3G
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:00:00

. /etc/profile

module load lang/miniconda3/4.10.3
module load lib/cuda/11.3

source activate torch

python make_movie.py
