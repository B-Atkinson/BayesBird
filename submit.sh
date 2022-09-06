#!/bin/bash
#SBATCH --mem=2G
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --time=01-12:00:00

. /etc/profile

module load lang/miniconda3/4.10.3
module load lib/cuda/11.3

source activate torch



python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM_EPS \
--human_decay=$HDECAY \
--human_influence=$INF
